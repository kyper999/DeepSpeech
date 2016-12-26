import rarfile
import threading
import tensorflow as tf
import unicodedata
import codecs
import fnmatch
import os
import pandas

from glob import glob
from math import ceil
from itertools import cycle
from threading import Thread
from Queue import PriorityQueue
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse
from tensorflow.python.platform import gfile
from util.audio import audiofile_to_input_vector
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

    def start_queue_threads(self, session):
        self._dev.start_queue_threads(session)
        self._test.start_queue_threads(session)
        self._train.start_queue_threads(session)

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test

class DataSet(object):
    def __init__(self, filelist, thread_count, batch_size, numcep, numcontext):
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._filelist = filelist
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def start_queue_threads(self, session):
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()

    def _create_files_circular_list(self):
        # 1. Sort by wav filesize
        # 2. Select just wav filename and transcript columns
        # 3. Create a cycle
        return cycle(self._filelist.sort_values(by="wav_filesize")
                                   .ix[:, ["wav_filename", "transcript"]]
                                   .itertuples(index=False))

    def _populate_batch_queue(self, session):
        for wav_file, transcript in self._files_circular_list:
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            target = text_to_char_array(transcript)
            target_len = len(target)
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except (RuntimeError, tf.errors.CancelledError):
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_filelist) % _batch_size != 0, this re-uses initial files
        return int(ceil(float(len(self._filelist)) /float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0):
    # Read the processed set files from disk if they exist, otherwise create 
    # them.
    train_files = None
    train_csv = os.path.join(data_dir, "lapsbm-train.csv")
    if gfile.Exists(train_csv):
        train_files = pandas.read_csv(train_csv)

    dev_files = None
    dev_csv = os.path.join(data_dir, "lapsbm-dev.csv")
    if gfile.Exists(dev_csv):
        dev_files = pandas.read_csv(dev_csv)

    test_files = None
    test_csv = os.path.join(data_dir, "lapsbm-test.csv")
    if gfile.Exists(test_csv):
        test_files = pandas.read_csv(test_csv)

    if train_files is None or dev_files is None or test_files is None:
        print("Processed dataset files not found, downloading and processing data...")

        # Conditionally download data
        LAPS_DATA = "LapsBM1.4.rar"
        LAPS_DATA_URL = "http://www.laps.ufpa.br/falabrasil/files/LapsBM1.4.rar"
        local_file = base.maybe_download(LAPS_DATA, data_dir, LAPS_DATA_URL)

        # Conditionally extract data
        LAPS_DIR = "LapsBM1.4"
        _maybe_extract(data_dir, LAPS_DIR, local_file)

        # Aggregate transcripts into a Pandas DataFrame
        filelist = _aggregate_transcripts(data_dir, LAPS_DIR)

        # Split data into train/validation/test sets
        train_files, dev_files, test_files = _split_sets(filelist)

        # Write sets to disk as CSV files
        train_files.to_csv(train_csv, index=False)
        dev_files.to_csv(dev_csv, index=False)
        test_files.to_csv(test_csv, index=False)

    # Create train DataSet
    train = _create_data_set(train_files, thread_count, train_batch_size, numcep, numcontext, limit=limit_train)

    # Create dev DataSet
    dev = _create_data_set(dev_files, thread_count, dev_batch_size, numcep, numcontext, limit=limit_dev)

    # Create test DataSet
    test = _create_data_set(test_files, thread_count, test_batch_size, numcep, numcontext, limit=limit_test)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        rar = rarfile.RarFile(archive)
        rar.extractall(data_dir)
        rar.close()

def _aggregate_transcripts(data_dir, extracted_data):
    source_dir = os.path.join(data_dir, extracted_data)

    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        dirnames.sort()
        filenames.sort()
        for filename in fnmatch.filter(filenames, "*.txt"):
            full_filename = os.path.join(root, filename)
            wav_filename = os.path.splitext(full_filename)[0] + ".wav"
            wav_filesize = os.path.getsize(wav_filename)
            with codecs.open(full_filename, encoding="utf-8") as f:
                transcript = f.read().strip()
            transcript = unicodedata.normalize("NFKD", transcript).encode("ascii", "ignore")
            files.append((wav_filename, wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

def _split_sets(filelist):
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return filelist.iloc[train_beg:train_end], filelist.iloc[dev_beg:dev_end], filelist.iloc[test_beg:test_end]

def _create_data_set(filelist, thread_count, batch_size, numcep, numcontext, limit=0):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    # Return DataSet
    return DataSet(filelist, thread_count, batch_size, numcep, numcontext)
