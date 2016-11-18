import rarfile
import threading
import tensorflow as tf
import unicodedata
import codecs
import fnmatch
import os

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
    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext):
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._txt_files = txt_files
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
        priorityQueue = PriorityQueue()
        for txt_file in self._txt_files:
          wav_file = os.path.splitext(txt_file)[0] + ".wav"
          wav_file_size = os.path.getsize(wav_file)
          priorityQueue.put((wav_file_size, (txt_file, wav_file)))
        files_list = []
        while not priorityQueue.empty():
            priority, (txt_file, wav_file) = priorityQueue.get()
            files_list.append((txt_file, wav_file))
        return cycle(files_list)

    def _populate_batch_queue(self, session):
        for txt_file, wav_file in self._files_circular_list:
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
                target = unicodedata.normalize("NFKD", open_txt_file.read().strip()).encode("ascii", "ignore")
                target = text_to_char_array(target)
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
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0):
    # Conditionally download data
    LAPS_DATA = "LapsBM1.4.rar"
    LAPS_DATA_URL = "http://www.laps.ufpa.br/falabrasil/files/LapsBM1.4.rar"
    local_file = base.maybe_download(LAPS_DATA, data_dir, LAPS_DATA_URL)

    # Conditionally extract data
    LAPS_DIR = "LapsBM1.4"
    _maybe_extract(data_dir, LAPS_DIR, local_file)

    # Conditionally split data into train/validation/test sets
    _maybe_split_sets(data_dir, LAPS_DIR, "laps-sets")
    
    # Create dev DataSet
    work_dir = os.path.join(data_dir, "laps-sets")
    dev = _read_data_set(work_dir, "dev", thread_count, dev_batch_size, numcep, numcontext, limit=limit_dev)

    # Create test DataSet
    test = _read_data_set(work_dir, "test", thread_count, test_batch_size, numcep, numcontext, limit=limit_test)

    # Create train DataSet
    train = _read_data_set(work_dir, "train", thread_count, train_batch_size, numcep, numcontext, limit=limit_train)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        rar = rarfile.RarFile(archive)
        rar.extractall(data_dir)
        rar.close()

def _maybe_split_sets(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    
    filelist = []
    
    # Python 2's glob doesn't support **, so we need os.walk
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.txt"):
            filelist.append(os.path.join(root, filename))
    
    filelist = sorted(filelist)
    
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))
    
    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg
    
    test_beg = dev_end
    test_end = len(filelist)
    
    _maybe_split_dataset(filelist[train_beg:train_end], os.path.join(target_dir, "train"))
    _maybe_split_dataset(filelist[dev_beg:dev_end], os.path.join(target_dir, "dev"))
    _maybe_split_dataset(filelist[test_beg:test_end], os.path.join(target_dir, "test"))
    
def _maybe_split_dataset(filelist, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        for txt_file in filelist:
            new_txt_file = os.path.join(target_dir, os.path.basename(txt_file))
            os.rename(txt_file, new_txt_file)
            
            wav_file = os.path.splitext(txt_file)[0] + ".wav"
            new_wav_file = os.path.join(target_dir, os.path.basename(wav_file))
            os.rename(wav_file, new_wav_file)

def _read_data_set(work_dir, data_set, thread_count, batch_size, numcep, numcontext, limit=0):
    # Create data set dir
    dataset_dir = os.path.join(work_dir, data_set)
    
    # Obtain list of txt files
    txt_files = glob(os.path.join(dataset_dir, "*.txt"))
    if limit > 0:
        txt_files = txt_files[:limit]
    
    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext)
