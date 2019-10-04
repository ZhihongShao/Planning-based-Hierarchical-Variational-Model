import numpy as np
import os
import sys
import copy
import re
import tensorflow as tf
import logging

def select(dataset, selection):
	"""
	:param dataset : dictionary
	:param selection : list of index
	:return: batch
	"""
	batch = {key : [] for key in dataset}
	for idx in selection:
		idx = int(idx)
		for key in dataset:
			batch[key].append(dataset[key][idx])
	return batch

def print_out(msg):
	print(msg)
	sys.stdout.flush()
		
def add_summary(summary_writer, global_step, tag, value):
	summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
	summary_writer.add_summary(summary, global_step)
	summary_writer.flush()

def get_logger(filename):
	filename = "{}.log".format(filename)
	logger = logging.getLogger(filename)
	logger.setLevel(logging.INFO)

	log_path = "./logs"
	if not os.path.exists(log_path):
		os.mkdir(log_path)
	handler = logging.FileHandler(log_path + "/{}".format(filename))
	handler.setLevel(logging.INFO)

	fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
	datefmt = "%a %d %b %Y %H:%M:%S"
	formatter = logging.Formatter(fmt, datefmt)

	# add handler and formatter to logger
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger