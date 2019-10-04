import tensorflow as tf
import numpy as np

def get_rnn_cell(rnn_type, num_layers, hidden_dim, keep_prob, scope):
    with tf.variable_scope(scope):
        lst = []
        for _ in range(num_layers):
            if rnn_type == 'gru':
                cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            lst.append(cell)
        if num_layers > 1:
            res = tf.contrib.rnn.MultiRNNCell(lst)
        else:
            res = lst[0]
        return res

def rnn_state_shape(type, num_layers, shape):
    lst = []
    for _ in range(num_layers):
        if type == 'gru':
            lst.append(tf.TensorShape(shape))
        else:
            lst.append(tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape(shape), h=tf.TensorShape(shape)))
    if num_layers > 1:
        return tuple(lst)
    else:
        return lst[0]

def add_summary(summary_writer, global_step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)
    summary_writer.flush()

def restore_latest_model(model, best_checkpoint_dir, tmp_checkpoint_dir):
    tmp_latest_ckpt = tf.train.latest_checkpoint(tmp_checkpoint_dir)
    best_latest_ckpt = tf.train.latest_checkpoint(best_checkpoint_dir)
    if tmp_latest_ckpt is not None:
        tmp = eval(tmp_latest_ckpt.split("_")[1].split("-")[0])
    else:
        tmp = -1
    if best_latest_ckpt is not None:
        best = eval(best_latest_ckpt.split("_")[1].split("-")[0])
    else:
        best = -1
    start_epoch = tmp + 1 if tmp != -1 else 0
    model_dir = tmp_latest_ckpt
    if model_dir is None or (best != -1 and best > tmp):
        model_dir = best_latest_ckpt
        start_epoch = best + 1
    worse_step = tmp - best if best != -1 and tmp != -1 and tmp > best else 0
    if model_dir is not None:
        model.best_saver.restore(model.sess, model_dir)
    return start_epoch, worse_step, model

def restore_model(model, best_checkpoint_dir, tmp_checkpoint_dir):
    saver = model.best_saver
    latest_ckpt = tf.train.latest_checkpoint(best_checkpoint_dir)
    if latest_ckpt is None:
        saver = model.tmp_saver
        latest_ckpt = tf.train.latest_checkpoint(tmp_checkpoint_dir)
    if latest_ckpt is not None:
        saver.restore(model.sess, latest_ckpt)