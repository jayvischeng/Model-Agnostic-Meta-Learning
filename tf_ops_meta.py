from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from datetime import datetime

class timer:
    def __init__(self):
        self.lastTime = datetime.now()

    def delta(self, just_print=False):
        nowTime = datetime.now()
        delta_sec = (nowTime - self.lastTime).seconds
        self.lastTime = nowTime
        if (just_print):
            print("delta-time: " + str(delta_sec) + " s")
        return delta_sec


def vars(name, shape, initializer, GPU=False):
    if(os.path.exists('~/gpu')):
        GPU = True
    if GPU ==True:
        with tf.device('/gpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var
    else:
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky RELU """
    return tf.maximum(x, leak * x)

def sigmoid(x, name='sigmoid'):
    return tf.nn.sigmoid(x, name)

def tanh(x, name='tanh'):
    return tf.nn.tanh(x, name)

def batch_norm(x, scope=None, train=True ):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

def fc(x, hidden_units, scope='fc', flatten = False):
    with tf.variable_scope(scope):
        input_shape = x.get_shape().as_list()
        if flatten:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            x = tf.reshape(x, [-1, dim])
        else:
            dim = input_shape[1]

        weights_init = vars('init_weights', shape=[dim, hidden_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases_init = vars('init_biases', [hidden_units], tf.constant_initializer(0.0))

        weights_copy = vars('copy_weights', shape=[dim, hidden_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases_copy = vars('copy_biases', [hidden_units], tf.constant_initializer(0.0))

        return tf.add(tf.matmul(x, weights_init), biases_init)


def add_rnn_layer(name, data, cell_size, sequence_len, states, cell = tf.contrib.rnn.BasicRNNCell, layer_count = 1, output_keep = 1.0, re_use=False):
    with tf.variable_scope("rnn_" + name,reuse=re_use):
        my_cell = cell(cell_size)
        if(layer_count>1):
            my_cell = tf.nn.rnn_cell.MultiRNNCell([my_cell] * layer_count)
        if(output_keep < 1):
            my_cell = tf.contrib.rnn_cell.DropoutWrapper(my_cell, output_keep_prob = output_keep)
        data = tf.expand_dims(data, 1) #for remove
        outputs, states = tf.nn.dynamic_rnn(my_cell, data, sequence_length = sequence_len, time_major = True, initial_state = states, dtype=tf.float32)
        outputs = tf.reshape(outputs,[-1,cell_size])
        return outputs, states

def rnn(data,cell_size, sequence_len, states, cell, layer_count = 1, output_keep = 1.0,scope = "rnn"):
    with tf.variable_scope(scope):
        my_cell = cell
        if(layer_count>1):
            my_cell = tf.nn.rnn_cell.MultiRNNCell([my_cell] * layer_count)
        if(output_keep < 1):
            my_cell = tf.nn.rnn_cell.DropoutWrapper(my_cell, output_keep_prob = output_keep)
        data = tf.expand_dims(data, 1) #for remove
        outputs, states = tf.nn.dynamic_rnn(my_cell, data, sequence_length = sequence_len, time_major = True, initial_state = states, dtype=tf.float32)
        outputs = tf.reshape(outputs,[-1,cell_size])
        return outputs, states


def conv(inputs, kernel_size, stride, num_features, scope='conv'):
    """
      Args:
        inputs: nhwc
        kernel_size: int
        stride: int
        num_features: int
      Returns:
        outputs: nhwc
      """
    with tf.variable_scope(scope):
        input_channels = inputs.get_shape()[3]

        weights_init = vars('init_weights', shape=[kernel_size, kernel_size, input_channels, num_features],
                                             initializer= tf.truncated_normal_initializer(stddev=0.1))

        biases_init = vars('init_biases', [num_features], tf.constant_initializer(0.01))
        weights_copy = vars('weights', shape=[kernel_size, kernel_size, input_channels, num_features],
                       initializer=tf.truncated_normal_initializer(stddev=0.1))

        biases_copy = vars('biases', [num_features], tf.constant_initializer(0.01))


        conv = tf.nn.conv2d(inputs, weights_init, strides=[1, stride, stride, 1], padding='SAME')
        conv_biased = tf.nn.bias_add(conv, biases_init)
        return conv_biased

def deconv(inputs, kernel_size, stride, num_features ,scope='deconv'):
    """
    Args:
      inputs: nhwc
      kernel_shape: int
      strides: [stride_h, stride_w]
      num_features: int
    Returns:
      outputs: nhwc
    """
    with tf.variable_scope(scope):
        return tf.contrib.layers.convolution2d_transpose(inputs, num_features, [kernel_size, kernel_size],
                                                         [stride, stride],
                                                         padding='SAME',
                                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                         biases_initializer=tf.constant_initializer(0.0))

def gated_conv(inputs, state, kernel_size, scope='gated_conv'):
    """
    Args:
      inputs: nhwc
      state:  nhwc
      kernel_shape: [height, width]
    Returns:
      outputs: nhwc
      new_state: nhwc
    """
    with tf.variable_scope(scope):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        # state route

        left = conv(state, kernel_size, 1, 2 * in_channel, 'left')
        left1 = left[:, :, :, 0:in_channel]
        left2 = left[:, :, :, in_channel:]
        left1 = tf.nn.tanh(left1)
        left2 = tf.nn.sigmoid(left2)
        new_state = left1 * left2


        left2right = conv(left, kernel_size, 1, 2 * in_channel, 'left2right')
        # input route
        right = conv(inputs, kernel_size, 1, 2 * in_channel, 'right1')
        right = right + left2right
        right1 = right[:, :, :, 0:in_channel]
        right2 = right[:, :, :, in_channel:]
        right1 = tf.nn.tanh(right1)
        right2 = tf.nn.sigmoid(right2)
        up_right = right1 * right2

        up_right = conv(up_right, kernel_size, 1, in_channel, 'right2')
        outputs = inputs + up_right

        return outputs

def resnet_block(inputs, kernel_shape, stride, num_outputs, scope='resnet', train=True):
    """
    Args:
      inputs: nhwc
      num_outputs: int
      kernel_shape: [kernel_h, kernel_w]
    Returns:
      outputs: nhw(num_outputs)
    """
    with tf.variable_scope(scope):
        conv1 = conv(inputs, kernel_shape, stride, num_outputs, scope="res_conv1")
        bn1 = batch_norm(conv1, train=train, scope='bn1')
        relu1 = tf.nn.relu(bn1)
        conv2 = conv(relu1, kernel_shape, stride, num_outputs,scope="res_conv2")
        bn2 = batch_norm(conv2, train=train, scope='bn2')
        output = inputs + bn2

        return output


