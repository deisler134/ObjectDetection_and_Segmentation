'''
Created on Jul. 29, 2019

@author: deisler
'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
import six

slim = tf.contrib.slim


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])

# TO DO: Move preprocessing into Tensorflow
def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)
    in_ = image[:, :, ::-1]
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # in_ = tf.subtract(tf.cast(in_, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    in_ = np.expand_dims(in_, axis=0)
    # in_ = tf.expand_dims(in_, 0)
    return in_

# TO DO: Move preprocessing into Tensorflow
def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label

def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    output = tf.nn.sigmoid(output)

    labels_pos = tf.cast(tf.greater(label, 0), tf.float32)
    labels_neg = tf.cast(tf.less(label, 1), tf.float32)

    num_labels_pos = tf.reduce_sum(labels_pos)
    num_labels_neg = tf.reduce_sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    loss_pos = tf.reduce_sum(tf.multiply(labels_pos, tf.log(output + 0.00001)))
    loss_neg = tf.reduce_sum(tf.multiply(labels_neg, tf.log(1 - output + 0.00001)))

    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg

    return final_loss




