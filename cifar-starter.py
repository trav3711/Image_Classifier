import numpy as np
import tensorflow as tf
import itertools

def unpickle(file):
    """Adapted from the CIFAR page: http://www.cs.utoronto.ca/~kriz/cifar.html"""
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Gather data
# data_dir = '/Users/drake/Documents/cifar-10-batches-py/' # My laptop
data_dir = '/home/users/drake/data/cifar-10-batches-py/' # BLT
train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in [1, 2, 3, 4]]
X_train = np.concatenate([t[b'data'] for t in train], axis=0)
y_train = np.array(list(itertools.chain(*[t[b'labels'] for t in train])))
valid = unpickle(data_dir + 'data_batch_5')
X_valid = valid[b'data']
y_valid = np.array(valid[b'labels'])

# Build network
n_inputs = 32 * 32 * 3
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    shaped = tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1))
    n_filters1 = 32
    conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')
    flat = tf.reshape(pool1, [-1, 16 * 16 * n_filters1])
    n_hidden1 = 1024
    hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.elu)
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")
