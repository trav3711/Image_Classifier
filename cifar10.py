import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import itertools

"""unpickles data so it can be read by the program"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""function for showing individual images from the data set"""
def show_image():
    image_index = 5
    t = unpickle('/Users/Travis/Code/AI/Labs/deep_learning/cifar-10-batches-py/data_batch_1')
    image = np.reshape(t[b'data'][image_index], (3, 32, 32)).transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()

"""Collect and organize data into training and validation sets"""
#data_dir = '/users/Travis/code/ai/labs/deep_learning/cifar-10-batches-py/'
data_dir = '/home/users/traviscwelch/cifar-10-batches-py/'
train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in [1, 2, 3, 4]]
X_train = np.concatenate([t[b'data'] for t in train], axis=0)
y_train = np.array(list(itertools.chain(*[t[b'labels'] for t in train])))
valid = unpickle(data_dir + 'data_batch_5')
x_valid = valid[b'data']
y_valid = np.array(valid[b'labels'])
test = unpickle(data_dir + 'test_batch')
x_test = test[b'data']
y_test = np.array(test[b'labels'])

"""define variables for CNN"""
n_inputs = 32*32*3
n_hidden1 = 1024
n_hidden2 = 256
n_outputs = 10
learning_rate = 0.001
n_epochs = 50

x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x") #a 4D tensor
y = tf.placeholder(tf.int64, shape=(None), name="y")

"""convolutional neural network"""
with tf.name_scope("dnn"):
    shaped = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), (0, 2, 3, 1))#shaped data tensor
    n_filters1 = 32
    n_filters2 = 64
    n_filters3 = 128

    conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')

    conv2 = tf.layers.conv2d(pool1, n_filters2, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='valid')

    conv3 = tf.layers.conv2d(pool2, n_filters3, kernel_size=3, strides=1, padding='same',activation=tf.nn.elu)

    flat = tf.reshape(conv3, [-1, 8 * 8 * n_filters3])
    hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.relu)
    #hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

"""calculates and minimizes loss with softmax regression"""
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

"""trains data with gradient descent"""
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

"""evaluates"""
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def next_batch(size, index):
    x = X_train[index:index+size]
    y = y_train[index:index+size]
    return x, y

def plot_acc(x, train, val):
    fig = plt.fig()
    fig.suptitle('training and validation accuracy')
    axis = fig.add_subplot(111)
    axis.set_ylim(0,50)
    axis.set_xlim(0,1)

    plt.plot(x, train, c='r', label='Training Accuracy')
    plt.plot(x, val, c='g', label='Validation Accuracy')

    axis.set_xlabel('Epochs')
    axis.set_ylabel('Accuracy')
    axis.legend()

    plt.savefig('/home/users/traviscwelch/cifarplot.png')

"""runs the program as a tensorflow session"""
with tf.Session() as sess:
    init.run()
    batch_size = 500
    plt.xlim((0,50))
    plt.ylim((0,1))
    train_arr = []
    valid_arr = []
    epoch_array = list(range(1, 51))
    for epoch in range(n_epochs):
        for iteration in range(int(len(X_train)/batch_size)):
            x_batch, y_batch = next_batch(batch_size, iteration*batch_size)
            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={x: X_train, y: y_train})
        acc_val = accuracy.eval(feed_dict={x: x_valid, y: y_valid})
        print("epoch:", epoch+1, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
    acc_test  = accuracy.eval(feed_dict={x: x_test, y: y_test})
    print("Test accuracy: ", acc_test)
    plot_acc(epoch_array, train_arr, valid_arr)
    save_path = saver.save(sess, "/tmp/model.ckpt")
