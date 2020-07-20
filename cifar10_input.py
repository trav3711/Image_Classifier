import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

t = unpickle('/Users/Travis/Code/AI/Labs/deep_learning/cifar-10-batches-py/data_batch_1')

image = np.reshape(t[b'data'][336], (3, 32, 32)).transpose(1, 2, 0)
plt.imshow(image, interpolation="nearest")
plt.show()
