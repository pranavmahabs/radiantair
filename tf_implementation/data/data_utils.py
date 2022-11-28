import random
import pickle
import numpy as np
import tensorflow as tf

M = 2**32 - 1

def init_fn(worker):
    #seed = torch.LongTensor(1).random_().item()
    seed = tf.random.uniform(shape=[], maxval=M, dtype=tf.int64)
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def sample(x, size):
    i = random.sample(range(x.shape[0]), size)
    return tf.Tensor(x[i], dtype=tf.int32)

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def get_all_coords(stride):
     return tf.Tensor(
         np.stack([v.reshape(-1) for v in
             np.meshgrid(
                     *[stride//2 + np.arange(0, s, stride) for s in _shape],
                     indexing='ij')],
             -1), dtype=tf.int16)


def gen_feats():
    x, y, z = 240, 240, 155
    feats = np.stack(
            np.meshgrid(
                np.arange(x), np.arange(y), np.arange(z),
                indexing='ij'), -1).astype('float32')
    shape = np.array([x, y, z])
    feats -= shape/2.0
    feats /= shape

    return feats