import tensorflow as tf
import keras.backend as K

def focal_loss(y_true, y_pred, gamma=2, alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def weighted_bce(target, output, pos_weight=20):
    '!!!try this out'
    w = target*pos_weight + 1

    bce_loss = K.binary_cross_entropy(target, output, w)
    return bce_loss