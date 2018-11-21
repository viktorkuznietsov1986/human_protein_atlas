import tensorflow as tf
import keras.backend as K

def focal_loss(y_true, y_pred, gamma=2, alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def weighted_bce(args, outputs, targets):
    '!!!try this out'
    w = targets*args.pos_weight + 1  # default value of args.pos_weight is 20

    bce_loss = K.binary_cross_entropy_with_logits(outputs, targets, w)
    return bce_loss