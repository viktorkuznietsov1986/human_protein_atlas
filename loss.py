import tensorflow as tf
import keras.backend as K


def focal_loss(y_true, logits, gamma=2, alpha=.25):
    y_pred = tf.nn.sigmoid(logits)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def weighted_binary_crossentropy(y_true, y_pred):
    false_positive_weight = 50
    thresh = 0.5
    y_pred_true = K.greater_equal(thresh,y_pred)
    y_not_true = K.less_equal(thresh,y_true)
    false_positive_tensor = K.equal(y_pred_true,y_not_true)


    #first let's transform the bool tensor in numbers - maybe you need float64 depending on your configuration
    false_positive_tensor = K.cast(false_positive_tensor,'float32')

    #and let's create it's complement (the non false positives)
    complement = 1 - false_positive_tensor

    #now we're going to separate two groups
    falsePosGroupTrue = y_true * false_positive_tensor
    falsePosGroupPred = y_pred * false_positive_tensor

    nonFalseGroupTrue = y_true * complement
    nonFalseGroupPred = y_pred * complement


    #let's calculate one crossentropy loss for each group
    falsePosLoss = K.binary_crossentropy(falsePosGroupTrue,falsePosGroupPred)
    nonFalseLoss = K.binary_crossentropy(nonFalseGroupTrue,nonFalseGroupPred)

    #return them weighted:
    return (false_positive_weight*falsePosLoss) + nonFalseLoss