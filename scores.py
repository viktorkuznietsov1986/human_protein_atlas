import keras.backend as K
import numpy as np

def f_score(y_true, logits, threshold=0.1, beta=1):
    y_pred = K.sigmoid(logits)

    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))


def tp_score(y_true, y_pred, threshold=0.1):

    tp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    return tp


def fp_score(y_true, y_pred, threshold=0.1):

    fp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=-1
    )

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    return fp


def fn_score(y_true, y_pred, threshold=0.1):

    fn_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(y_pred, K.constant(threshold)), 'float') - K.ones_like(y_pred)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    return fn

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    # need to rewrite it using Keras backend
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps) # sum along axis=0 (classes)
                                                            # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)    # sum along axis=0 (classes)
                                                            #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1) # we take the average of the individual f1 scores at the very end!
