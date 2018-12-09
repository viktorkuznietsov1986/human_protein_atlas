from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, \
    BatchNormalization, Dropout, regularizers
from keras import applications


def build_xception_classifier(input_shape, num_classes, l2_coeff=0.01):
    xception = applications.Xception(include_top=False, weights=None, input_shape=(input_shape[0], input_shape[1], 3))
    xception.trainable = True

    inputs = Input(shape=(input_shape[0], input_shape[1], 3), name='in1')

    x = xception(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(l2_coeff))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(l2_coeff))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x, name='xception_based_classifier')

    return model