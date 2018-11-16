from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, \
    BatchNormalization, Dropout
from keras import applications


def build_inceptionv3_classifier(input_shape, num_classes, use_dropout=False):
    inceptionv3 = applications.InceptionV3(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    inceptionv3.trainable = False

    inputs = Input(shape=(input_shape[0], input_shape[1], 4), name='in1')

    x = Conv2D(3, (1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = inceptionv3(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(200)(x)
    if use_dropout:
        x = Dropout(0.7)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(100)(x)
    if use_dropout:
        x = Dropout(0.7)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(num_classes)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x, name='inceptionv3_based_classifier')

    return model