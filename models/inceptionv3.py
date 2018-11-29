from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, \
    BatchNormalization, Dropout, regularizers
from keras import applications


def build_inceptionv3_classifier(input_shape, num_classes, l2_coeff=0.01):
    inceptionv3 = applications.InceptionV3(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    inceptionv3.trainable = True

    inputs = Input(shape=(input_shape[0], input_shape[1], 3), name='in1')

    #x = Conv2D(3, (1, 1), padding='same', kernel_regularizer=regularizers.l2(l2_coeff))(inputs)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = inceptionv3(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(l2_coeff))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(l2_coeff))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x, name='inceptionv3_based_classifier')

    return model