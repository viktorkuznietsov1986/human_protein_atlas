import cv2
import keras
import numpy as np
import sklearn
from preprocess import image_preprocess


class DataGenerator(keras.utils.Sequence):
    'Generates data for keras'
    def __init__(self, input_dir, samples, label_dict, image_shape, batch_size, shuffle=True, augment=False):
        self.input_dir = input_dir
        self.samples = samples
        self.labels_dict = label_dict
        self.image_shape = image_shape

        self.shuffle = shuffle
        self.augment = augment

        self.augment_ratio = 3
        self.batch_size = batch_size if augment == False else batch_size//self.augment_ratio

        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.samples)/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_samples = self.samples[index*self.batch_size:(index + 1)*self.batch_size]

        X, y = self.__data_generation(batch_samples)

        return X, y


    def on_epoch_end(self):
        'Shuffle samples after each epoch if needed'
        if self.shuffle:
            sklearn.utils.shuffle(self.samples)

    def __data_generation(self, batch_samples):
        'Generate data'
        images = []
        labels = []

        for batch_sample in batch_samples:
            image = image_preprocess(self.input_dir, batch_sample)
            if image.shape != self.image_shape:
                image = cv2.resize(image, self.image_shape)

            label = self.label_dict[batch_sample]

            if self.augment:
                flipped_horiz = cv2.flip(image, 0)
                flipped_vert = cv2.flip(image, 1)

                images.extend([image, flipped_horiz, flipped_vert])
                labels.extend([label for _ in range(self.augment_ratio)])

            else:
                images.append(image)
                labels.append(label)

        X_train = np.array(images)
        y_train = np.array(labels)
        return sklearn.utils.shuffle(X_train, y_train)

def generator(input_dir, samples, label_dict, image_shape, augment=False, batch_size=32):
    num_samples = len(samples)

    augment_ratio = 3 if augment else 1

    batch_size = batch_size // augment_ratio

    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            labels = []

            for batch_sample in batch_samples:
                image = image_preprocess(input_dir, batch_sample)
                if image.shape != image_shape:
                    image = cv2.resize(image, image_shape)

                label = label_dict[batch_sample]

                if augment:
                    flipped_horiz = cv2.flip(image, 0)
                    flipped_vert = cv2.flip(image, 1)

                    images.extend([image, flipped_horiz, flipped_vert])
                    labels.extend([label for i in range(augment_ratio)])

                else:
                    images.append(image)
                    labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, y_train)