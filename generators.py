import cv2
import keras
import numpy as np
import sklearn
from preprocess import image_preprocess
from imgaug import augmenters as iaa


class DataGenerator(keras.utils.Sequence):
    'Generates data for keras'
    def __init__(self, input_dir, samples, label_dict, image_shape, batch_size, shuffle=True, augment=False):
        self.input_dir = input_dir
        self.samples = samples
        self.label_dict = label_dict
        self.image_shape = image_shape

        self.shuffle = shuffle
        self.augment = augment

        self.augment_ratio = 2
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
                augmented = self.do_augmentation(image)

                images.extend([image, augmented])
                labels.extend([label, label])

            else:
                images.append(image)
                labels.append(label)

        X_train = np.array(images)
        y_train = np.array(labels)
        return sklearn.utils.shuffle(X_train, y_train)

    def do_augmentation(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug