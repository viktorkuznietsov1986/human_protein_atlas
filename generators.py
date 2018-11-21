import cv2
import numpy as np
import sklearn
from preprocess import image_preprocess


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