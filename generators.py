import cv2
import numpy as np
import sklearn
from preprocess import image_preprocess


def generator(input_dir, samples, label_dict, image_shape, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            labels = []

            for batch_sample in batch_samples:
                print (batch_sample)
                image = image_preprocess(input_dir, batch_sample)
                if image.shape != image_shape:
                    image = cv2.resize(image, image_shape)

                label = label_dict[batch_sample]

                images.append(image)
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, y_train)