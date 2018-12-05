import cv2
import pandas as pd
import numpy as np

from models.xception import build_xception_classifier
from preprocess import image_preprocess

submit = pd.read_csv('../input/sample_submission.csv')

input_shape = (300, 300)

num_classes = 28

n_channels = 3

model = build_xception_classifier(input_shape, num_classes, l2_coeff=0.01)

model.load_weights('model.h5')

def predict(model):
    predicted02 = []
    predicted03 = []
    predicted04 = []
    predicted05 = []

    for name in submit['Id']:
        image = image_preprocess('../input/test/', name)
        image = cv2.resize(image, input_shape)
        image = np.reshape(image, [1, input_shape[0], input_shape[1], n_channels])
        prediction = model.predict(image)[0]

        indices02 = np.argwhere(prediction >= 0.2).flatten()
        labels02 = ' '.join(str(l) for l in indices02)

        indices03 = np.argwhere(prediction >= 0.3).flatten()
        labels03 = ' '.join(str(l) for l in indices03)

        indices04 = np.argwhere(prediction >= 0.4).flatten()
        labels04 = ' '.join(str(l) for l in indices04)

        indices05 = np.argwhere(prediction >= 0.5).flatten()
        labels05 = ' '.join(str(l) for l in indices05)

        predicted02.append(labels02)
        predicted03.append(labels03)
        predicted04.append(labels04)
        predicted05.append(labels05)

        return predicted02, predicted03, predicted04, predicted05


predicted02, predicted03, predicted04, predicted05 = predict(model)

submit['Predicted'] = predicted02
submit.to_csv('submission02.csv', index=False)

submit['Predicted'] = predicted03
submit.to_csv('submission03.csv', index=False)

submit['Predicted'] = predicted04
submit.to_csv('submission04.csv', index=False)

submit['Predicted'] = predicted05
submit.to_csv('submission05.csv', index=False)
