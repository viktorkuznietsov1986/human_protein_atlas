import cv2
import pandas as pd
import numpy as np


from models.resnet50 import build_resnet50_classifier
from preprocess import image_preprocess

submit = pd.read_csv('../input/sample_submission.csv')

input_shape = (300, 300)

num_classes = 28

n_channels = 3

model = build_resnet50_classifier(input_shape, num_classes, l2_coeff=0.01)

model.load_weights('model.h5')

predicted = []
for name in submit['Id']:
    image = image_preprocess('../input/test/', name)
    image = cv2.resize(image, input_shape)
    image = np.reshape(image, [1, input_shape[0], input_shape[1], n_channels])
    prediction = model.predict(image)[0]
    score_predict = model.predict([image])
    label_predict = np.argwhere(score_predict>=0.5).flatten()
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)


submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)

