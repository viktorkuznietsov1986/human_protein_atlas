import os

import pandas as pd
import tqdm as tqdm
import numpy as np


from models.resnet50 import build_resnet50_classifier
from preprocess import image_preprocess

submit = pd.read_csv('../input/sample_submission.csv')

input_shape = (300, 300)

num_classes = 28

model = build_resnet50_classifier(input_shape, num_classes, l2_coeff=0.01)

model.load_weights('model.h5')

predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = image_preprocess(path, input_shape)
    score_predict = model.predict(image)
    label_predict = np.arange(num_classes)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)


submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)

