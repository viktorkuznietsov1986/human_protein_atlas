import os

import pandas as pd
import tqdm as tqdm
import numpy as np
from keras.engine.saving import load_model

from preprocess import image_preprocess

submit = pd.read_csv('../input/sample_submission.csv')

input_shape = (300, 300)

model = load_model('model.h5')

predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = image_preprocess(path, input_shape)
    score_predict = model.predict(image)
    label_predict = np.arange(28)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)


submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)

