import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from generators import generator, DataGenerator
from loss import focal_loss
from models.inceptionv3 import build_inceptionv3_classifier
from preprocess import name_label_dict, multihot_encode
from scores import f_score

# define the number of classes
from training import train_model

num_classes = len(name_label_dict)

# define the input file structure
input_folder = '../input/'
train_folder = os.path.join(input_folder, 'train_scaled/')
test_folder = os.path.join(input_folder, 'test/')
labels_file = os.path.join(input_folder, 'train.csv')
sample_submission_file = os.path.join(input_folder, 'sample_submission.csv')

# retrieve the train labels
train_labels_df = pd.read_csv(input_folder + 'train.csv')
train_labels_df['Target'] = [[int(i) for i in s.split()] for s in train_labels_df['Target']]

# build the multi-hot encoded label data
data = {row['Id']:multihot_encode(row['Target'], num_classes) for _, row in train_labels_df.iterrows()}

# define the train names
train_names = list({f[:36] for f in os.listdir(train_folder)})

# shuffle train_names first
train_names = shuffle(train_names)

# split them
train_n, dev_n = train_test_split(train_names, test_size=0.25)

# define the input shape
input_shape = (300, 300)

# build the model, show summary and compile
model = build_inceptionv3_classifier(input_shape, num_classes, l2_coeff=0.01)
model.summary()

model.compile(loss=focal_loss, optimizer='adam', metrics=['accuracy', f_score])


# set the number of epochs
num_epochs = 20

# set the batch size
batch_size = 8

train_params = {
    'input_dir':train_folder,
    'samples': train_n,
    'label_dict': data,
    'image_shape': input_shape,
    'batch_size': batch_size,
    'augment': True,
    'shuffle': True
}

dev_params = {
    'input_dir':train_folder,
    'samples': dev_n,
    'label_dict': data,
    'image_shape': input_shape,
    'batch_size': batch_size,
    'augment': False,
    'shuffle': True
}

# create generators
train_generator = DataGenerator(**train_params)#generator(train_folder, train_n, data, image_shape=input_shape, augment=True, batch_size=batch_size)
validation_generator = DataGenerator(**dev_params)#generator(train_folder, dev_n, data, image_shape=input_shape, batch_size=batch_size)

# do the training
train_model(model, train_generator, validation_generator, epochs=num_epochs,
            use_multiprocessing=True, workers=4)


