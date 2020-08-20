from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from utils.datasets import DataManager
from models.cnn import mini_XCEPTION
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# parameters
from keras.models import load_model
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_classifier = load_model(gender_model_path, compile=False)

batch_size = 32
num_epochs = 250
validation_split = .1
do_random_crop = False
patience = 100
num_classes = 2
dataset_name = 'imdb'
input_shape = (64, 64, 1)

model = mini_XCEPTION(input_shape, num_classes,include_top=False,weights='gender',pooling='max')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
img_path = 'E:/FYP37CE-B/Seperated/Datasetemc/cropedfinal!new/107.jpg'
img = image.load_img(img_path)
gender_prediction = gender_classifier.predict(img)



