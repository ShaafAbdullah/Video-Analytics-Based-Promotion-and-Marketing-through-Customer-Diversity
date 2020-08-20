

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from utils.datasets import DataManager
from models.cnn import mini_XCEPTION
from models.cnn import simple_CNN
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data
import os
import numpy as np
# parameters

0
batch_size = 32
num_epochs = 800
validation_split = .1
do_random_crop = False
patience = 100
num_classes = 3
dataset_name = 'imdb'
input_shape = (64,64, 1)
if input_shape[2] == 1:
    grayscale = True

images_path = '../datasets/imdb_crop/'
log_file_path = '../trained_models/gender_models/gender_training.log'
#trained_models_path = '../trained_models/gender_models/gender_mini_XCEPTION'


def load_data(data_directory): #function for calling of images with labels
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpg")]#change the format as .jpg,.ppm or anything else in which data is formatted
        for f in file_names:
            labels.append(int(d))
    return  labels

ROOT_PATH = "E:/FYP37CE-B/Seperated/face_classification-master/datasets" #root directory of images
train_data_directory = os.path.join(ROOT_PATH, "imdb_crop") #main directory of images in which train and tests images are present
train_labels=load_data(train_data_directory)
#print (train_labels)




images=[]
images2=[]
images3=[]


outdir = '0/'
outdir1='1/'
outdir2='2/'

for root, dirs, files in os.walk("E:/FYP37CE-B/Seperated/face_classification-master/datasets/imdb_crop/0"):
    for filename in files:
      full_path = os.path.join(outdir, filename)
      images.append(full_path)

#print ('here0',images)

for root, dirs, files in os.walk("E:/FYP37CE-B/Seperated/face_classification-master/datasets/imdb_crop/1"):
    for filename in files:
      full_path = os.path.join(outdir1, filename)
      images2.append(full_path)

for root, dirs, files in os.walk("E:/FYP37CE-B/Seperated/face_classification-master/datasets/imdb_crop/2"):
    for filename in files:
      full_path = os.path.join(outdir2, filename)
      images3.append(full_path)



imagesnew=images+images2+images3
ground_truth_data=dict(zip(imagesnew
                           , train_labels))

train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
#print(train_keys)

image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2],
                                 train_keys,val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 do_random_crop=do_random_crop)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes,l2_regularization=0.01)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# training model
history=model.fit_generator(image_generator.flow(mode='train'),
                   steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1
                   )
#model.save_weights('E:/FYP37CE-B/Seperated/face_classification-master/trained_models/gender_models/updated_weights.hdf5')
model.save('E:/FYP37CE-B/Seperated/face_classification-master/trained_models/gender_models/age_group_weights_new.hdf5')