from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
# helper functions from pyimagesearch.com

import dlib


import sys
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
import itertools
import cv2
from keras.models import load_model
import numpy as np
import os

from IPython.display import Image, display
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

prediction=[]
#confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




#actual labels

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



print(train_labels)
for filename in os.listdir('E:/FYP37CE-B/Seperated/face_classification-master/images/'):
    img = cv2.imread(os.path.join('E:/FYP37CE-B/Seperated/face_classification-master/images/',filename),0)
    gender_model_path = '../trained_models/gender_models/updated_weights.hdf5'
    gender_labels = get_labels('imdb')

    font = cv2.FONT_HERSHEY_SIMPLEX
    gender_offsets = (30, 60)

    detector = dlib.get_frontal_face_detector()

    gender_classifier = load_model(gender_model_path, compile=False)
    gender_target_size = gender_classifier.input_shape[1:3]





    gray_image=img
    gray_image = np.squeeze(gray_image)
    #print(gray_image)
    gray_image = gray_image.astype('uint8')


    hog_face_detector = dlib.get_frontal_face_detector()
    faces_hog = hog_face_detector(gray_image, 1)
    print(faces_hog)



    for face_coordinates in faces_hog:
        x = face_coordinates.left()
        y = face_coordinates.top()
        w = face_coordinates.right() - x
        h = face_coordinates.bottom() - y
        face_off=x,y,w,h

        x1, x2, y1, y2 = apply_offsets(face_off, gender_offsets)
        rgb_face = gray_image[y1:y2, x1:x2]

       # x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        #gray_face = gray_image[y1:y2, x1:x2]

        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        #gray_face = cv2.resize(gray_face, (emotion_target_size))

        rgb_face = np.expand_dims(rgb_face, 2)
        rgb_face = np.expand_dims(rgb_face, 0)

        rgb_face = preprocess_input(rgb_face, False)

        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        prediction.append(gender_label_arg)
        print (gender_label_arg)

cm=confusion_matrix(train_labels,prediction)
cm_plot_labels=['0','1','2']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')
plt.show()