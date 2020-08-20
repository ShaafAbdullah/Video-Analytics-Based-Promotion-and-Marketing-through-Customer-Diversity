from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

import sys

import dlib

import cv2
from keras.models import load_model
import numpy as np

from IPython.display import Image, display
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image



gender_model_path = '../trained_models/gender_models/updated_weights.hdf5'
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX
gender_offsets = (30, 60)
# Grab video from your webcam

# Face detector
detector = dlib.get_frontal_face_detector()
#detector2=dlib.cnn_face_detection_model_v1('C:/Users/Zunaira Akmal/Desktop/mmod_human_face_detector.dat')
gender_classifier = load_model(gender_model_path, compile=False)
gender_target_size = gender_classifier.input_shape[1:3]
image_path="E:/FYP37CE-B/Seperated/face_classification-master/images/21.jpg"
print(image_path)



# loading images
out_image=load_image(image_path, grayscale=True)
rgb_image = load_image(image_path, grayscale=True)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')


hog_face_detector = dlib.get_frontal_face_detector()
faces_hog = hog_face_detector(gray_image, 1)
    #faces = detect_faces(face_detection, gray_image)




for face_coordinates in faces_hog:
        x = face_coordinates.left()
        y = face_coordinates.top()
        w = face_coordinates.right() - x
        h = face_coordinates.bottom() - y
        face_off=x,y,w,h

        x1, x2, y1, y2 = apply_offsets(face_off, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

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
        print (gender_label_arg)

        
