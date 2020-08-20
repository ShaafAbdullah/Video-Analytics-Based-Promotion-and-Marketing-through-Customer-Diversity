import sys

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
# parameters for loading data and images
image_path = '../images/31.jpg'
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = '../trained_models/gender_models/CNN_updated_weights.hdf5'

#emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')

font = cv2.FONT_HERSHEY_SIMPLEX
gender_offsets = (30, 60)
gender_offsets = (10, 10)

# loading models
face_detection = load_detection_model(detection_model_path)

gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inferenc
gender_target_size =   gender_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=True)

gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')
faces = detect_faces(face_detection, gray_image)

for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]
    rgb_face = cv2.resize(rgb_face, (gender_target_size))
    rgb_face = preprocess_input(rgb_face, False)
    print ('line 60',np.shape(rgb_face))
    rgb_face = np.expand_dims(rgb_face, 2)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_classifier.predict(rgb_face)

    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]

    print (gender_label_arg)

    if gender_text == gender_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 3, 2)
#bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
resized_image = cv2.resize(rgb_image, (1024, 769))
cv2.imwrite('../images/predicted_test_image31.png', resized_image)

