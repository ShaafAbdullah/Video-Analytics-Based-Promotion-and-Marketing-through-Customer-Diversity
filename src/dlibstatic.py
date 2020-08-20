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
from utils.preprocessor import preprocess_input
import time

count_male=0
count_female=0
kids=0
elders=0
youngadults=0
# parameters for loading data and images
image_path = 'C:/Users/hp i7/Desktop/_MG_1318.jpg'
#detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = '../trained_models/gender_models/updated_weights.hdf5'
agegroup_model_path = '../trained_models/agegroup/age_group_weights.hdf5'

gender_labels = get_labels('imdb')
agegroup_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
agegroup_offsets = (20, 40)

# loading models
#face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)
agegroup_classifier = load_model(agegroup_model_path, compile=False)

# getting input model shapes for inference

gender_target_size = gender_classifier.input_shape[1:3]
agegroup_target_size = agegroup_classifier.input_shape[1:3]


# loading images
rgb_image = load_image(image_path, grayscale=True)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

hog_face_detector = dlib.get_frontal_face_detector()
#faces = detect_faces(face_detection, gray_image)
faces_hog = hog_face_detector(gray_image, 1)


for face_coordinates in faces_hog:
    start_time = time.time()
    x = face_coordinates.left()
    y = face_coordinates.top()
    w = face_coordinates.right() - x
    h = face_coordinates.bottom() - y
    face_off=x,y,w,h
    x1, x2, y1, y2 = apply_offsets(face_off, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]

    x1, x2, y1, y2 = apply_offsets(face_off, agegroup_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    rgb_face = cv2.resize(rgb_face, (64,64))
    gray_face = cv2.resize(gray_face, (agegroup_target_size))

    #   gray_face = cv2.resize(gray_face, (emotion_target_size))

    rgb_face = np.expand_dims(rgb_face, 2)
    rgb_face = np.expand_dims(rgb_face, 0)
    rgb_face = preprocess_input(rgb_face, False)

    gray_face = preprocess_input(gray_face, False)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)

    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]

    agegroup_label_arg = np.argmax(agegroup_classifier.predict(gray_face))
    agegroup_text = agegroup_labels[agegroup_label_arg]


    print("FPS: ", 1.0 / (time.time() - start_time))
    print(time.time() - start_time)

    if gender_text == gender_labels[0]:
        color = (255,0,0)
        count_female=count_female+1

    else:
        color = (255, 0, 0)
        count_male=count_male+1
    if agegroup_text == agegroup_labels[0]:

            kids += 1

    elif agegroup_text == agegroup_labels[1]:

            elders += 1
    else:

            youngadults += 1
    draw_bounding_box(face_off, rgb_image, color)
    #draw_text(face_off, rgb_image,gender_text,
             # color, 0, -20, 5, 5)
    draw_text(face_off, rgb_image,agegroup_text,
             color, 0, -45, 5, 5)




resized_image = cv2.resize(rgb_image, (748, 769))
#cv2.imwrite('../images/tested/7.jpg', resized_image)
cv2.imwrite('C:/Users/hp i7/Desktop/n5.jpg',resized_image)
#cv2.imshow('abc',resized_image)
cv2.waitKey(0)


