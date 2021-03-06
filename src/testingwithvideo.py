from statistics import mode
import dlib
import cv2
from keras.models import load_model
import numpy as np
#from update_counter import gender
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

female_count=0
male_count=0
kids=0
elders=0
youngadults=0
prev_max=0
temp=0
cor=0
diff_cor=0

gender_model_path = '../trained_models/gender_models/updated_weights.hdf5'
agegroup_model_path = '../trained_models/agegroup/age_group_weights.hdf5'

gender_labels = get_labels('imdb')
agegroup_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
agegroup_offsets = (20, 40)


gender_classifier = load_model(gender_model_path, compile=False)
agegroup_classifier = load_model(agegroup_model_path, compile=False)

# getting input model shapes for inference
agegroup_target_size = agegroup_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
agegroup_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture('C:/Users/hp i7/Desktop/HM2.mp4')

while True:

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    #gray_image=bgr_image
    #rgb_image=bgr_image
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

        try:
         rgb_face = cv2.resize(rgb_face, (gender_target_size))
         gray_face = cv2.resize(gray_face, (agegroup_target_size))
        except:
            print('no frame 1')

        #gray_face = cv2.resize(gray_face, (emotion_target_size))


       # gray_face = preprocess_input(gray_face, False)
       # gray_face = np.expand_dims(gray_face, 0)
        #gray_face = np.expand_dims(gray_face, -1)
        #emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        #emotion_text = emotion_labels[emotion_label_arg]
        #emotion_window.append(emotion_text)
        rgb_face = np.expand_dims(rgb_face, 2)
        rgb_face = np.expand_dims(rgb_face, 0)

        rgb_face = preprocess_input(rgb_face, False)
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        #cross corrletion
        try:
         if(temp>0):
          arr1_1D=np.reshape(cor, (len(cor)*4096))

          arr2_1D=np.reshape(rgb_face, (len(rgb_face)*4096))


          cor_out=np.correlate(arr1_1D,arr2_1D,'full')
         #print(cor)
          maxcor=np.max(cor_out)
          diff_cor=np.abs(prev_max-maxcor)
        # print(cor_out)
          print('difference',diff_cor)
          print ('maximum',maxcor)
          prev_max=maxcor
        except:
          print('no frame 2')
        cor=rgb_face
        temp+=1


        try:
         gender_prediction = gender_classifier.predict(rgb_face)
         gender_label_arg = np.argmax(gender_prediction)
         gender_text = gender_labels[gender_label_arg]
         gender_window.append(gender_text)
         #agegroup
         agegroup_label_arg = np.argmax(agegroup_classifier.predict(gray_face))
         agegroup_text = agegroup_labels[agegroup_label_arg]
         agegroup_window.append(agegroup_text)

         if len(gender_window) > frame_window:
            agegroup_window.pop(0)
            gender_window.pop(0)
         try:
            agegroup_mode = mode(agegroup_window)
            gender_mode = mode(gender_window)
         except:
            continue

         if gender_text == gender_labels[0]:
            color = (0, 0, 255)
            if(diff_cor>150):
             female_count+=1

         else:
            color = (255, 0, 0)
            if(diff_cor>150):
             male_count+=1
         #arr = [male_count, female_count]
         #gender(arr)
         draw_bounding_box(face_off, rgb_image, color)
         draw_text(face_off, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        except:
            print('no cap 3')
        print('female', female_count)
        print('male', male_count)

    cv2.imshow('window_frame', rgb_image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break

print('female',female_count)
print('male',male_count)
