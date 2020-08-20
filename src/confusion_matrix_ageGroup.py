import cv2
from keras.models import load_model
import numpy as np
from utils.preprocessor import preprocess_input
from utils.datasets import get_labels
import os
import matplotlib.pyplot as plt
import skimage
#import torch.nn as nn
from skimage import data
from sklearn.metrics import confusion_matrix
import itertools
gender_model_path = '../trained_models/gender_models/age_group_weights.hdf5'
gender_classifier = load_model(gender_model_path, compile=False)
gender_target_size = gender_classifier.input_shape[1:3]

prediction=[]
total_true=0
kids=0   #labels->0
adults=0 #labels->2
elders=0 #labels->1

#confusion matrix


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


#labels for prediction
def load_data(data_directory): #function for calling of images with labels
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpg")]#change the format as .jpg,.ppm or anything else in which data is formatted
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "E:/FYP37CE-B/Seperated/face_classification-master/datasets/dataconfusion/" #root directory of images
train_data_directory = os.path.join(ROOT_PATH, "labels")
images,training_labels=load_data(train_data_directory)



#prediction
for filename in os.listdir('E:/FYP37CE-B/Seperated/face_classification-master/images/'):
    img = cv2.imread(os.path.join('E:/FYP37CE-B/Seperated/face_classification-master/images/',filename),0)

    rgb_face = cv2.resize(img, (gender_target_size))
    rgb_face = np.expand_dims(rgb_face, 2)
    
    rgb_face = np.expand_dims(rgb_face, 0)

    rgb_face = preprocess_input(rgb_face, False)

    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)

    prediction.append(gender_label_arg)
    print (gender_label_arg)

print (len(training_labels))
print(len(prediction))
for i in range(8011):
    if prediction[i]==0 and training_labels[i]==0:
        kids+=1
    if prediction[i]==1 and training_labels[i]==1:
        elders+=1
    if prediction[i]==2 and training_labels[i]==2:
        adults+=1

for k in range(8011):
    if prediction[k]==training_labels[k]:
        total_true=total_true+1

print(kids,elders,adults)

print('Total Kids are 2578')
print('Total Adults are 2433')
print('Total Elders are 3008')
print('Total People are 8019')

kids_accuracy=kids/2578
Elders_accuracy=elders/3008
Adults_accuracy=adults/2433
total_accuracy=total_true/8019

print('accuracy of kids', kids_accuracy)
print('accuracy of Adults', Adults_accuracy)
print('accuracy of Elders', Elders_accuracy)
print('Overall accuracy', total_accuracy)

array=prediction[0:8011]
cm=confusion_matrix(training_labels,array)
cm_plot_labels=['0','1','2']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')
plt.show()
