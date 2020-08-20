import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
import itertools
#path1="C:/Users/Zunaira Akmal/Desktop/AIassignment3/inputdata"
path2="C:/Users/Zunaira Akmal/Desktop/AIassignment3/resize1"
from sklearn.metrics import confusion_matrix
#listing=os.listdir(path1)

#loading data
#for file in listing:
   # im=Image.open(path1 + '\\' + file)
   # img= im.resize((16,16))
   # gray= img.convert('L')
   # gray.save(path2+ '\\' + file, "JPEG")

#convert images into array
imlist= os.listdir(path2)
matrix= np.array([np.array(Image.open('C:/Users/Zunaira Akmal/Desktop/AIassignment3/resize1' +'\\' + im2)).flatten() for im2 in imlist], 'f')
print(np.shape(matrix))

#assign labels
labels=np.ones((120,), dtype=int)
labels[0:61]=0
labels[62:121]=1

#appending labels to data
train=[matrix,labels]
list=[]
#calculate for twenty neighbours
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)


    knn.fit(matrix, labels)
    a = knn.score(matrix, labels)
    list.append(a)

#graph when k ranges from 1-> 20
neighbours=np.arange(1,20)
plt.title('Accuracy by varying neighbours')
plt.plot(neighbours, list, label='Train Accuracy')
plt.legend()
plt.xlabel('Neighbours')
plt.ylabel('Accuracy')
plt.show()

#####splitting of data
data1, data2, label1, label2= train_test_split(matrix, labels,
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=123)

list1=[]
list2=[]

 #task3
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)


    knn.fit(data1, label1)
    a = knn.score(data1, label1)
    list1.append(a)
    s = knn.score(data2, label2)
    list2.append(s)

#graph when k ranges from 1-> 20
task3_neighbours=np.arange(1,20)
plt.title('Accuracy by varying neighbours'  
          'By splitting into equal parts')
plt.plot(task3_neighbours,list1, label='Train Accuracy')
plt.plot(task3_neighbours,list2, label='Test Accuracy')
plt.legend()
plt.xlabel('Neighbours')
plt.ylabel('Accuracy')
plt.show()

                                        #task4
list3=[]
list4=[]
data11, data22, label11, label22= train_test_split(matrix, labels,test_size=0.3,random_state=123) #dataSplitting
prediction=[]
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(data11, label11)
    j = knn.score(data11, label11)
    list3.append(j)
    ii = knn.score(data22, label22)
    list4.append(ii)
    prediction=knn.predict(np.array(data22))
    print(prediction)
    #predict_classes = prediction.argmax(axis=1)


#graph when k ranges from 1-> 20
task4_neighbours=np.arange(1,20)
print(list1)
plt.title('Accuracy by varying neighbours'  
          'By splitting into random dataset of test and train')
plt.plot(list3, label='Train Accuracy')
plt.plot(list4, label='Test Accuracy')
plt.legend()
plt.xlabel('Neighbours')
plt.ylabel('Accuracy')
plt.show()

#my function to calculate average
average=[]
import numpy as np
def avg(a,b):
    i=0
    g=len(a)
    print(g)
    for var in range(g):
        average.append((a[var]+b[var])/2)
    return average

myaverage=avg(list2,list4)
plt.plot(myaverage, label='Average Accuracy')
plt.legend()
plt.ylabel('Accuracy')
plt.show()

#Confusion Matrix
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

cm=confusion_matrix(label22,prediction)
cm_plot_labels=['0','1']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')
plt.show()
