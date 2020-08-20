

# prediction
for filename in os.listdir('E:/FYP37CE-B/Seperated/face_classification-master/images/'): #path of directory in which data is present
    img = cv2.imread(os.path.join('E:/FYP37CE-B/Seperated/face_classification-master/images/', filename), 0) #files will be stored in img one by one change it accordingly


    #use the above data here