import cv2
cv2.namedWindow('window_frame')
counter=0
video_capture = cv2.VideoCapture('rtsp://admin:hik@12345@192.168.1.64/1')
start_time = time.time()
while True:
    bgr_image = video_capture.read()[1]
    cv2.imshow('window_frame', bgr_image)
    bgr_image = cv2.rotate(bgr_image, 180)
    # start_time = time.time()
    print("FPS: ", counter / (time.time() - start_time))
    print('no of frames', counter)
    print('time', time.time() - start_time)
    counter+=1
    # if (time.time() - start_time) > x:

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break