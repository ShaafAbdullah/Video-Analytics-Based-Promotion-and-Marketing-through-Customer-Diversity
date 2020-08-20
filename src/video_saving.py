import numpy as np
import cv2
import time

i=0
while(i<10):

    capture_duration = 20
    cap = cv2.VideoCapture('rtsp://admin:hik@12345@192.168.1.100/1')


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_{0}.avi'.format(i),fourcc, 20.0, (640,480))

    start_time = time.time()
    while( int(time.time() - start_time) < capture_duration ):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,1)
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    i+=1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
