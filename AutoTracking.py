from genericpath import exists
from time import sleep
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cam = cv2.VideoCapture('traffic2.mp4')


# Check if camera opened successfully
if (cam.isOpened() == False):
    print("Error opening video file")

min_width_rect = 80
min_height_rect = 80

count_line_position = 550
#Inizialize Substructor
#algo = cv2.bgsegm.createBackgroundSubtractorMOG()
algo = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=200)

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy
 
detect = []
offset = 6  #Allowable error between pixel
counter_vehicle = 0


if (not exists("frame_mask.png")):
    if(cam.isOpened()):
        ret, frame = cam.read()
        cv2.imwrite('frame.jpg',frame)
        exit()

#prendo la mask
mask = cv2.imread("frame_mask.png")



while (cam.isOpened()):
    ret, frame = cam.read()
    height, width, _ = frame.shape
    # Extract Region of interest
    #roi = frame[340: 720,500: 800]
    # 1. Object Detection
    mask = algo.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('Frame', frame)
    sleep(1)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

exit()


while (True):

    exitPressed=False
# Read until video is completed
    while (cam.isOpened()):

        # Capture frame-by-frame
        ret, frame = cam.read()
        if ( frame is None):
            break

        frame = cv2.copyTo(frame, mask)


        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(3,3),5)
        #repeat on each frame
        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub,np.ones((5,5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilatata = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
        dilatata = cv2.morphologyEx(dilatata,cv2.MORPH_CLOSE, kernel)
        counterShape,h = cv2.findContours(dilatata,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,127,0),3)

        detect.clear()
        for (i,c) in enumerate(counterShape):
            (x,y,w,h) = cv2.boundingRect(c)
            validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
            if not validate_counter:
                continue

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            center = center_handle(x,y,w,h)
            detect.append(center)
            cv2.circle(frame,center,4,(0,0,255),-1)
            
        
            """
                for (x,y) in detect:
                    if y<(count_line_position+offset) and y>(count_line_position-offset):
                        counter_vehicle += 1
                    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                    detect.remove((x,y))
                    print("Vehicle Counter: "+str(counter_vehicle))
            """


        cv2.putText(frame, "VEHICLE COUNTER: "+str(len(detect)),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

        #cv2.imshow("Detecter ", dilatata)
        cv2.imshow("Detecter ", dilatata)

        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord(' '):
                exitPressed=True
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cam.release()
    if exitPressed:
        # Closes all the frames
        cv2.destroyAllWindows()
        exit()
    else:
        cam = cv2.VideoCapture('traffic2.mp4')  
    
