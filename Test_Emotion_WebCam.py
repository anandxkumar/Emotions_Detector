import cv2
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

#Creating frontal face detection classifer
#Directory of haarcasade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier("F:/My Projects/Emotion detector using keras and openCV/haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0) # Opens web cam

# Directory of classifier 
classifer = load_model("F:/My Projects/Emotion detector using keras and openCV/Emotion_trained.h5")

# Our 5 classes

class_labels =['Angry','Happy','Neutral','Sad','Surprise']  

while True :
    
    check , frame  = cam.read()  # check - returns True or False , frame - returns ndarray of the frame captured

    # Converting to gray image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Search the coordinates of the frame 
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)
    
    # making a rectangular box for the face detected
    for x,y,w,h in faces :
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # Cropping only the face detected and apply classifer on it for faster and efficient processing
        img = gray_img[y:y+h,x:x+w] # number of rows is y->y+h
        # Resizzing image according to classifier
        img = cv2.resize(img,(48,48), interpolation=cv2.INTER_AREA)
    
        if(np.sum([img])!=0): # It means face is detected
            # Now we have to rescale image accoring to test set
            img = img.astype(float)/255.0
            img = img_to_array(img) # Converting to to an array
            # increasing dimensions to be accepted by predict
            img = np.expand_dims(img,axis=0)
            
            # Now making prediction  on the img
            pred = classifer.predict(img)[0]
            # The prediction we got
            label = class_labels[pred.argmax()] # argmax will gives us the index of the class which has max prediction
            cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
   # Displaying frames
    cv2.imshow("Emoji Detector",frame) 
    
    key = cv2.waitKey(1) 
    if key & 0xFF==ord('q'):
        break
            
    
    
cam.release()

cv2.destroyAllWindows()