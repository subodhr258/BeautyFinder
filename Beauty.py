import cv2
import numpy as np 
import dlib
import os


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')


frame = cv2.imread("face3.jpg")
(h,w,_) = frame.shape

h2 = 600
w2 = int(h2 * h/ w)
frame = cv2.resize(frame,(h2,w2))
frame = cv2.flip(frame,1)
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

faces = detector(gray)
face = faces[0]
x1 = face.left()
y1 = face.top()
x2 = face.right()
y2 = face.bottom()

landmarks = predictor(gray, face)

len_eyecenter_nosetip = landmarks.part(30).y - landmarks.part(27).y
len_nosetip_chin = landmarks.part(8).y - landmarks.part(30).y
diff1 = abs(1.618-(len_nosetip_chin/len_eyecenter_nosetip))

len_of_eye = landmarks.part(45).x - landmarks.part(42).x
len_between_eye_and_center = landmarks.part(42).x - landmarks.part(27).x
diff2 = abs(1.618-(len_of_eye/len_between_eye_and_center))

len_nosetip_and_lipcenter = landmarks.part(62).y - landmarks.part(30).y
len_lipcenter_chin = landmarks.part(8).y - landmarks.part(62).y
diff3 = abs(1.618-(len_of_eye/len_between_eye_and_center))

diff = (diff1+diff2+diff3)/3
perc = (1 - diff/1.618)*100
print("Beauty percentage =",perc)

faces2 = face_cascade.detectMultiScale(gray,1.1,4)
(x,y,w,h) = faces2[0]
cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
frame = cv2.putText(frame,"Beauty percentage:"+str(round(perc,2)),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(250,30,30),2)

cv2.imshow("Frame",frame)
cv2.waitKey(0)

cv2.destroyAllWindows()