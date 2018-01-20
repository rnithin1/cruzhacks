import cv2
import dlib
import numpy as np
import imutils
from imutils.video import VideoStream

def show_webcam():
	cam = cv2.VideoCapture(0)

	face_cascade = cv2.CascadeClassifier('haarcascadedefault.xml')

	while True:
		ret_val, frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	
		if key == ord("q"):
			break
	cv2.destroyAllWindows()

def main():
	show_webcam()

if __name__ == '__main__':
	main()
