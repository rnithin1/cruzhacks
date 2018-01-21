import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

def show_webcam():
	cam = cv2.VideoCapture(0)

	face_cascade = cv2.CascadeClassifier('haarcascadefull.xml')

	#while True:
	#ret_val, frame = cam.read()
	frame = cv2.imread('../otsukevin2.png')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 127, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh,       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(frame, contours, -1, (0,255,0), 3)
	c = max(contours, key=cv2.contourArea)
	cv2.drawContours(frame, c, -1, (0,255,0), 3)
	#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	'''
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       	cv2.imshow("Frame", frame)
	
               '''
	cv2.imwrite('final.png', frame)
	#key = cv2.waitKey(1) & 0xFF
	#if key == ord("q"):
	#	break
	cv2.destroyAllWindows()

def main():
	show_webcam()

if __name__ == '__main__':
	main()
