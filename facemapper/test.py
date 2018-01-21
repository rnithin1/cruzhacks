import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

def show_webcam():

	face_cascade = cv2.CascadeClassifier('haarcascadefull.xml')

	#while True:
	#ret_val, frame = cam.read()
	frame = cv2.imread('../otsukevin2.png')
	frame2 = np.copy(frame)
	y, x, c = frame.shape
	cv2.rectangle(frame,(0, y - 3),(x, y),(255,255,255),4)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 127, 255, 0)
	frame[thresh == 0] = [0, 0, 255]
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(frame, contours, -1, (0,255,0), 3)
	c = max(contours, key=cv2.contourArea)
	contours.remove(c)
	c = max(contours, key=cv2.contourArea)
	#print(c)
	M = cv2.moments(c)
	cv2.drawContours(frame, [c], -1, (0,255,0), 4)
	#frame = frame[h:y-h, w:x-w]
	cx = int(M["m10"] / M["m00"])
	cy = int(M["m01"] / M["m00"])
	mar1, mar2 = [z for z in [list(x) for x in list(cv2.minAreaRect(c)[:-1])]]
	mar1 = [int(x) for x in mar1]
	mar2 = [int(x) for x in mar2]
	cv2.rectangle(frame2, tuple(mar1), tuple(mar2), (255, 0, 255), 4)
	cv2.circle(frame2, (cx, cy), 10, (255, 0, 0), -1)
	simg = cv2.imread('../qipao.png')
	#print(simg.shape, frame2.shape)
	#frame2[cy:cy+simg.shape[0], cx:cx+simg.shape[1]] = simg
	#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	'''
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       	cv2.imshow("Frame", frame)
	
               '''
	cv2.imwrite('final.png', frame2)
	#key = cv2.waitKey(1) & 0xFF
	#if key == ord("q"):
	#	break
	cv2.destroyAllWindows()

def main():
	show_webcam()

if __name__ == '__main__':
	main()
