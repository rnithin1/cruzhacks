import cv2
import numpy as np
import imutils
from scipy.interpolate import splprep, splev
import os

def run():

	# Copying the frame to be used as a reference point
	# Frame1 will be binarized, 2 will be reference and 3
	# will be used for facial detection purposes
	frame = cv2.imread('kevin1.jpg')
	frame2, frame3 = np.copy(frame), np.copy(frame)
	y, x, c = frame.shape #4032x3024
	print(y, x, c)
	
	# Facial Haar classifier; detects faces
	face_cascade = cv2.CascadeClassifier('haarcascadedefault.xml')
	gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (a,b,c,d) in faces:
		cv2.rectangle(frame3,(a,b),(a+c,b+d),(255,0,0),2)
	facial_lower_bound = b + d 

	# Binarizes the image using Otsu's algorithm
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                    
	frame = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] 

	# Cropping out the bottom half to allow for a contour
	cv2.rectangle(frame,(0, y - 3),(x, y),(255,255,255),4)
	ret, frame = cv2.threshold(frame, 127, 255, 0)
	im2, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imwrite("frame.png", frame)

	# Finding the second longest contour
	person_max = max(contours, key=cv2.contourArea)
	removearray(contours, person_max)
	contour_max = max(contours, key=cv2.contourArea)
	
	# Finding the boundary points of the underestimated rectangle
	#north, south, east, west = find_boundary_points(c)	


	# Drawing the contours
	M = cv2.moments(contour_max)
	cv2.drawContours(frame, [contour_max], -1, (128,255,0), 40)

	# Finding the centroid of the area using the moment of inertia
	cx = int(M["m10"] / M["m00"])
	cy = int(M["m01"] / M["m00"])

	#mar1, mar2 = [z for z in [list(x) for x in list(cv2.minAreaRect(c)[:-1])]]
	#mar1 = [int(x) for x in mar1]
	#mar2 = [int(x) for x in mar2]
	#cv2.rectangle(frame2, tuple(mar1), tuple(mar2), (255, 0, 255), 4)

	# Plotting the centroid
	cv2.circle(frame, (cx, cy), 10, (255, 0, 0), 3)

	# Choosing the shirt
	simg = cv2.imread('../qipao.png')
	simg = cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY)                    
	simg = cv2.threshold(simg, 0, 255, cv2.THRESH_OTSU)[1] 
	ret, frame = cv2.threshold(simg, 127, 255, 0)
	img, scontours, hierarchys = cv2.findContours(simg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imwrite("frame.png", simg)
	simg_max = max(scontours, key=cv2.contourArea)
	removearray(scontours, simg_max)
	simg_contour = max(scontours, key=cv2.contourArea)
	M = cv2.moments(simg_contour)
	cv2.drawContours(simg, [simg_contour], -1, (128,255,0), 5)
	cv2.imwrite('shirtfinal.png', simg)
	'''
	interpolation_difference = max(simg_contour.shape[0], contour_max.shape[0])
	gaurav = interpolate((lambda : simg_contour if simg_contour.shape[0] <= contour_max.shape[0] \
							else contour_max)(), interpolation_difference)

	if simg_contour.shape[0] <= contour_max.shape[0]:
		simg_contour = gaurav[0].reshape(1,-1,2)
		contour_max = contour_max.reshape(1,-1,2)
	else:
		contour_max = gaurav[0].reshape(1,-1,2)
		simg_contour = simg_contour.reshape(1,-1,2)
	'''
	f = 8
	simg_contour = interpolate(simg_contour, f)[0]
	contour_max = interpolate(contour_max, f)[0]
	# EstimateRigidTransform
	affine_matrix = cv2.estimateRigidTransform(simg_contour,contour_max,True)
	print(affine_matrix)
	print(simg_contour, contour_max)
	

	#print(simg.shape, frame2.shape)
	#frame2[cy:cy+simg.shape[0], cx:cx+simg.shape[1]] = simg
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


def removearray(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def find_boundary_points(c):
	"test"

def interpolate(contour, difference):
	smoothened = []
	x,y = contour.T
	x = x.tolist()[0]
	y = y.tolist()[0]
	tck, u = splprep([x,y], u=None, s=1.0, per=1)
	u_new = np.linspace(u.min(), u.max(), difference)
	x_new, y_new = splev(u_new, tck, der=0)
	res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
	smoothened.append(np.asarray(res_array, dtype=np.int32))
	return smoothened
	
def main():
	run()

if __name__ == '__main__':
	main()
