#based on this tutorial: https://www.youtube.com/watch?v=88HdqNDQsEk
#face database: https://www.kairos.com/blog/60-facial-recognition-databases
import cv2
import sys
import os
import numpy as py
import time

#grab command line arguments
#file_name = sys.argv[1]

#cascade files http://alereimondo.no-ip.org/OpenCV/34
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
leye_cascade = cv2.CascadeClassifier('haar/ojoI.xml')
reye_cascade = cv2.CascadeClassifier('haar/ojoD.xml')
nose_cascade = cv2.CascadeClassifier('haar/Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('haar/Mouth.xml')

#img = cv2.imread(file_name)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def measureBiometrics():
	return

def main():
	num = 0
	start_time = time.time()
	#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for filename in os.listdir("test_images"): #will be replaced with a larger dataset
		file_name = "test_images/{0}.pgm".format(num)
		img = cv2.imread(file_name)
		print(file_name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			reye = reye_cascade.detectMultiScale(roi_gray)
			leye = leye_cascade.detectMultiScale(roi_gray)
			nose = nose_cascade.detectMultiScale(roi_gray)
			mouth = mouth_cascade.detectMultiScale(roi_gray)

			for (rx,ry,rw,rh) in reye:
				cv2.rectangle(roi_color,(rx,ry),(rx+rw,ry+rh),(0,255,0),2)

			for (lx,ly,lw,lh) in leye:
				cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)

			for (mx,my,mw,mh) in mouth:
				cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)

			for (nx,ny,nw,nh) in nose:
				cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(120,120,0),2)

		cv2.imwrite(('output_images/{0}.pgm').format(num),img)
		num+=1

	print("Runtime: %s seconds" %(time.time()-start_time))
	#cv2.imshow('img',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
