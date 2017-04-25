#based on this tutorial: https://www.youtube.com/watch?v=88HdqNDQsEk
#CUDA Programming Python: https://developer.nvidia.com/how-to-cuda-python
import cv2
import sys
import os
import numpy as py
import time
import threading

#grab command line arguments
#file_name = sys.argv[1]
#cascade files http://alereimondo.no-ip.org/OpenCV/34
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
leye_cascade = cv2.CascadeClassifier('haar/ojoI.xml')
reye_cascade = cv2.CascadeClassifier('haar/ojoD.xml')
nose_cascade = cv2.CascadeClassifier('haar/Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('haar/Mouth.xml')

num_threads = int(sys.argv[1])

def measureBiometrics():
	return

def thread_func(rank, num_files):
	offset = int(((num_files/num_threads)*rank))
	end = int(offset+(num_files/num_threads))
	for i in range(offset, end):
		file_name = "test_images/{0}.pgm".format(i)
		print("Thread {0} processing {1}...".format(rank, file_name))
		img = cv2.imread(file_name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
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

		cv2.imwrite(('output_images/{0}.pgm').format(i),img)
	
	print("Thread {0} is done!".format(rank))
		
def main():
	num = 1
	start_time = time.time()
	num_files = len(os.listdir("test_images"))
	threads = []

	print("Initiating {0} threads...".format(num_threads))
	for i in range(0, num_threads):
		threads.append(threading.Thread(target = thread_func, args=(i,num_files)))
	print("Done!")

	
	for i in range(0, num_threads):
		print("Running thread {0}...".format(i))
		threads[i].start()
	
	for i in range(0, num_threads):
		threads[i].join()

	print("Runtime: %s seconds" %(time.time()-start_time))


if __name__ == "__main__":
	main()
