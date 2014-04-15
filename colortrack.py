# Reference : Dalielpert's colortracking git 
#  https://github.com/danlipert/colortracking/blob/master/colorid.py
#   		: documentation http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
# 			: this question on stack overflow http://stackoverflow.com/questions/10948589/choosing-correct-hsv-values-for-opencv-thresholding-with-inranges

import cv2
import numpy as np
import sys


	
def main():
	print ("menu:")
	print ("select an option by entering 1 through 4 to get to the specific functions. once you finish press 5")
	print ("Option 1: Converts live feed to B/W, Resizes Video Frames and Blurs")
	print ("Option 2: follows orange colors. You can compare HSV and RGB with this option")
	print ("Option 3: Edge Detection, Erode and Dilate")
	print ("Option 4: Optic Flow")
	print ("Option 5: face tracking")
	response = input(": ")
	#getting images
	c = cv2.VideoCapture(0)
	width,height = c.get(3),c.get(4)
	print "frame width and height : ", width, height
	if response == 1:
		while(1):
			_,f = c.read() 		#reading the frame to f
			f = cv2.flip(f,1) 	#flipping the frame
			#values for resizing the image
			width  = 320
			height = 240
			f = cv2.resize(f, (width, height))
			cv2.imshow('img',f)	
			fbw = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) #black and white image
			cv2.imshow('B/W',fbw)
			fblur = cv2.medianBlur(fbw, 5) #black and white image
			cv2.imshow('Blur',fblur)
			if cv2.waitKey(25) == 27: #for some reason this was needed
				break
	if response == 2:
		while(1):
			_,f = c.read() 		#reading the frame to f
			f = cv2.flip(f,1) 	#flipping the frame
			width  = 320
			height = 240
			f = cv2.resize(f, (width, height))
			fRGB = f
			#cv2.imshow('fHSV',fHSV)
			#Small bb
			#RGB values from gimp
			#  		B  G  R 			H   S   V
			#Max    99 45 246           344 82 96
			#min	100 33 248
			H = 344/2
			S = 82*(255/100)
			V = 96*(255/100)
			diff = 25
			#tennisRGB = cv2.inRange(fRGB,np.array((0.11*256, 0.60*256, 0)),np.array((0.14*256, 1.00*256, 1.00*256)))
			#tennisHSV = cv2.inRange(fHSV,np.array((0.11*256, 0.60*256, 0)),np.array((0.14*256, 1.00*256, 1.00*256)))
			orangeRGB = cv2.inRange(fRGB,np.array((0.02*256, 0.00*256, .58*256)),np.array((.73*256, .80*256, 1.00*256)))
			cv2.imshow('oRGB',orangeRGB)
			cntRGB, h = cv2.findContours(orangeRGB, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in cntRGB:
				(x, y), radius = cv2.minEnclosingCircle(cnt)
				center = (int(x), int(y))
				radius = int(radius)
				if (radius > 10): # if the radius is greater than 50 pixels then place a circle around it
					circle = cv2.circle(f, center, radius, (0, 255, 0), 2)
			cv2.imshow('RGB',f)	
			
			
			_,f = c.read() 		#reading the frame to f
			f = cv2.flip(f,1) 	#flipping the frame
			width  = 320
			height = 240
			f = cv2.resize(f, (width, height))
			fHSV = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
			H = 344/2
			S = 82*(255/100)
			V = 96*(255/100)
			diff = 25
			orangeHSV = cv2.inRange(fHSV,np.array((H - 10, S - diff, V - diff)),np.array((H + 10,  255, 255)))
			cv2.imshow('oHSV',orangeHSV)
			cntHSV, h = cv2.findContours(orangeHSV, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt2 in cntHSV:
				(x, y), radius = cv2.minEnclosingCircle(cnt2)
				center = (int(x), int(y))
				radius = int(radius)
				if (radius > 10): # if the radius is greater than 50 pixels then place a circle around it
					circle = cv2.circle(f, center, radius, (0, 255, 0), 2)
			cv2.imshow('HSV',f)		
			if cv2.waitKey(25) == 27: #for some reason this was needed
				break
	if response == 3:
		element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,1), (1, 0))
		element2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,1), (1, 0))
		while(1):
			_,f = c.read() 		#reading the frame to f
			f = cv2.flip(f,1) 	#flipping the frame
			width  = 320
			height = 240
			f = cv2.resize(f, (width, height))
			fEdge = cv2.Canny(f, 2000, 4000, apertureSize = 5)    
			cv2.imshow('Edge Detection',fEdge)	
			fErode = cv2.erode(fEdge, element1, 10)# constants from Dalielpert's colortracking git referenced in the begining.
			cv2.imshow('Erode',fErode)	
			fDilate = cv2.dilate(fErode, element2, 4)# constants from Dalielpert's colortracking git referenced in the begining.
			cv2.imshow('Dilate',fDilate)	
			
			
			if cv2.waitKey(25) == 27: #for some reason this was needed
				break
	
	if response == 4:
	#for this part I used the sample code (lk_track.py) that was in the directory \opencv\samples\python2
	#		website: http://docs.opencv.org/trunk/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
	#I modified it a little bit for my purposes. 
	
		track_len = 10
		detect_interval = 5
		tracks = []
		cam = cv2.VideoCapture(0)
		frame_idx = 0
		# params for ShiTomasi corner detection
		feature_params = dict( maxCorners = 100,
							   qualityLevel = 0.3,
							   minDistance = 7,
							   blockSize = 7 )

		# Parameters for lucas kanade optical flow
		lk_params = dict( winSize  = (15,15),
						  maxLevel = 2,
						  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		while True:
			ret, frame = cam.read()
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			vis = frame.copy()
			if len(tracks) > 0:
				img0, img1 = prev_gray, frame_gray
				p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
				p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				d = abs(p0-p0r).reshape(-1, 2).max(-1)
				good = d < 1
				new_tracks = []
				for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
					if not good_flag:
						continue
					tr.append((x, y))
					if len(tr) > track_len:
						del tr[0]
					new_tracks.append(tr)
					cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
				tracks = new_tracks
				cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

			if frame_idx % detect_interval == 0:
				mask = np.zeros_like(frame_gray)
				mask[:] = 255
				for x, y in [np.int32(tr[-1]) for tr in tracks]:
					cv2.circle(mask, (x, y), 5, 0, -1)
				p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
				if p is not None:
					for x, y in np.float32(p).reshape(-1, 2):
						tracks.append([(x, y)])


			frame_idx += 1
			prev_gray = frame_gray
			cv2.imshow('Optic follow', vis)

			ch = 0xFF & cv2.waitKey(1)
			if ch == 27:
				break
	if response == 5:
		#This code was adopted from: http://docs.opencv.org/trunk/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
		# I edited it so that it would work for my purposes
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
		while True:
			_,f = c.read() 		#reading the frame to f
			f = cv2.flip(f,1)
			gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
				cv2.rectangle(f,(x,y),(x+w,y+h),(255,0,0),2)
				roi_gray = gray[y:y+h, x:x+w]
				roi_color = f[y:y+h, x:x+w]
				eyes = eye_cascade.detectMultiScale(roi_gray)
				print "Face stuff: ", x,y,w,h
				for (ex,ey,ew,eh) in eyes:
					if (eh< h/3 and ex> int(0.35*x) and ey< int(0.80*y) ):
						print "Eye stuff: " ,ex,ey,ew,eh
						cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv2.imshow('frame',f)
			if cv2.waitKey(25) == 27: #for some reason this was needed
				break
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	cv2.destroyAllWindows()
	c.release()
main()