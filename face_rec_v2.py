import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from dbr import *
import cv2
license_key = "t0070fQAAAA/lgr6TSGYD/mYnKxOQQZdL5rCdz5VQaxSlgPnN4Fl+HhYfVISxWKOqL5dTllJaV8zwuNRTPk0MiyivIZVKR0Ia4g=="
reader = BarcodeReader()

# Apply for a trial license: https://www.dynamsoft.com/customer/license/trialLicense?product=dbr&utm_source=github
reader.init_license(license_key)


path = './faces'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	classNames.append(os.path.splitext(cl)[0])

def detect_and_decode(image):

	text_results = reader.decode_buffer(image)

	if text_results != None:
		for text_result in text_results:
			temp = text_result.localization_result.localization_points
			contour = np.array([list(x) for x in temp])
			# font
			font = cv2.FONT_HERSHEY_SIMPLEX
			
			# org
			org = (00, 185)
			
			# fontScale
			fontScale = 0.5
			
			# Red color in BGR
			color = (0, 0, 255)
			
			# Line thickness of 2 px
			thickness = 1

			#text
			text = text_result.barcode_text
			
			# Using cv2.putText() method


			cv2.drawContours(image, [contour], -1, (0, 255, 0), 5)
			cv2.putText(image, text, (contour[0][0], contour[0][1]), font, fontScale, 
				color, thickness, cv2.LINE_AA, False)
	return image 

def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList
 
def markAttendance(name):
	with open('Attendance.csv','r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString = now.strftime('%H:%M:%S')
		f.writelines(f'n{name},{dtString}')

 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
	success, img = cap.read()
	#img = captureScreen()
	imgS = cv2.resize(img,(0,0),None,0.25,0.25)
	imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
	
	facesCurFrame = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
	
	for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
		matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
		faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
		#print(faceDis)
		matchIndex = np.argmin(faceDis)
		
		if matches[matchIndex] > 0.8:
			name = classNames[matchIndex].upper()
		else:
			name = 'Unknown'
			#print(name)
		y1,x2,y2,x1 = faceLoc
		y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
		cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
		cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
		cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

	img_detected = detect_and_decode(img)

	cv2.imshow('Video', img_detected)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break