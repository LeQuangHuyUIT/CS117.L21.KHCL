from dbr import *
import cv2
import numpy as np
license_key = "t0070fQAAAA/lgr6TSGYD/mYnKxOQQZdL5rCdz5VQaxSlgPnN4Fl+HhYfVISxWKOqL5dTllJaV8zwuNRTPk0MiyivIZVKR0Ia4g=="
reader = BarcodeReader()

# Apply for a trial license: https://www.dynamsoft.com/customer/license/trialLicense?product=dbr&utm_source=github
reader.init_license(license_key)

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

def test_detect_and_decode(image_path):
	img = cv2.imread(image_path)
	new_img = detect_and_decode(img)
	cv2.imshow("result", new_img)
	cv2.waitKey(0)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

success, img = cap.read()
while success:
	success, img = cap.read()

	img_detected = detect_and_decode(img)
	cv2.imshow('Result', img_detected)
	if cv2.waitKey(1) == ord("q"):
		break
cap.release()
cv2.destroyAllWindows()
