#Import Libs
import os
import glob
import numpy as np
import sys
import shutil 
import cv2
import time
from PIL import Image

#specify directory folder where .jpg and .csv file pairs are located 
directory = sys.argv[1]

csv_input_path='./' + directory + '/*.csv'

image_input_path='./' + directory + '/*.jpg'

#import file name lists into an array
csv_files = glob.glob(csv_input_path)
image_files = glob.glob(image_input_path)

#sort files in acending number order
csv_files = sorted(csv_files)
image_files = sorted(image_files)
#flength=len(image_files)+1
flength = 1488+1
i=1425

while i < flength:
	#move CSVs and Image files to individual folders	
	os.mkdir('./' + directory + '/' + str(i))
	shutil.move('./' + directory + '/' + str(i) + '.csv' , './' + directory + '/' + str(i))
	shutil.move('./' + directory + '/' + str(i) + '.jpg' , './' + directory + '/' + str(i))
	
	#load in pre-trained mask-rcnn neural network for mask construction  
	labelsPath = './mask-rcnn-coco/object_detection_classes_coco.txt'
	LABELS = open(labelsPath).read().strip().split("\n")
	
	weightsPath = './mask-rcnn-coco/frozen_inference_graph.pb'

	configPath = './mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
	
	
	net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
	
	#load in image from newly made folder from which masks will be generated 
	image = cv2.imread('./' + directory + '/' + str(i) + '/' + str(i) + '.jpg')  
	#find image spatial dimentions 	
	(H, W) = image.shape[:2]
	
	# construct a blob from the input image and then perform a forward
	# pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
	# of the objects in the image along with (2) the pixel-wise segmentation
	# for each specific object
	blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
	end = time.time()
	
	# show timing information and volume information on Mask R-CNN
	print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
	print("[INFO] boxes shape: {}".format(boxes.shape))
	print("[INFO] masks shape: {}".format(masks.shape))
	
	# open csv
	frame_dict = {}
	with open('./' + directory + '/' + str(i) + '/' + str(i) + '.csv', 'rb') as frame:
		lines = [l.decode('utf-8','ignore') for l in f.readlines()]
		for line in lines:
			line = line.strip('\n').split(',')
			x,y,dist = int(line[1]), int(line[2]), float(int(line[0])/1000)
			frame_dict[(x,y)] = dist

	for j in range(0, boxes.shape[2]):
		# extract the class ID of the detection along with the confidence
		# (i.e., probability) associated with the prediction
		classID = int(boxes[0, 0, j, 1])
		confidence = boxes[0, 0, j, 2]
	
		if confidence > 0.5:
			# clone our original image so we can draw on it
			clone = image.copy()
			H,W,C = clone.shape
			label_mask = np.zeros((H,W))
 
			# scale the bounding box coordinates back relative to the
			# size of the image and then compute the width and the height
			# of the bounding box
			box = boxes[0, 0, j, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY
			
			# extract the pixel-wise segmentation for the object, resize
			# the mask such that it's the same dimensions of the bounding
			# box, and then finally threshold to create a *binary* mask
			mask = masks[j, classID]
			mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_NEAREST)
			mask = (mask > 0.3)
			
			# extract the ROI of the image
			roi = clone[startY:endY, startX:endX]
			
			visMask = (mask * 255).astype("uint8")
			
			label_mask[startY:endY, startX:endX][mask] = 255			
			
			candidates = np.argwhere(label_mask == 255).tolist()
			
			distances = []
			for candidate in candidates:
				c = (candidate[1], candidate[0])
				if c in frame_dict.keys():
					distances.append(frame_dict[c])

			im = Image.fromarray(label_mask).convert("L")	
			im.save('./'+directory+'/'+ str(i) +'/mask-item-' +str(j) + '.jpg')
			with open('./'+directory+'/'+ str(i) +'/mask-item-' +str(j) + '.dist','w') as write:
				for d in distances:
					write.write(str(d)+'\n')
				write.write(str(LABELS[classID]) + '\n')
			with open('./'+directory+'/'+ str(i)+'/mask-item-'+str(j) + '.bbox','w') as boxwrite:
				boxwrite.write(','.join([str(startX),str(startY),str(endX),str(endY)]) + '\n')
	i=i+1
