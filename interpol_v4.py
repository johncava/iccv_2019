import os
import glob
import numpy as np
import sys
from scipy.interpolate import griddata
from PIL import Image
import matplotlib.pyplot as plt
import cv2

directory = sys.argv[1]
results = glob.glob(directory+'/*/')
r = [a.split('/')[-2] for a in results]

for p,q in zip(r,results):

    #load in .csv files

    file_ = q + p + '.csv'
	
    distance = np.genfromtxt(file_, delimiter=',', usecols=0)
	
    max_distance = np.max(distance)

    pixel_x = np.genfromtxt(file_, delimiter=',', usecols=1)

    pixel_y = np.genfromtxt(file_, delimiter=',', usecols=2)

    # get the image size (x,y)
    size_x, size_y = Image.open(q + p + '.jpg').size

    # crop extra pixels to match camera frame size
    hcrop=np.where(pixel_y<size_y)

    cropped_x = pixel_x[hcrop]
    cropped_y = pixel_y[hcrop]
    cropped_d = distance[hcrop]    
    

    min_y = np.min(cropped_y)

    max_y = np.max(cropped_y)

    min_x = np.min(cropped_x)
    max_x =	np.max(cropped_x)

    max_d = np.max(cropped_d)

    # interpolate depth data and construct array full     
    xi = np.linspace(min_x, max_x, size_x)
    yi = np.linspace(min_y, max_y, size_y-min_y)

    XI, YI = np.meshgrid(xi, yi)
    points = np.vstack((cropped_x, cropped_y)).T
    values = np.asarray(cropped_d)
    points = np.asarray(points)
    DEM = griddata(points, values, (XI,YI),method='nearest') 
        
    # generate zeros matrix
    zeros_array = np.full((int(min_y), size_x),max_d)         
    
    # add zero array to top of DEM array
    full_array = np.concatenate((zeros_array, DEM), axis=0)
    
    # save array as a color image to specified directory
    scan_name = q + str(p) + '.jpg'
    plt.imsave(scan_name, full_array) 