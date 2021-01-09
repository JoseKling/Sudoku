#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:49:11 2021

@author: kling
"""

#%% Importing packages

import numpy as np
import cv2 as cv
import os

#%% Function

def normalize( img, target_heigth, target_width ):
    contours, _ = cv.findContours( cv.Canny( img, 100, 200 ), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
    if len( contours ) > 1:
        return False
    x, y, width, height = cv.boundingRect( contours[0] )
    img = img[ y:y+height, x:x+width ]
    height, width = img.shape
    ratio = target_height/height
    img = cv.resize( img, ( int( width*ratio), target_height ) )
    width = img.shape[1]
    ncols = target_width-width
    fill = np.ones( (target_height, ncols) )*255    
    return( np.hstack( (img, fill) ) )

#%% Script

parent_folder = './dataset/images/'
folders = 'Sample0'
target_height = 30
target_width = 60
dataset = np.zeros( (target_height*target_width) + 1, dtype = int)
for i in range(1,10):
    files = os.listdir( parent_folder + folders + '0' + str(i) )
    for file in files:
        img = cv.imread( parent_folder + folders + '0' + str(i) + '/' + file, cv.IMREAD_GRAYSCALE )
        img = normalize( img, target_height, target_width )
        if not isinstance( img, bool):
            line = np.zeros( (target_height*target_width) +1, dtype = int )
            line[ :target_height*target_width ] = img.flatten()
            line[-1] = i-1
            dataset = np.vstack( (dataset, line ) )
files = os.listdir( parent_folder + folders + '10' )
for file in files:
    img = cv.imread( parent_folder + folders + '10' + '/' + file, cv.IMREAD_GRAYSCALE )
    img = normalize( img, target_height, target_width )
    if not isinstance( img, bool):
        line = np.zeros( (target_height*target_width) + 1, dtype = int )
        line[ :target_height*target_width ] = img.flatten()
        line[-1] = 9
        dataset = np.vstack( (dataset, line ) )    
dataset = dataset[1:]
np.savetxt( 'dataset.csv', dataset, fmt = '%d', delimiter = ',' )
