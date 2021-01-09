#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:02:06 2020

@author: kling
"""

#%% Importing packages

import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier

#%% Functions

def board_from_img( img_path ):
    '''
    Reads the sudoku board in an image, returning a 9x9 integer numpy array.

    Parameters
    ----------
    img_path : STRING
        THE PATH OF THE IMAGE FILE TO BE READ.

    Returns
    -------
    board: 9X9 NUMPY ARRAY
        ARRAY WITH THE VALUES READ IN THE IMAGE

    '''
    #Loads the image
    img = cv.imread( img_path, cv.IMREAD_GRAYSCALE )
    #Crops the image to get only the board
    board_img = extract_board( img )
    #Returns a list of images from the numbers in the cells
    cells = extract_cells( board_img )
    #Interprets the images in the cells to get the actual values
    board = read_digits( cells )
    #Returns the 9x9 sudoku grid with zeros occupying the blank cells
    return( board )
    
def extract_board( img ):
 
    #Simple preprocessing of the image
    # processed_img = cv.Canny( img, 100, 200)
    processed_img = cv.GaussianBlur( img, (5,5), 0.5)
    _, processed_img = cv.threshold( processed_img, 170, 255, cv.THRESH_BINARY)
    #Get the outermost contours
    contours, _= cv.findContours( cv.bitwise_not( processed_img ), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
    #Calculate the area of each contour and keep the largest one
    areas = np.zeros( len(contours) )
    for i, contour in enumerate( contours ):
        area = cv.contourArea( contour )
        areas[ i ] = area
    board_contour = contours[ np.argmax( areas ) ]
    #We are going to use this contour to get the corners of the sudoku board
    #and apply a perspective warp to straighten the image
    epsilon = 0.1*cv.arcLength( board_contour, True )
    approx = cv.approxPolyDP( board_contour, epsilon, True )
    #Now we need to order the corners. We order them in clockwise order, starting
    #with the top left corner.
    #Remember that opencv uses the convetion (x,y) for coordinates
    board_corners = order_corners( approx )
    target_size = 600
    corners = np.float32( [ [0,0], [target_size-1,0], [target_size-1,target_size-1], [0,target_size-1] ] )
    warp_matrix = cv.getPerspectiveTransform( board_corners, corners )
    warped_img = cv.warpPerspective( img, warp_matrix, (target_size,target_size) )
    return( warped_img )

def order_corners( corners ):
    corners = np.array( corners[:,0,:] )
    index = np.argmin( np.sum( corners, axis = 1 ) )
    top_left = corners[ index ]
    corners = np.delete( corners, index, axis = 0 )
    index = np.argmax( np.sum( corners, axis = 1 ) )
    bottom_right = corners[ index ]
    corners = np.delete( corners, index, axis = 0 )
    index = np.argmin( corners[ :, 1]-top_left[1] )
    top_right = corners[ index ]
    corners = np.delete( corners, index, axis=0 )
    bottom_left = corners[0]
    return( np.float32( [ top_left, top_right, bottom_right, bottom_left ]))
    
def extract_cells( board ):
    cell_height, cell_width = board.shape[0]/9, board.shape[1]/9
    #Simple preprocessing of the image
    processed_img = cv.GaussianBlur( board, (5,5), 0.5)
    _, processed_img = cv.threshold( processed_img, 170, 255, cv.THRESH_BINARY)
    # processed_img = cv.Canny( board, 100, 200 )
    #Get the contours of the image, the parent being the whole board
    contours, hierarchy = cv.findContours( cv.bitwise_not( processed_img ), cv.RETR_TREE, cv.CHAIN_APPROX_NONE )
    # contours, hierarchy = cv.findContours( processed_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE )
    parent = np.argmin( hierarchy[0,:,3])
    cells = [ np.zeros( (30,60), dtype = np.uint8 ).flatten() for i in range(81) ]
    for i, contour in enumerate(contours):
        #If it is a child of the whole board, then this is a inner contour
        #of a cell
        if hierarchy[0,i,3] == parent:
            x,y,width,height = cv.boundingRect( contour )
            #If the contour is too small, it is probably just some error due to
            #blur in the image, so we can ignore those
            if width > cell_width/2 or height > cell_height/2:
                if width > 3*cell_width/2 or height > 3*cell_height/2:
                    print( 'The board was probably not read correctly!' )
                #The actual digit is the child of the inner contour of the cell
                index = hierarchy[ 0, i, 2 ]
                #If the inner contour has nothing inside it, then it is an
                #empty cell
                if index == -1:
                    line = np.zeros( (30, 60), dtype = np.uint8).flatten()
                #We keep only the rectangle containing the digit
                else:
                    x,y,width,height = cv.boundingRect( contours[ index ] )
                    digit = board[ y:y+height, x:x+width ]
                    line = normalize_cell( digit ).flatten()
                pos_y = (y+20)//(600/9)
                pos_x = (x+20)//(600/9)
                pos = (pos_y*9)+pos_x
                cells[ int( pos ) ] = line
    #Note that the contours are stored starting from the right-bottom row by row
    #so we invert the order
    return( np.array( cells, dtype = np.uint8 ) )

def order_cells( cells, positon ):
    pass

def read_digits( cells ):
    dataset = np.loadtxt( 'dataset.csv', dtype = np.uint8, delimiter=',' )
    labels = dataset[ :, -1 ]
    features = dataset[ :, :-1 ]
    model = KNeighborsClassifier( )
    model.fit( features, labels )
    non_empty = np.where( np.any( cells != 0, axis=1 ) )
    board = np.zeros( 81, dtype = int )
    board[ non_empty ] = model.predict( cells[ non_empty ] )
    board = np.array( board, dtype = int ).reshape( (9,9) )
    return( board )

def normalize_cell( cell ):
    height, width = cell.shape
    target_height = 30
    target_width = 60
    ratio = target_height/height
    cell = cv.resize( cell, ( int( width*ratio), target_height ) )
    cell = cv.GaussianBlur( cell, (7,7), 1 )
    cell = otsu( cell )
    width = cell.shape[1]
    ncols = target_width-width
    fill = np.ones( (target_height, ncols) )*255    
    return( np.hstack( (cell, fill) ) )

def otsu( img ):
    old_max = 0
    cut = 0
    for i in range(1,256):
        class_1 = img[ img < i ]
        class_2 = img[ img >= i]
        prob_1 = class_1.size/img.size
        mean_1 = np.average( class_1 )
        mean_2 = np.average( class_2 )
        maximize = prob_1*(1-prob_1)*((mean_1-mean_2)**2)
        if maximize > old_max:
           cut = i
           old_max = maximize
    new_img = np.copy(img)
    new_img[ img < cut ] = 0 
    new_img[ img >= cut ] = 255
    return( new_img )