#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:02:06 2020

@author: kling

This script will take an image of a sudoku board, read it, and return the grid
as a numpy array to be solved with the sudoku_solver.py file.
We use opencv findContours to find the four corners of the largest contour in
the image (that we assume to be the board). With the contours we can perform a
perspective transformation to 'untilt' the image. We then search each of the
largest contours' sub contours, which are cells' contours, and if a cell contour
has any contour inside it, then it is the contour of the digit inside it.
For the digit recognition we simply used the K-neighbors algorithm with the 
'Chars74K dataset' (http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/). The
standard we used for the dataset is to get the bounding rectangle around the
black digit on white background, resize it have height of 30 pixels and add as 
many white columns as necessary for the image to have width 60.
So all images in the dataset are 30x60 (hxw) with the digit to the left.
"""

#%% Importing packages

import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sudoku_aux import *

#%% Functions

def board_from_img( img_path: str ) -> np.array:
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
    try:
        img = cv.imread( img_path, cv.IMREAD_GRAYSCALE )
    except:
        raise Exception('Could not read image file. Check if path to file is correct.')
    #Crops the image to get only the board
    board_img = extract_board( img )
    #Returns a list of images from the numbers in the cells
    cells = extract_cells( board_img )
    #Interprets the images in the cells to get the actual values
    board = read_digits( cells )
    #Returns the 9x9 sudoku grid with zeros occupying the blank cells
    return( board )
    
def extract_board( img: np.array ) -> np.array:
    '''
    Segment out the sudoku board present in the image. 

    Parameters
    ----------
    img : NUMPY ARRAY
        THE IMAGE CONTAINING THE SUDOKU BOARD.

    Returns
    -------
    board: NUMPY ARRAY
        SQUARE IMAGE OF THE SEGMENTED AND WARPED SUDOKU BOARD
    '''
 
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
    if len( approx ) != 4:
        #If the largest shape of the image does not have 4 sides, then it is
        #not a sudoku board
        raise Exception('Unable to find a sudoku board in the image.')
    board_corners = order_corners( approx )
    #The new image will be of size target_size x target_size
    target_size = 600
    corners = np.float32( [ [0,0], [target_size-1,0], [target_size-1,target_size-1], [0,target_size-1] ] )
    #Gets the perspective transform matrix and apply to the image
    warp_matrix = cv.getPerspectiveTransform( board_corners, corners )
    warped_img = cv.warpPerspective( img, warp_matrix, (target_size,target_size) )
    return( warped_img )
    
def extract_cells( board: np.array ) -> np.array:
    '''
    Segments out the digits in each cell. If the cell is empty, returns a white
    30x60 image, else returns a rectangle containing the digit.

    Parameters
    ----------
    board : NUMPY ARRAY
        IMAGE CONTAINING ONLE THE SUDOKU BOARD.

    Returns
    -------
    cells : NUMPY ARRAY
        EACH LINE IS THE FLATTENED 30x60 IMAGE OF THE DIGIT IN THE 
        CORRESPONDING CELL.
    '''
    
    cell_height, cell_width = board.shape[0]/9, board.shape[1]/9
    #Simple preprocessing of the image
    processed_img = cv.GaussianBlur( board, (5,5), 0.5)
    _, processed_img = cv.threshold( processed_img, 170, 255, cv.THRESH_BINARY)
    # processed_img = cv.Canny( board, 100, 200 )
    #Get the contours of the image, the parent being the whole board
    contours, hierarchy = cv.findContours( cv.bitwise_not( processed_img ), cv.RETR_TREE, cv.CHAIN_APPROX_NONE )
    parent = np.argmin( hierarchy[0,:,3])
    cells = [ np.zeros( (30,60), dtype = np.uint8 ).flatten() for i in range(81) ]
    for i, contour in enumerate(contours):
        #If it is a child of the whole board (parent contour), then this is a 
        #inner contour of a cell
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
                #Get the position of the digit, so we can store it in the right order
                pos_y = (y+20)//(600/9)
                pos_x = (x+20)//(600/9)
                pos = (pos_y*9)+pos_x
                cells[ int( pos ) ] = line
    cells = np.array( cells, dtype = np.uint8 )
    if len( cells ) != 81:
        raise Exception('Unable to detect all the cells.')
    return( np.array( cells, dtype = np.uint8 ) )

def read_digits( cells: np.array ) -> np.array:
    '''
    Recognizes each of the digits in the images in cells

    Parameters
    ----------
    cells : NUMPY ARRAY
        AN ARRAY WITH 81 LINES, EACH LINE BEING A FLATTENED 30x60 IMAGE OF THE
        DIGIT IN THE CORRESPONDING CELL.

    Returns
    -------
    board : NUMPY ARRAY
        A 9x9 ARRAY CONTAINING THE SUDOKU BOARD.
    '''
    
    #Loading the dataset
    dataset = np.loadtxt( 'dataset.csv', dtype = np.uint8, delimiter=',' )
    #Splitting the labels from the features
    labels = dataset[ :, -1 ]
    features = dataset[ :, :-1 ]
    #Initializes the model
    model = KNeighborsClassifier( )
    #Fit the model to the dataset
    model.fit( features, labels )
    #We need only to predict the non empty cells, so we need their indexes
    non_empty = np.where( np.any( cells != 0, axis=1 ) )
    if len(non_empty[0])==0:
        raise Exception('Could not read board. All cells were read as empty.')
    board = np.zeros( 81, dtype = int )
    #Get the model prediction
    board[ non_empty ] = model.predict( cells[ non_empty ] )
    board = np.array( board, dtype = int ).reshape( (9,9) )
    return( board )