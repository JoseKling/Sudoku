#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:11:50 2020

@author: kling
"""

import numpy as np
import cv2 as cv
from itertools import combinations

'''
Function to use in a script
'''

def print_board( board: np.array ):
    '''
    Prints the board.

    Parameters
    ----------
    board : Numpy array
        The board to be printed

    Returns
    -------
    None.

    '''
    
    line_width = 7 
    h_line = np.zeros( (line_width, (128*9) + (10*line_width) ), dtype = np.uint8 )
    v_line = np.zeros( (128, line_width), dtype = np.uint8 )
    img = np.copy( h_line )
    for i in range(9):
        row = np.copy( v_line )
        for j in range(9):
            number = cv.imread( './Numbers/' + str( board[i,j] ) + '.png', cv.IMREAD_GRAYSCALE )
            row = np.hstack( ( row, number, v_line ) )
        row = np.vstack( ( row, h_line ) )
        img = np.vstack( ( img, row ) )
    cv.imshow( 'Board', cv.resize( img, (400,400) ) )
    cv.waitKey()
    cv.destroyAllWindows()

'''
Function used in both 'backtrack_solver' and 'crooks_solver'
'''
    
def first_check( board: np.array ) -> bool:
    '''
    Check if a board have repeated values in any of the lines, columns or squares

    Parameters
    ----------
    board : 9x9 numpy array
        A 9x9 array containing the board, with zeros replacing blank spaces.

    Returns
    -------
    Valid : A boolean. 'True' if the initial condition has no problems

    '''
    
    #Loop through all values fromm 1 to 9
    for value in range(1,10):
        #Loop through each of the 9 rows, columns and boxes
        for step in range(9):
            #Check if this row has duplicate values
            line_ok = ( np.sum( board[ step,:] == value ) < 2 )
            if not line_ok:
                return( False )
            #Check if this column has duplicate values
            column_ok = ( np.sum( board[ :,step] == value ) < 2 )
            if not column_ok:
                return( False )
            #Check if this box has duplicate values
            box = board[ (step//3)*3:((step//3)*3)+3, (step%3)*3:((step%3)*3)+3 ]
            box_ok = ( np.sum( box  == value ) < 2 )
            if not box_ok:
                return( False )
    return( True )

def get_box( board: np.array, index: tuple ) -> np.array:
    '''
    Provides the indexes of the 3x3 square for the corresponding index

    Parameters
    ----------
    index : tuple
        THE INDEX OF THE ELEMENT IN QUESTION.
        
    Returns
    -------
    box : numpy array of length 9
        FLATTENED ARRAY OF THE BOX CORRESPONDING TO 'INDEX'

    '''
    box_row = (index[0]//3)*3 
    box_col = (index[1]//3)*3
    return( board[ box_row:box_row+3, box_col:box_col+3 ].reshape(9) )

'''
Function used only by the backtrack_solver
'''

def get_zeros( board: np.array, shuffle: bool = True):
    indexes = np.array(np.where( board == 0))
    if indexes.size == 0:
        return( np.array([[],[]]))
    zeros = np.array([ np.array(indexes[:,i]) for i in range(len( indexes[0] )) ])
    if shuffle:
        np.random.shuffle( zeros )
    return( zeros )

'''
Functions used only by the crooks_solver
'''

def get_markup( board: np.array ) -> list:
    '''
    Provides the initial markup, looking only for values that are already in 
    the same row, column or box

    Parameters
    ----------
    board : Numpy array
        The board from which we will get the markup

    Returns
    -------
    markup : A 9x9 matrix, each entry consisting of a 1D array containing the 
    possible numbers for that cell.
    '''
    
    #Initialize the markup matrix
    markup = np.zeros( (9,9), dtype = np.ndarray )
    
    #Loop through each row
    for row in range(9):
        #Loop through each column
        for col in range(9):
            #If this is not an empty cell, the markup is a single value
            if board[ row, col ] != 0:
                markup[ row, col ] = [ board[ row, col ] ]
            #If it is empty, get what values were not use in any of the 
            #corresponding row, column and box
            else:
                row_values = board[ row , :]
                col_values = board[ : , col]
                box_values = get_box( board, tuple((row , col)) )
                taken = np.hstack( (row_values , col_values, box_values ) )
                markup[ row , col ] = [ i for i in range(1,10) if i not in taken ]

    return( markup )

def get_solution( markup: list ) -> np.array:
    '''
    Provides the solution based on the current markup. If the cell markup is
    empty, returns -1, if it is a singleton (it is solved) returns this value,
    if it has more than one value, returns 0.

    Parameters
    ----------
    markup : 9x9 numpy array of arrays
        EACH POSITION CONTAINS THE MARKUP FOR THE CORRESPONDING CELL.

    Returns
    -------
    solution : 9X9 numpy array
        PARTIAL SOLUTION. ZERO MEANING THERE ARE MORE THEN ONE POSSIBLE VALUES
        AND -1 IF THERE ARE NO POSSIBLE VALUES FOR THE CORRESPONDING CELL.
    '''
    
    #Initialize solution matrix
    solution = np.zeros( (9,9), dtype = int )
    
    #Loop through each row
    for row in range(9):
        #Loop through each column
        for col in range(9):
            #If the mark is a singleton, then the solution is that value
            if len( markup[ row, col ] ) == 1:
                solution[ row, col ] = markup[ row, col ][0]
            #If the mark is empty, the solution contains -1
            elif len( markup[ row, col ] ) == 0:
                solution[ row, col ] = -1
            #If it has more than 1 values, returns 0
            else:
                solution[ row, col ] = 0
    return( solution )

def preemptive_markup( markup: list, solution: np.array, prev_solution: np.array ) -> list:
    '''
    Loops through every possible row, column and box and every possible 
    combination of 2 to 8 indexes and check if we have a preemptive set. If so,
    update the markup.
    Keep doing this until this process do not make any changes to the markup.

    Parameters
    ----------
    markup : 9x9 numpy array of arrays
        EACH POSITION CONTAINS THE MARKUP FOR THE CORRESPONDING CELL.
    solution : 9X9 numpy array
        PARTIAL SOLUTION. ZERO MEANING THERE ARE MORE THEN ONE POSSIBLE VALUES
        AND -1 IF THERE ARE NO POSSIBLE VALUES FOR THE CORRESPONDING CELL.
    prev_solution : 9x9 numpy array
        SAME AS SOLUTION, BUT FOR A PREVIOUS STATE.
        
    Returns
    -------
    markup : A 9x9 matrix, each entry consisting of a 1D array containing the 
    possible numbers for that cell.

    '''
    
    #If something has changed during the process, we do this again, until no
    #changes are made
    changed = True
    while changed:
        
        #We set this to False and, if any changes are made, we update it.
        changed = False
        
        #Loop through all possible combination sizes
        for length in range(8,1,-1):
            #Loop through every row, column and box
            for i in range(9):
                #Loop through every combination of size 'length'
                for combination in combinations( range(9), length ):
                    index_in = np.array( combination )
                    #Check i-th line
                    row_marks = markup[ i, : ]
                    row_lengths = [ len(row_marks[i]) for i in range(9) ]
                    #We do not consider preemptive sets that contain singletons
                    if np.all( np.take( row_lengths, index_in ) > 1 ):
                        #If check is 'True', then we have found a preemptive
                        #set, and some marks in 'markup' must be updated
                        check, marks = check_preemptive( row_marks, index_in )
                        if check == True:
                            changed = True
                            markup[ i, : ] = marks
                            prev_solution= np.copy( solution )
                            solution = get_solution( markup )
                            markup, solution, prev_solution = update_singletons( markup, solution, prev_solution)
                    #Check i-th column
                    col_marks = markup[ :, i ]
                    col_lengths = [ len(col_marks[i]) for i in range(9) ]
                    if np.all( np.take( col_lengths, index_in ) > 1 ):
                        check, marks = check_preemptive( col_marks, index_in )
                        if check == True:
                            changed = True
                            markup[ :, i ] = marks
                            prev_solution= np.copy( solution )
                            solution = get_solution( markup )
                            markup, solution, prev_solution= update_singletons( markup, solution, prev_solution)
                    #Check i-th box
                    box_row, box_col = (i//3)*3, (i%3)*3
                    box_marks = get_box( markup, tuple( ( box_row, box_col ) ))
                    box_lengths = [ len(box_marks[i]) for i in range(9) ]
                    if np.all( np.take( box_lengths, index_in ) > 1 ):
                        check, marks = check_preemptive( box_marks, index_in )
                        if check == True:
                            changed = True
                            markup[box_row:box_row+3, box_col:box_col+3] = marks.reshape((3,3))
                            prev_solution= np.copy( solution )
                            solution = get_solution( markup )
                            markup, solution, prev_solution= update_singletons( markup, solution, prev_solution)

    return( markup, solution, prev_solution )

def check_preemptive( marks: np.array, index_in: np.array ) -> (bool, np.array):
    '''
    Check if the set of 'marks' given by 'index_in' is as preemptive set, that
    is, if the union of the values in these indexes has the same amount of
    members as the amount of indexes and any of these values appear in any
    marks outside the set.

    Parameters
    ----------
    marks : numpy array of length 9
        VALUES OF A ROW, COLUMN OR BOX.
    index_in : numpy array
        THE INDEXES OF 'MARKS' FOR US TO CHECK IF THEY ARE A PREEMPTIVE SET.

    Returns
    -------
    is_preemptive : boolean
       'TRUE' IF IT IS A PREEMPTIVE SET, 'FALSE' OTHERWISE.
    new_marks : numpy array of length 9
        IF THE SET IS PREEMPTIVE, UPDATE THE MARKUP, IF NOT, RETURN THE MARKUP 
        UNCHANGED
    '''
    
    #Get the amount of indexes
    length = len( index_in )
    
    #Get the indexes that are outside of the set to be checked
    index_out = [ i for i in range(9) if i not in index_in ]

    #Get the union of the values in the cells of 'index_in'
    values = np.unique( np.hstack( marks[ index_in ] ) )
    
    #If the size of the union and the amount of indexes is different, then
    #this is not a preemptive set
    if len( values ) != length:
        return( False, marks )
    
    #Otherwise, we have a preemptive set
    else:
        new_marks = np.copy( marks )
        #Erase the values in the union from the indexes outside the set
        for index in index_out:
            new_marks[ index ] = [ mark 
                                  for mark in marks[ index ]
                                  if mark not in values ]
    
    #Check if we have actually made any changes to the markup
    marks_length = [ len(marks[i]) for i in range(9) ]
    newmarks_length = [ len(new_marks[i]) for i in range(9) ]
    if np.array_equal( marks_length, newmarks_length ):
        return( False, marks )
    else:
        return( True, new_marks )

def update_singletons( markup: list, solution: np.array, prev_solution: np.array ) -> (list, np.array, np.array):
    '''
     Update the markup and solutions considering only the new singletons.  

    Parameters
    ----------
    markup : 9x9 numpy array of arrays
        EACH POSITION CONTAINS THE MARKUP FOR THE CORRESPONDING CELL.
    solution : 9X9 numpy array
        PARTIAL SOLUTION. ZERO MEANING THERE ARE MORE THEN ONE POSSIBLE VALUES
        AND -1 IF THERE ARE NO POSSIBLE VALUES FOR THE CORRESPONDING CELL.
    prev_solution : 9x9 numpy array
        SAME AS SOLUTION, BUT FOR A PREVIOUS STATE.

    Returns
    -------
    markup, solution, prev_solution : updated arrays.

    '''
    
    #We must update the markups referring to all singletons until these
    #updates do not change the markup
    changed = True
    while changed:
        
        #Get the indexes of the new singletons
        changes = (solution>0) & (prev_solution == 0)
        indexes = np.array( np.where( changes ) )
    
        #Loop through each of the indexes
        for i in range(len(indexes[0])):
            index = tuple( (indexes[:,i]) )
            
            #During the loop, some of the cells to be updated may have been
            #erased, so ensure the value is still there
            if len( markup[ index ] ) == 1:
                value = markup[ index ][0]
                markup = update_markup( index, value, markup )
        
        #Update the solutions 
        prev_solution = np.copy( solution )
        solution = get_solution( markup )
        
        #Check if the update changed anything, if so, go through the loop
        #again, if not, we are done
        changed = not( np.array_equal( solution, prev_solution ) )
        
    return( markup, solution, prev_solution )

def update_markup( index: tuple, value: int, markup: list ) -> list:
    '''
    Sets the value for index as value, updating the markup accordingly

    Parameters
    ----------
    index : tuple
        INDEX FOR THE VALUE TO BE INCLUDED.
    value : int
        VALUE TO BE INCLUDED IN THE SOLUTION.
    markup : 9x9 numpy array of arrays
        EACH POSITION CONTAINS THE MARKUP FOR THE CORRESPONDING CELL.

    Returns
    -------
    markup : updated array
    '''
    row = index[0]
    col = index[1]
    #Loops through each of the 9 cells from the corresponding row, column and box.
    for i in range(9):
        #Erase the entry 'value' from the marks of this row
        markup[ row , i ] = [ mark 
                             for mark in markup[ row , i ] 
                             if mark != value ] 
        #Erase the entry 'value' from the marks of this column
        markup[ i , col ] = [ mark 
                             for mark in markup[ i , col ] 
                             if mark != value ]
        box_row = (index[0]//3)*3
        box_col = (index[1]//3)*3
        #Erase the entry 'value' from the marks of this box
        markup[ box_row+(i//3) , box_col+(i%3) ] = [ mark 
                                                     for mark in markup[ box_row+(i//3) , box_col+(i%3) ]
                                                     if mark != value ]
    
    #Set the mark for the index as a singleton
    markup[ index ] = [ value ]

    return( markup )

'''
Functions used in sudoku_reader.py
'''

def normalize_cell( cell: np.array ) -> np.array:
    '''
    Resize the digit image so it has height 30, apply otsu's algorithm for 
    binarization and fill with white columns until the image has width 60.

    Parameters
    ----------
    cell : np.array
        IMAGE OF THE DIGIT IN THE CELL.

    Returns
    -------
    normalized : NUMPY ARRAY
        30x60 IMAGE, THE DIGIT TO THE LEFT.
    '''
    
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

def otsu( img: np.array ) -> np.array:
    '''
    APPLY oTSU'S ALGORITHM FOR BINARIZATON OF THE IMAGE'

    Parameters
    ----------
    img : np.array
        ORIGINAL IMAGE.

    Returns
    -------
    new_img : np.array
        BINARY IMAGE
    '''
    
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

def order_corners( corners: np.array ) -> np.array:
    '''
    Takes the coordinates of the corners of a square in any order and order
    them in clockwise order, starting from the top left.

    Parameters
    ----------
    corners : NUMPY ARRAY
        CORNER COORDINATES AS GIVEN BY OPENCV approxPoly FUNCTION.

    Returns
    -------
    ordered: NUMPY ARRAY
        A 4X2 NUMPY ARRAY, EACH LINE BEING THE COORDINATE OF ONE OF THE CORNERS
    '''
    
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