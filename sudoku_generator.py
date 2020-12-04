#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:07:35 2020

@author: kling
"""

from sudoku_solver import crook
from sudoku_aux import get_zeros
import numpy as np

def generate( solution ):
    '''
    Generates a random minimal sudoku board from the 'solution' provided.
    Each iteration it erases one of the cells from 'solution' and checks if
    this new board has a unique solution. If yes, then we proceed to the next
    cell, if not, then we reinsert the value and erase some other cell, if
    we tried all cells and after erasing all of them we ended with a board
    that has multiple solutions, then we are done.

    Parameters
    ----------
    solution : 9x9 integer numpy array
        A COMPLETE SUDOKU BOARD.

    Returns
    -------
    game : 9x9 integer numpy array
        A MINIMAL SUDOKU BOARD, ZEROS MEANING AN EMPTY CELL.

    '''
    
    game = np.copy( solution )
    #List all possible indexes of the board in a random order
    indexes = np.array( [ [i,j] for i in range(9) for j in range(9) ] )
    np.random.shuffle( indexes )
    #This will tell us if we have tested all possible cells
    tries = 0

    end = False
    while not end:
        
        if tries == len( indexes ):
            #This condition is reached if we tried erasing all of the remaining
            #cells and for all of them we got a non unique solution
            end = True
        else:
            #Get the next index of our list and erase the corresponding cell
            index = tuple( indexes[0,:] )
            game[ index ] = 0
            #Check if this new board has a unique solution
            is_unique, _ = unique( game, solution )
            if is_unique:
                #If so, we reset out 'tries' counter and erase this index from
                #our list of non empty cells
                tries = 0
                indexes = indexes[ 1: ]
            else:
                #If not, we restore the value, update the 'tries' counter and
                #go to the next index
                game[ index ] = solution[ index ]
                indexes = np.roll( indexes, -1, axis=0 )
                tries += 1
    return( game )

def unique( board, solution ):
    '''
    Checks if 'board' admits any solution other than 'solution'.
    Simply go through all of the empty cells of 'board' and try all values
    different from the one in 'solution'


    Parameters
    ----------
    board : Numpy array
        A 9x9 numpy array containing the board board with zeros replacing the
        blank spaces.
    solution : Numpy array
        A solution for the game 

    Returns
    -------
    unique : A boolean. 'True' meaning that the solution provided is the only one. 
    second_solution : A 9x9 numpy array. If unique is 'True', then this is just
    the initial condition, otherwise this is a different solution. 

    '''
    
    #Lists all the indexes of empty cells in a random order
    zeros = get_zeros( board )
    
    #Go through all empty cells
    for line in zeros:
        modified = np.copy( board )
        index = tuple( line )
        #List all values that do not coincide with the one in 'solution'
        possibilities = np.array( [ value for value in range(1,10) if value != solution[ index ] ] )
        np.random.shuffle( possibilities )
        #Go through all values
        for value in possibilities:
            #Construct a initial board by using 'board' and adding the 'value'
            #to the current cell and check if this board is solvable
            modified[ index ] = value 
            solvable, second_solution = crook( modified )
            if solvable:
                #If any of the modifications is solvable, then we have found
                #a new solution to this board
                return( False, second_solution )
        
    return( True, board )