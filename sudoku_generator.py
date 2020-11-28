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
    Each iteration it erases one of the cells from 'solution' and checks 

    Parameters
    ----------
    solution : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    game = np.copy( solution )
    indexes = np.array( [ [i,j] for i in range(9) for j in range(9) ] )
    np.random.shuffle( indexes )
    tries = 0

    end = False
    while not end:
        if tries == len( indexes ):
            end = True
        else:
            index = tuple( indexes[0,:] )
            game[ index ] = 0
            is_unique, _ = unique( game, solution )
            if is_unique:
                tries = 0
                indexes = indexes[ 1: ]
            else:
                game[ index ] = solution[ index ]
                indexes = np.roll( indexes, -1, axis=0 )
                tries += 1
    return( game )

def unique( board, solution ):
    '''
    Checks if 'board' admits any solution other than 'solution'

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
    
    zeros = get_zeros( board )
    #We simply add a blank space with a different value from the solution
    #to the initial condition and check if this game is solvable. If it is, then
    #the solution is not unique. If none of the modifications result in a solvable
    #game, then this solution is unique. 
    for line in zeros:
        modified = np.copy( board )
        index = tuple( line )
        possibilities = np.array( [ value for value in range(1,10) if value != solution[ index ] ] )
        np.random.shuffle( possibilities )
        for value in possibilities:
            modified[ index ] = value 
            solvable, second_solution = crook( modified )
            if solvable:
                return( False, second_solution )
        
    return( True, board )