#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:09:45 2020

@author: kling

STRATEGIES TRIED SO FAR:
For solving:
    - Backtracking
        - Order    
            - In order
            - Randomize the order in which to fill the blanks
    - Guesses
        - Fill with a number and check if it works
        - Fill with the number that works
For generating single solution grids:
    - Add a random number to the empty grid and check if the solution is unique
    - Use the random backtracking to get a initial solution and remove one number at a time
    
IDEAS:
- For generating a random complete grid, start with any complete grid and permute
    rows and columns of squares or rows and columns of the same length 3 segment
    - Can this generate any complete grid?
- Can the run time of the backtracking algorithm be improved by simultaneously
    trying to fill the zeros forward and backwards? And how to do this?
- Fill all the blanks with the possibilities and each time one field is filled
    we update the possibilities. If one of the fields has only one possible
    value, then update this one first.
"""

#%% Import packages

import numpy as np
from sudoku_aux import *
#%% Functions
       
def crooks_solver( board ):
    '''
    This function checks if a sudoku board is solvable and, if possible, solves
    it using Crook's algorithm, described in 'A Pencil-and-Paper algorithm
    for solving sudoku puzzles', found at
    https://www.ams.org/notices/200904/tx090400460p.pdf
    
    Parameters
    ----------
    board : 9x9 numpy array
        THE INITIAL BOARD TO BE SOLVED, REPLACING THE EMPTY CELLS BY ZERO.

    Returns
    -------
    solvable : boolean
        IF THE INITIAL BOARD IS NOT A VALID GAME RETURNS FALSE AND RETURNS TRUE
        OTHERWISE.
    solution : 9x9 numpy array
        THE SOLUTION OF THE GAME. IF THE GAME IS NOT VALID THEN RETURNS THE
        INITIAL BOARD.

    '''
    
    #Checking if there are duplicates in any of the rows, columns
    #or boxes
    if not first_check( board ):
        return( False, board )
    
    
    #The first thing is to get the initial markup
    markup = get_markup( board )
    prev_solution = board
    solution = get_solution( markup )

    #Then we need to  and check if there are any empty marks, meaning there are no
    #possible valid numbers
    if np.any( solution == -1 ):
        return( False, board )
    
    #We will have to keep track of some states
    prev_markups = list( [] )
    prev_choices = list( [] )
    
    #Update the initial markup and see if it is valid
    markup, solution, prev_solution = update_singletons( markup, solution, prev_solution )
    markup, solution, prev_solution = preemptive_markup( markup, solution, prev_solution )
    if np.any( solution == -1 ):
        return( False, board )

    #Now we start the algorithm. 
    #We check if all marks are singletons, if so we have solved the game.
    #If not, then we must choose a random number from some os the cells to
    #continue.
    #If there are empty marks, this means that one of the random choices we 
    #made before is wrong, so we have to backtrack to that point and try
    #another number.
    #Update the markup for singletons and preemptive and go to the first step
    #until we solve the game.
    solved = False
    while not solved:
        
        #If the solution is complete, then we have solved the game
        if np.all( solution > 0 ):
            solution = get_solution( markup )
            solvable = True
            solved = True
    
        #If we haven't solved it, we might have come to a dead end, meaning 
        #one of the cells has no possible numbers
        elif np.any( solution == -1 ): 
            #We have to backtrack to a point we did not run out of options
            go_back = True
            while go_back:
        
                #First we go back to the previous state and erase the value we
                #chose previously, since it did not work
                markup = np.copy( prev_markups[ -1 ] )
                prev_index = prev_choices[-1][0]
                prev_value = prev_choices[-1][1]
                markup[ prev_index ] = [ mark 
                                        for mark in markup[ prev_index ]
                                        if mark != prev_value ]
                
                #If there are no other numbers left, go back to the previous 
                #state and try again
                if len( markup[ prev_index ] ) == 0:
                
                    #If we are at the first state, then we cannot go back, so
                    #this game is unsolvable
                    if len( prev_markups ) == 1:
                        go_back = False
                        solved = True
                        solvable = False
                        solution = board
                    else:
                        prev_markups = prev_markups[ :-1 ]
                        prev_choices = prev_choices[ :-1 ]
                        go_back = True
                
                #If there are numbers left, try another one
                else:
                    prev_markups[-1] = np.copy( markup )
                    solution = get_solution( markup )
                    value = np.random.choice( markup[ prev_index ] )
                    markup[ prev_index ] = [ value ]
                    prev_choices[-1] = ( [ prev_index, value] )
                    go_back = False
            
        #If we have not solved nor reached a dead end, we must try some
        #random number and store this state in case we have to backtrack
        else:
            condition = np.array(np.where( solution == 0 ))
            choice = np.random.choice( np.arange( len( condition[0,:] ) ) )
            index = tuple( condition[:,choice] )
            value = np.random.choice( markup[ index ] )
            prev_markups.append( np.copy( markup ) )
            markup[ index ] = [ value ]
            prev_choices.append( [index, value] )

        #If the puzzle is not solved, we update update the markup for
        #preemptive sets and check again
        if not solved:
            prev_solution = np.copy( solution )
            solution = get_solution( markup )
            markup, solution, prev_solution = update_singletons( markup, solution, prev_solution  )
            markup, solution, prev_solution = preemptive_markup( markup, solution, prev_solution )
            
    return( solvable, solution )
    
def backtrack_solver( board ):
    '''
     Solves the game 'board', if possible, using a simple backtracking algorithm

    Parameters
    board : numpy array
    ----------
        The 9x9 board to be solved. The blank spaces should be replaced with zeros.

    Returns
    -------
    Solvable : A boolean. 'True' meaning the game can be solved.
    Solution : A 9x9 numpy array. If the game is solvable, then the solution is 
    returned, otherwise returns the initial board.

    '''

    #Check if the initial values contains duplicates in any rows, columns or boxes
    if not check_initial( board ):
        return( False, board )
    
    #This is where we will compute the solution
    solution = np.copy( board )
    #This matrix stores the number to be used to fill the corresponding cell
    current_try = np.ones( (9,9), dtype = int )
    #Get the indexes of the empty cells
    zeros = get_zeros( board, shuffle = True )
    current_zero = 0
    
    solved = False    
    while not solved:
        
        if np.all( solution != 0):
            solved = True
            solvable = True
        else:
            index = tuple( zeros[ current_zero , : ] ) 
            taken = np.hstack( (solution[index[0],:], solution[:,index[1]], get_box( solution, index)))
            possible = [ value for value in range(current_try[ index ],10) if value not in taken ]
            if len( possible ) == 0:
                if current_zero == 0:
                    solved = True
                    solvable = False
                    solution = board
                else:
                    solution[ index ] = 0
                    current_try[ index ] = 1
                    current_zero -= 1
            else:
                solution[ index ] = np.min( possible )
                current_try[ index ] += 1
                current_zero += 1

    return( solvable, solution )