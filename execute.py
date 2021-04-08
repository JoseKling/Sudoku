#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script let's the user choose one option among: detecting a sudoku board
from the camera; detect a board from an image file; type a board manually;
generate a random board. The program them displays the result and asks if the
user wants to solve the board, possibly with correction made by the user. 
"""
#%% Pachages

import numpy as np
import cv2 as cv
from sudoku_solver import crook
from sudoku_reader import board_from_img
from sudoku_generator import generate, unique
from sudoku_aux import print_board

#%% Script

def input_int(prompt, range_vals):
    incorrect = True
    while incorrect:
        choice = input(prompt)
        incorrect = False
        try:
            int(choice)
        except:
            print('Must be an integer from {} to {}'
                  .format(range_vals[0], range_vals[1]))
            incorrect = True
        if not incorrect:
            if choice < range_vals[0] or choice > range_vals[1]:
                print('Must be an integer from {} to {}'
                  .format(range_vals[0], range_vals[1]))
                incorrect = True
    return(choice)

choice = input_int('Choose one of the options:\n' +
                   '1 - Read the board from the camera.\n' +
                   '2 - Read the board from an image file.\n' +
                   '3 - Manually provide the values in the board.\n' +
                   '4 - Generate a random sudoku board (takes some time).\n' +
                   '5 - Exit.', (1,4))
    
#Exceptions?
if choice == 1:
    print("Press any key to take the picture.")
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv.imshow('Camera', frame)
        if cv.waitKey(1):
            path = './img.jpg'
            cv.imwrite(path, frame)
            break
    board = board_from_img(path)
    print_board(board)
    
elif choice == 2:
    path = input('What is the complete path to the image?')
    board = board_from_img(path)
    print_board(board)
    
elif choice == 3:
    board = np.zeros((9,9), dtype = int)
    print('Type each number in the board, starting from the top left cell and' +
          ' going down row by row. Type "0" or leave it blank for blank cells.')
    for line in range(9):
        for row in range(9):
            cell = input_int('What is the number in cell at row {} and column {}'
                         .format(line+1, row+1), (1,9))
            if cell == '':
                cell = 0
            #Catch exception                        
            board[line, row] = int(cell)
    print_board(board)
    
elif choice == 4:
    solution = crook()
    board = generate(solution)
    print_board(board)
    
else:
    exit()

choice = input_int('Choose one of the options:\n' +
               '1 - Solve the board.\n' +
               '2 - Correct errors in the board before solving.\n' +
               '3 - Exit.', (1,3))

if choice == 3:
    exit()
elif choice == 2:
    again = True
    while again:
        row = input_int('What is the row of the incorrect cell?', (1,9))
        col = input_int('What is the column of the incorrect cell?', (1,9))
        value = input_int('What is the correct value?', (1,9))
        board[row-1, col-1] = value
        print_board(board)
        again = input_int('Is there any other incorrect value? (0-yes, 1-no)', (0,1))
        if again == 1:
            again = False
        
solution = crook(board)
print_board(solution)
exit()
