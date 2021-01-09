I'm building a sudoku application that will perform various tasks. For now I just have some individual scripts that will be used later.
I'm using opencv for image processing and segmentation, scikit-learn K-neighbors algorithm for digit recognition and numpy for handling arrays.
Below a brief description of each file/folder

- sudoku_solver
    Implementation of two algorithms for solving sudoku boards. One is a simple backtracking algorithm and the other is crook's algorithm. They solve the board in a random fashion, so by passing a blank board it returns a random complete board.
- sudoku_generator
    Given a complete sudoku board, it generates a minimal board, meaning it returns a board which admits a unique solution and if any of the entries are erased, this uniqueness is lost.
- sudoku_reader
    Given an image of a sudoku board, it reads the board and returns a 9x9 matrix of the board. Used in conjunction with sudoku_solver to solve a game from an image.
- sudoku_aux
    Some auxiliary functions used in previous scripts. Only exception is print_board, that generates an image from a 9x9 matrix with some board.
- create_dataset
    Used only to generate the file dataset.csv from the 'Chars74K' dataset
- dataset.csv
    Dataset used for the training of the digit recognition algorithm
- Numbers (folder)
    Images used for the print_board function in sudoku_aux
