def find_next(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col

    return None, None


def is_valid(board, row, col, num):
    if num in board[row]:
        return False

    for i in range(9):
        if board[i][col] == num:
            return False

    row_start = (row // 3) * 3
    col_start = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[row_start + i][col_start + j] == num:
                return False

    return True


def solve_sudoku(board):

    row, col = find_next(board)

    if row is None:
        return board

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return board

        board[row][col] = 0

    return False
