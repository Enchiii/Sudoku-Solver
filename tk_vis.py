import tkinter as tk
import threading

from tkinter import messagebox
from main import solve_sudoku, is_valid

root = tk.Tk()
root.title("Sudoku solver")


def solve_sudoku_thread(board):
    board = solve_sudoku(board=board)
    root.after(0, update_interface, solved_board)


def check_board(board):
    # checking row
    values = []
    for row in board:
        for num in row:
            if num != 0:
                if num in values:
                    return False
            values.append(num)

    # checking column
    values = []
    for col in range(9):
        for row in board:
            num = row[i]
            if num != 0:
                if num in values:
                    return False
            values.append(num)

    # checking 3x3  squares
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            values = []
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    num = board[row][col]
                    if num != 0:
                        if num in values:
                            return False
                        values.append(num)

    return True


def solve():
    board = []
    for row in entries:
        board_row = []
        for entry in row:
            val = entry.get()
            if val.isdigit() and 1 <= int(val) <= 9:
                board_row.append(int(val))
            elif val == 0 or val == "":
                board_row.append(0)
            else:
                messagebox.showerror(
                    "Input Error", "You can only enter numbers from 1 to 9."
                )
                return
        board.append(board_row)

    if not check_board(board=board):
        messagebox.showerror("Input Error", "Check your inputs!!!")
        return

    threading.Thread(target=solve_sudoku_thread, args=(board,)).start()

    if board is False:
        messagebox.showerror("Input Error", "This sudoku is unsolveable.")
        return

    for i, row in enumerate(entries):
        for j, entry in enumerate(row):
            if entry.get() == "" or entry.get() == 0:
                entry.insert(0, board[i][j])
                entry.config(fg="green")

    return board


def clear():
    for row in entries:
        for entry in row:
            entry.delete(0, tk.END)
            entry.config(fg="black")


frame = tk.Frame(
    root,
    highlightbackground="black",
    highlightcolor="black",
    highlightthickness=2,
    bd=0,
)
frame.grid(row=0, column=0, columnspan=9, padx=30, pady=30)

entries = []
for i in range(9):
    row_entries = []
    for j in range(9):
        entry = tk.Entry(
            frame,
            width=2,
            font=("Arial", 24),
            justify="center",
            highlightbackground="black",
            highlightcolor="black",
            highlightthickness=1,
        )
        entry.config(fg="black")
        entry.grid(row=i, column=j)
        row_entries.append(entry)
    entries.append(row_entries)

pady = (0, 30)

submit_button = tk.Button(
    root, text="solve", command=solve, width=11, height=1, font=("Arial", 12)
)
submit_button.grid(row=1, columnspan=4, column=1, pady=pady)

clear_button = tk.Button(
    root, text="clear", command=clear, width=11, height=1, font=("Arial", 12)
)
clear_button.grid(row=1, columnspan=4, column=4, pady=pady)

tk.mainloop()
