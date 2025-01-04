import tkinter as tk
import threading

from tkinter import messagebox
from sudoku_solver import solve_sudoku


class TkSudoku:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sudoku solver")

        self.entries = None
        self.pady = (0, 30)
        self.frame = tk.Frame(
            self.root,
            highlightbackground="black",
            highlightcolor="black",
            highlightthickness=2,
            bd=0,
        )

    def generate_board(self, board=None):
        self.entries = []
        for i in range(9):
            row_entries = []
            for j in range(9):
                entry = tk.Entry(
                    self.frame,
                    width=2,
                    font=("Arial", 24),
                    justify="center",
                    highlightbackground="black",
                    highlightcolor="black",
                    highlightthickness=1,
                )

                entry.config(fg="black")
                if board is not None and board[i][j] not in ('0', '#'):
                    entry.insert(0, board[i][j])

                row_entries.append(entry)
            self.entries.append(row_entries)

    @staticmethod
    def solve_sudoku_thread(board):
        solve_sudoku(board=board)

    @staticmethod
    def check_board(board):
        # checking row
        for row in board:
            values = []
            for num in row:
                if num != 0:
                    if num in values:
                        return False
                values.append(num)

        # checking column
        for col in range(9):
            values = []
            for row in board:
                num = row[col]
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

    def solve(self):
        board = []
        for i, row in enumerate(self.entries):
            board_row = []
            for j, entry in enumerate(row):
                val = entry.get()
                if val.isdigit() and 1 <= int(val) <= 9:
                    board_row.append(int(val))
                elif val == 0 or val == "":
                    board_row.append(0)
                else:
                    messagebox.showerror(
                        "Input Error",
                        f"You can only enter numbers from 1 to 9. (at row: {i + 1} and column: {j + 1})",
                    )
                    return
            board.append(board_row)

        if not self.check_board(board=board):
            messagebox.showerror("Input Error", "Check your inputs!!!")
            return

        threading.Thread(target=self.solve_sudoku_thread, args=(board,)).start()

        if board is False:
            messagebox.showerror("Input Error", "This sudoku is unsolveable.")
            return

        for i, row in enumerate(self.entries):
            for j, entry in enumerate(row):
                if entry.get() == "" or entry.get() == 0:
                    entry.insert(0, board[i][j])
                    entry.config(fg="green")

        return board

    def clear(self):
        if not self.entries:
            raise ValueError("entries must be set by generate_board function")

        for row in self.entries:
            for entry in row:
                entry.delete(0, tk.END)
                entry.config(fg="black")

    def main_loop(self):
        if not self.entries:
            raise ValueError("entries must be set by generate_board function")

        self.frame.grid(row=0, column=0, columnspan=9, padx=30, pady=30)

        for i in range(9):
            for j in range(9):
                self.entries[i][j].grid(row=i, column=j)

        submit_button = tk.Button(
            self.root, text="solve", command=self.solve, width=11, height=1, font=("Arial", 12)
        )
        submit_button.grid(row=1, columnspan=4, column=1, pady=self.pady)

        clear_button = tk.Button(
            self.root, text="clear", command=self.clear, width=11, height=1, font=("Arial", 12)
        )
        clear_button.grid(row=1, columnspan=4, column=4, pady=self.pady)

        tk.mainloop()


if __name__ == '__main__':
    app = TkSudoku()
    app.generate_board()
    app.main_loop()
