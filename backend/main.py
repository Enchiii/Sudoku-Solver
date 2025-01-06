from pydantic import BaseModel, conlist
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sudoku_solver import solve_sudoku

app = FastAPI()

origins = [
    # "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)


class SudokuInput(BaseModel):
    grid: conlist(conlist(str, min_length=9, max_length=9), min_length=9, max_length=9)


@app.post("/board/solve")
async def solve(board: SudokuInput):
    board = board.grid

    board = [[int(x) if x else 0 for x in row] for row in board]

    solved_board = solve_sudoku(board)

    if solved_board is False:
        return {"Solved": False, "Board": board}

    return {"Solved": True, "Board": solved_board}
