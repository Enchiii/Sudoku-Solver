import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
from sudoku_solver import solve_sudoku
from model import Net
from sudoku_recognizer import prepare_images, sudoku_recognizer
from schema.schemas import list_serial
from db_config.database import collection


app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)


class SudokuGames(BaseModel):
    name: str
    board: list[list[str]]


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


@app.post("/board/upload-image")
async def upload_image(file: UploadFile = File(...)):
    path = "./models/m1.pth"
    recognizer = Net()
    recognizer.load_state_dict(torch.load(path, weights_only=True))

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    sudoku_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    sudoku_cells = sudoku_recognizer(sudoku_image)

    classes = ('', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    preds = []
    dLoader = prepare_images(sudoku_cells)
    with torch.no_grad():
        for data, _ in dLoader:
            outputs = recognizer(data)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted)

    board = []
    for y in range(9):
        row = []
        for x in range(9):
            row.append(classes[preds[y * 9 + x]])
        board.append(row)

    return {"Board": board}


@app.get("/games")
def sudoku_games():
    sudoku_games = list_serial(collection.find())
    return sudoku_games
