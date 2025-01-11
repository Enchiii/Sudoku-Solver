def individual_serial(sudoku_game) -> dict:
    return {
        "id": str(sudoku_game["_id"]),
        "name": sudoku_game["name"],
        "board": sudoku_game["board"],
    }


def list_serial(sudoku_games) -> list:
    return [individual_serial(sudoku_game) for sudoku_game in sudoku_games]
