import React, { useState, useEffect } from "react";
import "./App.css";

const url = "http://127.0.0.1:8000";

const Cell = ({ value, onChange, rowIndex, colIndex, highlight }) => {
  return (
    <input
      type="text"
      maxLength="1"
      value={value}
      onChange={(e) => onChange(rowIndex, colIndex, e.target.value)}
      className={`cell ${highlight ? "highlight" : ""}`}
    />
  );
};

const Grid = ({ grid, onCellChange, highlights }) => {
  return (
    <div className="sudoku-grid">
      {grid.map((row, rowIndex) => (
        <div key={rowIndex} className="row">
          {row.map((value, colIndex) => {
            const isHighlighted = highlights.some(
              (highlight) =>
                highlight.row === rowIndex && highlight.col === colIndex
            );
            return (
              <Cell
                key={colIndex}
                value={value}
                onChange={onCellChange}
                rowIndex={rowIndex}
                colIndex={colIndex}
                highlight={isHighlighted}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
};

const predefinedPuzzle = [
  ["5", "3", "", "", "7", "", "", "", ""],
  ["6", "", "", "1", "9", "5", "", "", ""],
  ["", "9", "8", "", "", "", "", "6", ""],
  ["8", "", "", "", "6", "", "", "", "3"],
  ["4", "", "", "8", "", "3", "", "", "1"],
  ["7", "", "", "", "2", "", "", "", "6"],
  ["", "6", "", "", "", "", "2", "8", ""],
  ["", "", "", "4", "1", "9", "", "", "5"],
  ["", "", "", "", "8", "", "", "7", "9"]
];

function App() {
  const [grid, setGrid] = useState(Array(9).fill(Array(9).fill("")));
  const [highlights, setHighlights] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const savedGrid = getCookie("sudokuGrid");
    if (savedGrid) {
      setGrid(JSON.parse(savedGrid));
    }
  }, []);

  const getCookie = (name) => {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
    return null;
  };

  const setCookie = (name, value) => {
    const date = new Date();
    date.setTime(date.getTime() + 15 * 60 * 1000);
    document.cookie = `${name}=${value};expires=${date.toUTCString()};path=/`;
  };

  const findConflicts = (board, row, col, num) => {
    const conflicts = [];

    for (let i = 0; i < 9; i++) {
      if (board[row][i] === num && i !== col) {
        conflicts.push({ row, col: i });
      }
    }

    for (let i = 0; i < 9; i++) {
      if (board[i][col] === num && i !== row) {
        conflicts.push({ row: i, col });
      }
    }

    const rowStart = Math.floor(row / 3) * 3;
    const colStart = Math.floor(col / 3) * 3;
    for (let i = rowStart; i < rowStart + 3; i++) {
      for (let j = colStart; j < colStart + 3; j++) {
        if (board[i][j] === num && (i !== row || j !== col)) {
          conflicts.push({ row: i, col: j });
        }
      }
    }

    if(conflicts.length > 0){
      conflicts.push({row: row, col: col})
    }

    return conflicts;
  };

  const handleChange = (row, col, value) => {
    if (/^[1-9]?$/.test(value)) {
      const updatedGrid = grid.map((r, i) =>
        i === row ? r.map((c, j) => (j === col ? value : c)) : r
      );
      setGrid(updatedGrid);
      setCookie("sudokuGrid", JSON.stringify(updatedGrid));

      const conflicts = value ? findConflicts(updatedGrid, row, col, value) : [];

      setHighlights(conflicts);
    }
  };

  const handleSolve = async () => {
    setError(null);

    try {
      const response = await fetch(url + "/board/solve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ grid }),
      }).catch(err => {
        throw new Error("Failed to solve Sudoku");
      });

      const data = await response.data;

      if (data.Solved) {
        setGrid(data.Board);
      } else {
        setError("No solution found for the given Sudoku.");
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLoadPuzzle = () => {
    setGrid(predefinedPuzzle);
  };

  const handleClearGrid = () => {
    setGrid(Array(9).fill(Array(9).fill("")));
    setHighlights([]);
    document.cookie = "sudokuGrid=;expires=Thu, 01 Jan 1970 00:00:00 GMT"; // Usuwamy zapisany grid
  };

  return (
    <>
      <h1>Sudoku Solver</h1>
      <div className="App">
        <Grid grid={grid} onCellChange={handleChange} highlights={highlights} />
        <div className="controls">
          <button className={"pink-neon"} onClick={handleSolve}>
            Solve Sudoku
          </button>
          <button className={"green-neon"} onClick={handleClearGrid}>
            Clear Grid
          </button>
          <button className={"yellow-neon"} onClick={handleLoadPuzzle}>
            Load Puzzle
          </button>
        </div>
        {error && <p className={"error red-neon"}>Error: {error}</p>}
      </div>
    </>
  );
}

export default App;
