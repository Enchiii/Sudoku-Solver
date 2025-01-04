import React, { useState } from "react";
import "./App.css";

const url = "http://127.0.0.1:8000";

const Cell = ({ value, onChange, rowIndex, colIndex }) => {
  return (
    <input
      type="text"
      maxLength="1"
      value={value}
      onChange={(e) => onChange(rowIndex, colIndex, e.target.value)}
      className="cell"
    />
  );
};

const Grid = ({ grid, onCellChange }) => {
  return (
    <div className="sudoku-grid">
      {grid.map((row, rowIndex) => (
        <div key={rowIndex} className="row">
          {row.map((value, colIndex) => (
            <Cell
              key={colIndex}
              value={value}
              onChange={onCellChange}
              rowIndex={rowIndex}
              colIndex={colIndex}
            />
          ))}
        </div>
      ))}
    </div>
  );
};

// Predefined Sudoku Puzzle (Example)
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
  const [error, setError] = useState(null);

  const handleChange = (row, col, value) => {
    if (/^[1-9]?$/.test(value)) {
      const updatedGrid = grid.map((r, i) =>
        i === row ? r.map((c, j) => (j === col ? value : c)) : r
      );
      setGrid(updatedGrid);
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
  };

  return (
      <div className="App">
        <h1>Sudoku Solver</h1>

        <Grid grid={grid} onCellChange={handleChange}/>
        <div className="controls">
          <button onClick={handleSolve}>Solve Sudoku</button>
          <button onClick={handleClearGrid}>Clear Grid</button>
          <button onClick={handleLoadPuzzle}>Load Puzzle</button>
        </div>
        {error && <p className="error">Error: {error}</p>}
      </div>
  );
}

export default App;
