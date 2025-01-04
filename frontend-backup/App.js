import React, { useState } from "react";
import "./App.css";

function App() {
  const [grid, setGrid] = useState(Array(9).fill(Array(9).fill("")));
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Obsługa zmiany wartości w komórce
  const handleChange = (row, col, value) => {
    const updatedGrid = grid.map((r, i) =>
      i === row ? r.map((c, j) => (j === col ? value : c)) : r
    );
    setGrid(updatedGrid);
  };

  // Obsługa wysyłania planszy do rozwiązania
  const handleSolve = async () => {
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/board/solve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ grid }),
      });

      if (!response.ok) {
        throw new Error("Failed to solve Sudoku");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="App">
      <h1>Sudoku Solver</h1>
      <div className="sudoku-grid">
        {grid.map((row, rowIndex) => (
          <div key={rowIndex} className="row">
            {row.map((value, colIndex) => (
              <input
                key={colIndex}
                type="text"
                maxLength="1"
                value={value}
                onChange={(e) =>
                  handleChange(rowIndex, colIndex, e.target.value)
                }
                className="cell"
              />
            ))}
          </div>
        ))}
      </div>
      <button onClick={handleSolve}>Solve Sudoku</button>
      {error && <p className="error">Error: {error}</p>}
      {result && (
        <div className="result">
          <h2>{result.Solved ? "Solved Sudoku:" : "No Solution Found"}</h2>
          {result.Board.map((row, rowIndex) => (
            <div key={rowIndex} className="row">
              {row.map((value, colIndex) => (
                <div key={colIndex} className="cell">
                  {value}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
