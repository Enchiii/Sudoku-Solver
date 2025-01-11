import React, { useState, useEffect } from "react";
import Grid from "./components/Grid.jsx";
import Popup from "./components/Popup.jsx";
import { getCookie, setCookie } from "./utils/cookiesUtils.jsx";
import { findConflicts } from "./utils/sudokuUtils.jsx";
import "./App.css";

const url = "http://127.0.0.1:8000";

const App = () => {
  const [grid, setGrid] = useState(Array(9).fill(Array(9).fill("")));
  const [highlights, setHighlights] = useState([]);
  const [error, setError] = useState(null);
  const [isPopupVisible, setIsPopupVisible] = useState(false);
  const [games, setGames] = useState([]);

  useEffect(() => {
    const savedGrid = getCookie("sudokuGrid");
    if (savedGrid) {
      setError(null);
      setGrid(JSON.parse(savedGrid));
      setHighlights(findConflicts(JSON.parse(savedGrid)));
    }
  }, []);

  const fetchGames = async () => {
    try {
      const response = await fetch(url + "/games");
      const data = await response.json();
      setGames(data);
    } catch (err) {
      setError("Error fetching games: " + err.message);
    }
  };

  const handleChange = (row, col, value) => {
    if (/^[1-9]?$/.test(value)) {
      const updatedGrid = grid.map((r, i) =>
        i === row ? r.map((c, j) => (j === col ? value : c)) : r
      );
      setGrid(updatedGrid);
      setCookie("sudokuGrid", JSON.stringify(updatedGrid));
      setHighlights(findConflicts(updatedGrid));
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
      });
      const data = await response.json();
      if (data.Solved) {
        setGrid(data.Board);
      } else {
        setError("No solution found for the given Sudoku.");
      }
    } catch (err) {
      setError("Error solving Sudoku: " + err.message);
    }
  };

  const handleClearGrid = () => {
    setError(null);
    setGrid(Array(9).fill(Array(9).fill("")));
    setHighlights([]);
    document.cookie = "sudokuGrid=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
  };

  const handleLoadPuzzle = () => {
    setIsPopupVisible(true);
    fetchGames();
  };

  const loadGame = (board) => {
    setError(null);
    setGrid(board);
    setCookie("sudokuGrid", JSON.stringify(board));
    setHighlights(findConflicts(board));
    setIsPopupVisible(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(url + "/board/upload-image", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        setGrid(data.Board);
        setCookie("sudokuGrid", JSON.stringify(data.Board));
        setHighlights(findConflicts(data.Board));
      } catch (err) {
        setError("Error uploading the image: " + err.message);
      }
    }
    setIsPopupVisible(false);
  };

  return (
    <>
      <h1>Sudoku Solver</h1>
      <div className="App">
        <Grid grid={grid} onCellChange={handleChange} highlights={highlights} />
        <div className="controls">
          <button className="pink-neon" onClick={handleSolve}>
            Solve Sudoku
          </button>
          <button className="green-neon" onClick={handleClearGrid}>
            Clear Grid
          </button>
          <button className="yellow-neon" onClick={handleLoadPuzzle}>
            Load Puzzle
          </button>
        </div>
        {isPopupVisible && (
           <Popup
            onClose={() => setIsPopupVisible(false)}
            games={games}
            onLoadGame={loadGame}
            onFileUpload={handleFileUpload}
          />
        )}
        {error && <p className="error red-neon">Error: {error}</p>}
      </div>
    </>
  );
};

export default App;
