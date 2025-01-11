// import React, { useState, useEffect } from "react";

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

export default Grid;
