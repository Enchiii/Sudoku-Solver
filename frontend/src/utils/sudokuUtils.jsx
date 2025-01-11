export const findConflicts = (board) => {
  const conflicts = [];
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      const value = board[row][col];
      if (value === "") continue;

      // Check row conflicts
      for (let i = 0; i < 9; i++) {
        if (i !== col && board[row][i] === value) {
          conflicts.push({ row, col: i });
        }
      }

      // Check column conflicts
      for (let i = 0; i < 9; i++) {
        if (i !== row && board[i][col] === value) {
          conflicts.push({ row: i, col });
        }
      }

      // Check box conflicts
      const rowStart = Math.floor(row / 3) * 3;
      const colStart = Math.floor(col / 3) * 3;
      for (let i = rowStart; i < rowStart + 3; i++) {
        for (let j = colStart; j < colStart + 3; j++) {
          if ((i !== row || j !== col) && board[i][j] === value) {
            conflicts.push({ row: i, col: j });
          }
        }
      }
    }
  }
  return conflicts;
};
