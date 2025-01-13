import React from "react";

const Popup = ({ onClose, games, onLoadGame, onFileUpload }) => {
  const classNames = ["yellow-neon", "pink-neon", "green-neon", "red-neon", "blue-neon"];

  const getRandomClassName = () => {
    return classNames[Math.floor(Math.random() * classNames.length)];
  };

  return (
    <div className="popup">
      <div className="popup-content">
        <nav>
          <div>Load Puzzle</div>
          <button className="cancel" onClick={onClose} />
        </nav>
        <section>
          <span>Load a predefined puzzle or upload an image</span>
          <div className="file-input-wrapper">
            <label htmlFor="file-upload" className="custom-file-upload green-neon">
              Choose File
            </label>
            <input
              id="file-upload"
              className="custom-file-input"
              type="file"
              onChange={onFileUpload}
            />
          </div>
        </section>
        <footer className="games">
            {games.map((game, index) => (
                <button
                    key={game.id}
                    className={getRandomClassName()}
                    onClick={() => onLoadGame(game.board)}
                >
                  Game {index+1}
                </button>
            ))}
        </footer>
      </div>
    </div>
);
};

export default Popup;
