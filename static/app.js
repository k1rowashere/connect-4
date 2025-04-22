const ROWS = 6;
const COLS = 7;
let currentPlayer = 1; // 1 for Player 1 (Red), 2 for Player 2 (Yellow)
const board = Array.from({length: ROWS}, () => Array(COLS).fill(0)); // 6x7 grid

const columns = document.querySelectorAll('.column');

// Drop a piece into a column
function dropPiece(col) {
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][col] === 0) {
      board[row][col] = currentPlayer;
      animatePiece(row, col);
      currentPlayer = currentPlayer === 1 ? 2 : 1; // Switch players
      break;
    }
  }
}


// Animate the falling piece
function animatePiece(row, col) {
  // const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
  const grid = document.getElementById('grid');
  const piece = document.createElement('div');
  piece.classList.add('piece');
  piece.classList.add(currentPlayer === 1 ? 'red' : 'yellow');
  piece.style.gridColumn = col + 1;
  piece.style.setProperty('--row', row);

  grid.appendChild(piece);
}

// Add event listeners to columns
columns.forEach(column => {
  column.addEventListener('click', () => {
    const col = parseInt(column.dataset.column);
    dropPiece(col);
  });
});





document.addEventListener('DOMContentLoaded', function() {
  const grid = document.getElementById('grid');
  const columns = document.querySelectorAll('.column');
  const player1Engine = document.getElementById('player1-engine');
  const player2Engine = document.getElementById('player2-engine');
  const player1Depth = document.getElementById('player1-depth');
  const player2Depth = document.getElementById('player2-depth');
  const player1DepthValue = document.getElementById('player1-depth-value');
  const player2DepthValue = document.getElementById('player2-depth-value');
  const currentPlayerDisplay = document.getElementById('current-player');

  let currentPlayer = 1; // 1 for Player 1 (Red), 2 for Player 2 (Yellow)
  let board = Array(6).fill().map(() => Array(7).fill(0));

  // Initialize the grid
  function initializeGrid() {
    grid.innerHTML = '';
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 7; col++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.row = row;
        cell.dataset.col = col;
        grid.appendChild(cell);
      }
    }
  }

  // Update depth display values
  player1Depth.addEventListener('input', function() {
    player1DepthValue.textContent = this.value;
  });

  player2Depth.addEventListener('input', function() {
    player2DepthValue.textContent = this.value;
  });

  // Handle engine changes
  [player1Engine, player2Engine].forEach(select => {
    select.addEventListener('change', function() {
      const playerNum = this.id.includes('1') ? 1 : 2;
      const depthSlider = document.getElementById(`player${playerNum}-depth`);

      if (this.value === 'minimax_cpp') {
        depthSlider.max = 13;
      } else {
        depthSlider.max = 5;
      }

      // Send POST request with the new settings
      sendSettingsUpdate(playerNum);
    });
  });

  // Handle depth changes
  [player1Depth, player2Depth].forEach(slider => {
    slider.addEventListener('change', function() {
      const playerNum = this.id.includes('1') ? 1 : 2;
      sendSettingsUpdate(playerNum);
    });
  });

  // Send settings update to server
  function sendSettingsUpdate(playerNum) {
    const engine = document.getElementById(`player${playerNum}-engine`).value;
    const depth = document.getElementById(`player${playerNum}-depth`).value;

    fetch('/update_settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        player: playerNum,
        engine: engine,
        depth: depth
      })
    }).then(response => response.json())
      .then(data => {
        console.log('Settings updated:', data);
      })
      .catch(error => {
        console.error('Error updating settings:', error);
      });
  }

  // Handle column clicks
  columns.forEach(column => {
    column.addEventListener('click', function() {
      const col = parseInt(this.dataset.column);
      makeMove(col);
    });
  });

  // Make a move in the specified column
  function makeMove(col) {
    for (let row = 5; row >= 0; row--) {
      if (board[row][col] === 0) {
        board[row][col] = currentPlayer;
        updateBoard();

        // Check for win or draw
        if (checkWin(row, col)) {
          setTimeout(() => {
            alert(`Player ${currentPlayer} wins!`);
            resetGame();
          }, 100);
          return;
        } else if (isBoardFull()) {
          setTimeout(() => {
            alert("It's a draw!");
            resetGame();
          }, 100);
          return;
        }

        // Switch player
        currentPlayer = currentPlayer === 1 ? 2 : 1;
        updateTurnIndicator();

        // If next player is AI, make AI move
        const currentEngine = currentPlayer === 1 ? player1Engine.value : player2Engine.value;
        if (currentEngine !== 'human') {
          setTimeout(() => makeAIMove(), 500);
        }

        return;
      }
    }
  }

  // Simulate AI move (placeholder - would call your backend)
  function makeAIMove() {
    const depth = currentPlayer === 1 ? player1Depth.value : player2Depth.value;
    const engine = currentPlayer === 1 ? player1Engine.value : player2Engine.value;

    // This would actually call your backend AI
    fetch('/ai_move', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        board: board,
        player: currentPlayer,
        engine: engine,
        depth: depth
      })
    }).then(response => response.json())
      .then(data => {
        makeMove(data.column);
      })
      .catch(error => {
        console.error('Error getting AI move:', error);
      });
  }

  // Update the visual board
  function updateBoard() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => {
      const row = parseInt(cell.dataset.row);
      const col = parseInt(cell.dataset.col);
      cell.className = 'cell';
      if (board[row][col] === 1) {
        cell.classList.add('red');
      } else if (board[row][col] === 2) {
        cell.classList.add('yellow');
      }
    });
  }

  // Check for a win
  function checkWin(row, col) {
    const directions = [
      [0, 1],  // horizontal
      [1, 0],   // vertical
      [1, 1],   // diagonal down-right
      [1, -1]   // diagonal down-left
    ];

    const player = board[row][col];

    for (const [dx, dy] of directions) {
      let count = 1;

      // Check in positive direction
      for (let i = 1; i < 4; i++) {
        const newRow = row + i * dx;
        const newCol = col + i * dy;
        if (newRow < 0 || newRow >= 6 || newCol < 0 || newCol >= 7 || board[newRow][newCol] !== player) {
          break;
        }
        count++;
      }

      // Check in negative direction
      for (let i = 1; i < 4; i++) {
        const newRow = row - i * dx;
        const newCol = col - i * dy;
        if (newRow < 0 || newRow >= 6 || newCol < 0 || newCol >= 7 || board[newRow][newCol] !== player) {
          break;
        }
        count++;
      }

      if (count >= 4) {
        return true;
      }
    }

    return false;
  }

  // Check if board is full
  function isBoardFull() {
    return board.every(row => row.every(cell => cell !== 0));
  }

  // Reset the game
  function resetGame() {
    board = Array(6).fill().map(() => Array(7).fill(0));
    currentPlayer = 1;
    updateBoard();
    updateTurnIndicator();
  }

  // Update the turn indicator
  function updateTurnIndicator() {
    if (currentPlayer === 1) {
      currentPlayerDisplay.textContent = 'Player 1 (Red)';
      currentPlayerDisplay.style.color = '#ff4444';
    } else {
      currentPlayerDisplay.textContent = 'Player 2 (Yellow)';
      currentPlayerDisplay.style.color = '#ffeb3b';
    }
  }

  // Initialize the game
  initializeGrid();
  updateTurnIndicator();
});