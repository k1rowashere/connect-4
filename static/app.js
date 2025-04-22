const ROWS = 6;
const COLS = 7;
let currentPlayer = 1; // 1 for Player 1 (Red), 2 for Player 2 (Yellow)
let board = Array.from({length: ROWS}, () => Array(COLS).fill(0)); // 6x7 grid

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
}

function setCookie(name, value, days) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = name + '=' + value + '; expires=' + expires + '; path=/';
}


document.addEventListener('DOMContentLoaded', function () {
  const columns = document.querySelectorAll('.column');
  const player1Engine = document.getElementById('player1-engine');
  const player2Engine = document.getElementById('player2-engine');
  const player1Depth = document.getElementById('player1-depth');
  const player2Depth = document.getElementById('player2-depth');
  const player1DepthValue = document.getElementById('player1-depth-value');
  const player2DepthValue = document.getElementById('player2-depth-value');


  // load cookies json
  const playerSettings = JSON.parse(getCookie('playerSettings') || '{}');
  if (playerSettings.player1) {
    player1Engine.value = playerSettings.player1.engine;
    player1Depth.max = player1Engine.value === 'minimax_cpp' ? 13 : 5;
    player1Depth.disabled = playerSettings.player1.engine === 'human';
    player1Depth.value = playerSettings.player1.depth;
  }
  if (playerSettings.player2) {
    player2Engine.value = playerSettings.player2.engine;
    player2Depth.max = player2Engine.value === 'minimax_cpp' ? 13 : 5;
    player2Depth.disabled = playerSettings.player2.engine === 'human';
    player2Depth.value = playerSettings.player2.depth;
  }
  // Set initial depth display values
  player1DepthValue.textContent = player1Depth.value;
  player2DepthValue.textContent = player2Depth.value;

  // Update depth display values
  player1Depth.addEventListener('input', function () {
    player1DepthValue.textContent = this.value;
  });

  player2Depth.addEventListener('input', function () {
    player2DepthValue.textContent = this.value;
  });

  document.getElementById("start-game").addEventListener("click", async () => {
    resetGame();
    if (player1Engine.value !== 'human')
      await handleGameTurn();
  });

  // Handle engine changes
// Handle engine changes
  [player1Engine, player2Engine].forEach(select => {
    select.addEventListener('change', function () {
      const playerNum = this.id.includes('1') ? 1 : 2;
      const depthSlider = document.getElementById(`player${playerNum}-depth`);
      const depthValue = document.getElementById(`player${playerNum}-depth-value`);

      player1Depth.disabled = player1Engine.value === 'human';
      player2Depth.disabled = player2Engine.value === 'human';

      // Visual feedback
      depthSlider.classList.add('changed');
      setTimeout(() => depthSlider.classList.remove('changed'), 1000);

      if (this.value === 'minimax_cpp') {
        depthSlider.max = 13;
        // If current value is higher than new max, set to max
        if (parseInt(depthSlider.value) > 13) {
          depthSlider.value = 13;
          depthValue.textContent = '13';
        }
      } else {
        depthSlider.max = 5;
        // If current value is higher than new max, set to max
        if (parseInt(depthSlider.value) > 5) {
          depthSlider.value = 5;
          depthValue.textContent = '5';
        }
      }

      // Send POST request with the new settings
      sendSettingsUpdate(playerNum);
    });
  });

  // Handle depth changes
  [player1Depth, player2Depth].forEach(slider => {
    slider.addEventListener('input', function () {
      const playerNum = this.id.includes('1') ? 1 : 2;
      const engine = document.getElementById(`player${playerNum}-engine`).value;
      const maxDepth = engine === 'minimax_cpp' ? 13 : 5;

      // Ensure value doesn't exceed max
      if (parseInt(this.value) > maxDepth) {
        this.value = maxDepth;
      }
      document.getElementById(`player${playerNum}-depth-value`).textContent = this.value;
    });

    slider.addEventListener('change', function () {
      const playerNum = this.id.includes('1') ? 1 : 2;
      sendSettingsUpdate(playerNum);
    });
  });

  // Handle depth changes
  [player1Depth, player2Depth].forEach(slider => {
    slider.addEventListener('change', function () {
      const playerNum = this.id.includes('1') ? 1 : 2;
      sendSettingsUpdate(playerNum);
    });
  });

  columns.forEach(column => {
    column.addEventListener('click', async () => {
      const col = parseInt(column.dataset.column);
      await handleGameTurn(col);
    });
  });


  // Drop a piece into a column
  async function makeMove(col) {
    for (let row = ROWS - 1; row >= 0; row--) {
      if (board[row][col] === 0) {
        board[row][col] = currentPlayer;
        animatePiece(row, col);
        currentPlayer = currentPlayer === 1 ? 2 : 1; // Switch players
        updateTurnIndicator();
        break;
      }
    }

    if (isGameOver()) {
      const score = await getScore();
      let message = score === 0 ? "It's a draw!" : `Player ${score > 0 ? 1 : 2} wins!`;
      message += `\n\nScore = ${score}`;
      alert(message);
    }
  }

  async function getScore() {
    // gets score from server
    const url = `/get_score`;
    const data = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify({board: board})
    });
    if (data.ok) {
      out = await data.json();
      return out.score;
    } else {
      throw new Error('Network response was not ok');
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

  // Send settings update to server
  function sendSettingsUpdate(playerNum) {
    const engine = document.getElementById(`player${playerNum}-engine`).value;
    const depth = document.getElementById(`player${playerNum}-depth`).value;

    // set cookies to remember settings, json
    const settings = {
      engine: engine,
      depth: depth
    };
    const playerSettings = JSON.parse(getCookie('playerSettings') || '{}');
    playerSettings[`player${playerNum}`] = settings;
    setCookie('playerSettings', JSON.stringify(playerSettings), 365);
  }

  async function handleGameTurn(col = null) {
    // Get current player's engine and depth
    const engine = currentPlayer === 1 ? player1Engine.value : player2Engine.value;
    const depth = currentPlayer === 1 ? player1Depth.value : player2Depth.value;

    // Human move handling
    if (col !== null) {
      if (engine !== 'human') return;
      await processMove(col);
      return;
    }

    // AI move handling
    if (engine !== 'human') {
      await processAIMove(depth, engine);

    }
  }

  async function processMove(col) {
    try {
      await makeMove(col);  // Your existing move implementation
      if (!isGameOver()) {
        setTimeout(() => handleGameTurn(), 500);  // Next turn
      }
    } catch (error) {
      console.error('Move failed:', error);
      // Handle error (e.g., show message to user)
    }
  }

  async function processAIMove(depth, engine) {
    try {
      const response = await fetch('/ai_move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
          board: board,
          depth: depth,
          engine: engine,
          player: currentPlayer
        })
      });

      if (!response.ok) { // noinspection ExceptionCaughtLocallyJS
        throw new Error('AI move failed');
      }

      const data = await response.json();
      await processMove(data.column);

    } catch (error) {
      console.error('AI move error:', error);
    }
  }

  // Initialize the game
  updateTurnIndicator();
});

function isGameOver() {
  return board.every(row => row.every(cell => cell !== 0));
}

function resetGame() {
  board = Array.from({length: ROWS}, () => Array(COLS).fill(0)); // 6x7 grid
  currentPlayer = 1;
  document.querySelectorAll('.piece')
    .forEach(piece => piece.remove());
  updateTurnIndicator();
}

function updateTurnIndicator() {
  const indicator = document.getElementById('turn-indicator');
  if (currentPlayer === 1) {
    indicator.children[0].classList.remove('hidden');
    indicator.children[1].classList.add('hidden');
  } else {
    indicator.children[0].classList.add('hidden');
    indicator.children[1].classList.remove('hidden');
  }
}
