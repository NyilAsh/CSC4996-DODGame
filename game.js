let defenderImg, attackerImg;

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject("Failed to load " + src);
    img.src = "./" + src + "?v=" + Date.now();
  });
}

Promise.all([loadImage("Defender.png"), loadImage("Attacker.png")])
  .then(images => {
    defenderImg = images[0];
    attackerImg = images[1];
    console.log("Images loaded successfully");
    newGame();
  })
  .catch((error) => {
    console.error("Failed to load images:", error);
    defenderImg = null;
    attackerImg = null;
    newGame();
  });

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const GRID_SIZE = 10;
const CELL_SIZE = 80;
const CANVAS_SIZE = GRID_SIZE * CELL_SIZE;
canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;
const TURNS_PER_NEW_ATTACKER = 20;

let turnCount = 0;
let isDefenderTurn = true;
let targetedTiles = new Set();
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let shotTile = null;

let gameStates = []; // Stores last 4 visible states
const MAX_STORED_STATES = 4;
let predictedStates = []; // Stores 4 predicted future states

function createEmptyBoard() {
  let arr = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    arr[r] = [];
    for (let c = 0; c < GRID_SIZE; c++) {
      arr[r][c] = 0;
    }
  }
  return arr;
}

function placeDefenders(boardArr) {
  boardArr[8][2] = 1;  
  boardArr[8][7] = 1;  
  boardArr[4][5] = 1;  
}

function bresenhamLine(r0, c0, r1, c1) {
  let path = [];
  let dr = Math.abs(r1 - r0);
  let dc = Math.abs(c1 - c0);
  let sr = r0 < r1 ? 1 : -1;
  let sc = c0 < c1 ? 1 : -1;
  let err = dr - dc;
  path.push([r0, c0]);
  while (r0 !== r1 || c0 !== c1) {
    let e2 = 2 * err;
    if (e2 > -dc) { err -= dc; r0 += sr; }
    if (e2 < dr) { err += dc; c0 += sc; }
    path.push([r0, c0]);
  }
  return path;
}

function generatePathStraight(r0, c0, r1, c1) {
  return bresenhamLine(r0, c0, r1, c1);
}

function generatePathCurve(r0, c0, r1, c1) {
  let curvePoints = [];
  let midR = (r0 + r1) / 2 - 2;
  let midC = (c0 + c1) / 2;
  for (let t = 0; t <= 1.0001; t += 0.1) {
    let x = (1 - t) * (1 - t) * c0 + 2 * (1 - t) * t * midC + t * t * c1;
    let y = (1 - t) * (1 - t) * r0 + 2 * (1 - t) * t * midR + t * t * r1;
    let rx = Math.round(x);
    let ry = Math.round(y);
    curvePoints.push([ry, rx]);
  }
  let uniquePath = [];
  for (let i = 0; i < curvePoints.length; i++) {
    let rr = curvePoints[i][0];
    let cc = curvePoints[i][1];
    if (i === 0 || rr !== curvePoints[i - 1][0] || cc !== curvePoints[i - 1][1]) {
      uniquePath.push([rr, cc]);
    }
  }
  return uniquePath;
}

function placeAttackers() {
  // Number of attackers to spawn (2-3)
  const numAttackers = Math.floor(Math.random() * 2) + 2;
  
  let usedCols = new Set();
  
  for (let i = 0; i < numAttackers; i++) {
    // Find an unused column that's at least 5 tiles away from defenders
    let validSpawnPositions = [];
    for (let c = 0; c < GRID_SIZE; c++) {
      if (usedCols.has(c)) continue;
      
      let validSpawn = true;
      for (let r = 0; r < GRID_SIZE; r++) {
        for (let c2 = 0; c2 < GRID_SIZE; c2++) {
          if (board[r][c2] === 1) {
            let distance = Math.sqrt(Math.pow(r - 0, 2) + Math.pow(c2 - c, 2));
            if (distance < 5) {
              validSpawn = false;
              break;
            }
          }
        }
        if (!validSpawn) break;
      }
      if (validSpawn) {
        validSpawnPositions.push(c);
      }
    }
    
    if (validSpawnPositions.length > 0) {
      let spawnCol = validSpawnPositions[Math.floor(Math.random() * validSpawnPositions.length)];
      usedCols.add(spawnCol);
      let spawnPos = [0, spawnCol];
      
      // Find all defenders as potential targets
      let targetOptions = [];
      for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
          if (board[r][c] === 1) {
            targetOptions.push([r, c]);
          }
        }
      }
      
      if (targetOptions.length > 0) {
        let target = targetOptions[Math.floor(Math.random() * targetOptions.length)];
        let speed = Math.floor(Math.random() * 3) + 1; // Speed 1-3
        let pathType = Math.random() < 0.5 ? "straight" : "curve";
        let fullPath = pathType === "straight" ? 
          generatePathStraight(spawnPos[0], spawnPos[1], target[0], target[1]) :
          generatePathCurve(spawnPos[0], spawnPos[1], target[0], target[1]);
        
        let steppedPath = [];
        for (let j = 0; j < fullPath.length; j += speed) {
          steppedPath.push(fullPath[j]);
        }
        if (!steppedPath.includes(fullPath[fullPath.length - 1])) {
          steppedPath.push(fullPath[fullPath.length - 1]);
        }
        
        attackers.push({
          path: steppedPath,
          currentIndex: 0,
          speed,
          pathColor: `hsl(${(i * 120 + Math.random() * 60) % 360}, 70%, 50%)`,
          target: target
        });
      }
    }
  }
}

function drawBoard(boardArr) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "#333";
      ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      
      if (boardArr[r][c] === 1) {
        if (defenderImg) {
          ctx.drawImage(defenderImg, 
            c * CELL_SIZE + 10, 
            r * CELL_SIZE + 10, 
            CELL_SIZE - 20, 
            CELL_SIZE - 20
          );
        } else {
          ctx.fillStyle = "#4444ff";
          ctx.beginPath();
          ctx.moveTo(c * CELL_SIZE + CELL_SIZE/2, r * CELL_SIZE + 10);
          ctx.lineTo(c * CELL_SIZE + CELL_SIZE - 10, r * CELL_SIZE + CELL_SIZE - 10);
          ctx.lineTo(c * CELL_SIZE + 10, r * CELL_SIZE + CELL_SIZE - 10);
          ctx.closePath();
          ctx.fill();
        }
      }
      
      if (targetedTiles.has(`${r},${c}`)) {
        ctx.fillStyle = "rgba(255,0,0,0.3)";
        ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }
  }
  
  if (hoveredCell && isDefenderTurn) {
    const [row, col] = hoveredCell;
    if (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE) {
      ctx.fillStyle = "rgba(0,255,0,0.2)";
      ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }
  }
}

function drawPaths() {
  for (let atk of attackers) {
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = atk.pathColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = atk.currentIndex; i < atk.path.length; i++) {
      let [r, c] = atk.path[i];
      let x = c * CELL_SIZE + CELL_SIZE / 2;
      let y = r * CELL_SIZE + CELL_SIZE / 2;
      if (i === atk.currentIndex) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "20px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = atk.currentIndex; i < atk.path.length; i++) {
      let [r, c] = atk.path[i];
      let x = c * CELL_SIZE + CELL_SIZE / 2;
      let y = r * CELL_SIZE + CELL_SIZE / 2;
      
      let text = (i + 1).toString();
      let textWidth = ctx.measureText(text).width;
      ctx.fillStyle = "white";
      ctx.fillRect(x - textWidth/2 - 4, y - 14, textWidth + 8, 28);
      ctx.fillStyle = "black";
      ctx.fillText(text, x, y);
    }

    let [currentRow, currentCol] = atk.path[atk.currentIndex];
    if (attackerImg) {
      ctx.drawImage(attackerImg, 
        currentCol * CELL_SIZE + 10,
        currentRow * CELL_SIZE + 10,
        CELL_SIZE - 20,
        CELL_SIZE - 20
      );
    } else {
      ctx.fillStyle = "#ff4444";
      ctx.beginPath();
      ctx.moveTo(currentCol * CELL_SIZE + CELL_SIZE/2, currentRow * CELL_SIZE + 10);
      ctx.lineTo(currentCol * CELL_SIZE + CELL_SIZE - 10, currentRow * CELL_SIZE + CELL_SIZE/2);
      ctx.lineTo(currentCol * CELL_SIZE + CELL_SIZE/2, currentRow * CELL_SIZE + CELL_SIZE - 10);
      ctx.lineTo(currentCol * CELL_SIZE + 10, currentRow * CELL_SIZE + CELL_SIZE/2);
      ctx.closePath();
      ctx.fill();
    }
  }

  if (isDefenderTurn && targetedTiles.size > 0) {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    
    for (let tile of targetedTiles) {
      const [row, col] = tile.split(",").map(Number);
      const padding = 15;
      const x = col * CELL_SIZE;
      const y = row * CELL_SIZE;
      
      ctx.beginPath();
      ctx.moveTo(x + padding, y + padding);
      ctx.lineTo(x + CELL_SIZE - padding, y + CELL_SIZE - padding);
      ctx.moveTo(x + CELL_SIZE - padding, y + padding);
      ctx.lineTo(x + padding, y + CELL_SIZE - padding);
      ctx.stroke();
    }
  }
}

function newGame() {
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  shotTile = null;
  hoveredCell = null;
  for (let atk of attackers) { atk.currentIndex = 0; }
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoard(board);
  drawPaths();
}

function captureGameState() {
  const state = {
    board: JSON.parse(JSON.stringify(board)),
    attackers: attackers.map(atk => ({
      position: atk.path[atk.currentIndex],
      pathColor: atk.pathColor
    })),
    targetedTiles: Array.from(targetedTiles),
    turnCount: turnCount,
    isDefenderTurn: isDefenderTurn
  };
  
  gameStates.push(state);
  if (gameStates.length > MAX_STORED_STATES) {
    gameStates.shift();
  }
  
  return state;
}

function predictFutureStates(currentState) {
  // This is where the memory model would generate predictions
  // For now, we'll use a simple rule-based prediction
  predictedStates = [];
  
  // Example prediction logic (to be replaced by the memory model)
  for (let i = 0; i < 4; i++) {
    let prediction = JSON.parse(JSON.stringify(currentState));
    // Add simple predictions based on current attacker positions and paths
    predictedStates.push(prediction);
  }
}

function nextTurn() {
  if (isDefenderTurn) {
    // Capture state before switching turns
    const currentState = captureGameState();
    predictFutureStates(currentState);
    
    isDefenderTurn = false;
  } else {
    turnCount++;
    // Process attacker movement
    let remainingAttackers = [];
    for (let atk of attackers) {
      if (atk.currentIndex < atk.path.length - 1) {
        let nextIndex = atk.currentIndex + 1;
        let nextTile = atk.path[nextIndex];
        let tileKey = `${nextTile[0]},${nextTile[1]}`;
        
        if (targetedTiles.has(tileKey)) {
          // Attacker destroyed
          continue;
        } else {
          atk.currentIndex = nextIndex;
          // Check if attacker reached a defender
          if (board[nextTile[0]][nextTile[1]] === 1) {
            // Defender eliminated
            board[nextTile[0]][nextTile[1]] = 0;
          } else {
            remainingAttackers.push(atk);
          }
        }
      }
    }
    attackers = remainingAttackers;
    targetedTiles.clear();
    
    // Spawn new attacker every 20 turns
    if (turnCount % TURNS_PER_NEW_ATTACKER === 0) {
      placeAttackers();
    }
    
    isDefenderTurn = true;
  }
  
  drawBoard(board);
  drawPaths();
  
  // Update turn indicator
  document.getElementById("turnIndicator").textContent = 
    `Turn ${turnCount} - ${isDefenderTurn ? "Defender's" : "Attacker's"} Turn`;
}

canvas.addEventListener("mousemove", function(e) {
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor(x / CELL_SIZE);
  let row = Math.floor(y / CELL_SIZE);
  hoveredCell = [row, col];
  drawBoard(board);
  drawPaths();
});

canvas.addEventListener("click", function(e) {
  if (!isDefenderTurn) return; // Only allow targeting during defender's turn
  
  if (hoveredCell) {
    const [row, col] = hoveredCell;
    
    // Don't allow targeting defenders or current attacker positions
    if (board[row][col] === 1) return;
    
    // Check if tile is already targeted
    const tileKey = `${row},${col}`;
    if (targetedTiles.has(tileKey)) {
      targetedTiles.delete(tileKey);
    } else {
      // Check if we're not targeting an attacker's current position
      for (let atk of attackers) {
        let current = atk.path[atk.currentIndex];
        if (current[0] === row && current[1] === col) return;
      }
      
      targetedTiles.add(tileKey);
    }
    
    drawBoard(board);
    drawPaths();
  }
});

document.getElementById("instructionsBtn").addEventListener("click", function() {
  const panel = document.getElementById("instructionsPanel");
  const overlay = document.getElementById("modalOverlay");
  panel.classList.add("visible");
  overlay.classList.add("visible");
});

document.querySelector(".close-btn").addEventListener("click", function() {
  const panel = document.getElementById("instructionsPanel");
  const overlay = document.getElementById("modalOverlay");
  panel.classList.remove("visible");
  overlay.classList.remove("visible");
});

document.getElementById("modalOverlay").addEventListener("click", function() {
  const panel = document.getElementById("instructionsPanel");
  const overlay = document.getElementById("modalOverlay");
  panel.classList.remove("visible");
  overlay.classList.remove("visible");
});

document.getElementById("newGameBtn").addEventListener("click", newGame);
document.getElementById("nextTurnBtn").addEventListener("click", nextTurn);
