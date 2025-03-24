let defenderImg, attackerImg;

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject("Failed to load " + src);
    img.src = "./" + src + "?v=" + Date.now();
  });
}
<<<<<<< HEAD

Promise.all([loadImage("Defender.png"), loadImage("Attacker.png")])
  .then(images => {
=======
Promise.all([loadImage("Defender.png"), loadImage("Attacker.JPG")])
  .then((images) => {
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
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
const newGameBtn = document.getElementById("newGameBtn");
const nextTurnBtn = document.getElementById("nextTurnBtn");
const statusMessage = document.getElementById("statusMessage");
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
<<<<<<< HEAD
let shotTile = null;

let gameStates = []; // Stores last 4 visible states
const MAX_STORED_STATES = 4;
let predictedStates = []; // Stores 4 predicted future states

=======
let shotTiles = [];
let gameOver = false;
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
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
<<<<<<< HEAD

function bresenhamLine(r0, c0, r1, c1) {
=======
function generateManhattanPath(r0, c0, r1, c1) {
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
  let path = [];
  let current = [r0, c0];
  path.push([r0, c0]);
  let step = c1 > c0 ? 1 : -1;
  while (current[1] !== c1) {
    current = [current[0], current[1] + step];
    path.push([current[0], current[1]]);
  }
  step = r1 > r0 ? 1 : -1;
  while (current[0] !== r1) {
    current = [current[0] + step, current[1]];
    path.push([current[0], current[1]]);
  }
  return path;
}
<<<<<<< HEAD

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
=======
function generateManhattanCurvePath(r0, c0, r1, c1) {
  let detour = Math.floor((r1 - r0) / 2);
  let intermediate = [r0 + detour + (Math.random() < 0.5 ? 1 : -1), c0];
  let part1 = generateManhattanPath(r0, c0, intermediate[0], intermediate[1]);
  let part2 = generateManhattanPath(intermediate[0], intermediate[1], r1, c1);
  return part1.concat(part2.slice(1));
}
function nearestDefender(spawn) {
  let def1 = [8, 2],
    def2 = [7, 7];
  let dist1 = Math.abs(def1[0] - spawn[0]) + Math.abs(def1[1] - spawn[1]);
  let dist2 = Math.abs(def2[0] - spawn[0]) + Math.abs(def2[1] - spawn[1]);
  return dist1 <= dist2 ? def1 : def2;
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
}

function placeAttackers() {
<<<<<<< HEAD
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
=======
  attackers = [];
  let usedCols = [];
  while (usedCols.length < 3) {
    let randCol = Math.floor(Math.random() * GRID_SIZE);
    if (!usedCols.includes(randCol)) usedCols.push(randCol);
  }
  let pathColors = ["orange", "green", "purple"];
  for (let i = 0; i < 3; i++) {
    let col = usedCols[i];
    let spawn = [0, col];
    let chosenTarget = nearestDefender(spawn);
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath =
      pathType === "straight"
        ? generateManhattanPath(
            spawn[0],
            spawn[1],
            chosenTarget[0],
            chosenTarget[1]
          )
        : generateManhattanCurvePath(
            spawn[0],
            spawn[1],
            chosenTarget[0],
            chosenTarget[1]
          );
    if (
      fullPath[fullPath.length - 1][0] !== chosenTarget[0] ||
      fullPath[fullPath.length - 1][1] !== chosenTarget[1]
    ) {
      fullPath.push(chosenTarget);
    }
    let steppedPath = [fullPath[0]]; // Start with spawn position
    let currentIndex = 0;

    while (currentIndex < fullPath.length - 1) {
      let stepsRemaining = fullPath.length - 1 - currentIndex;
      let nextStep = Math.min(speed, stepsRemaining);
      currentIndex += nextStep;
      steppedPath.push(fullPath[currentIndex]);
    }
    attackers.push({
      fullPath,
      steppedPath,
      speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget,
    });
  }
}
function countDefenders() {
  let count = 0;
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (board[r][c] === 1) count++;
    }
  }
  return count;
}
function drawBoard(boardArr) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "black";
      ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      let val = boardArr[r][c];
      if (val === 1) {
        if (defenderImg)
          ctx.drawImage(
            defenderImg,
            c * CELL_SIZE + 5,
            r * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10
          );
      }
    }
  }
  for (let tile of shotTiles) {
    ctx.fillStyle = "rgba(255,0,0,0.3)";
    ctx.fillRect(
      tile[1] * CELL_SIZE,
      tile[0] * CELL_SIZE,
      CELL_SIZE,
      CELL_SIZE
    );
  }
  if (!shotTiles.length && hoveredCell) {
    ctx.fillStyle = "rgba(0,255,0,0.3)";
    ctx.fillRect(
      hoveredCell[1] * CELL_SIZE,
      hoveredCell[0] * CELL_SIZE,
      CELL_SIZE,
      CELL_SIZE
    );
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
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
<<<<<<< HEAD
=======
    let cr = atk.steppedPath[atk.currentIndex][0];
    let cc = atk.steppedPath[atk.currentIndex][1];
    if (attackerImg)
      ctx.drawImage(
        attackerImg,
        cc * CELL_SIZE + 5,
        cr * CELL_SIZE + 5,
        CELL_SIZE - 10,
        CELL_SIZE - 10
      );
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
  }
}

function newGame() {
  gameOver = false;
  statusMessage.textContent = "";
  nextTurnBtn.disabled = false;
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  shotTiles = [];
  hoveredCell = null;
  for (let atk of attackers) {
    atk.currentIndex = 0;
  }
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoard(board);
  drawPaths();
}
<<<<<<< HEAD

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
=======
function endGame(reason) {
  gameOver = true;
  nextTurnBtn.disabled = true;
  statusMessage.textContent = reason;
}
function redirectAttackers(destroyedDefender) {
  const remainingDefenders = [];
  // Find remaining defenders
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (
        board[r][c] === 1 &&
        (r !== destroyedDefender[0] || c !== destroyedDefender[1])
      ) {
        remainingDefenders.push([r, c]);
      }
    }
  }

  if (remainingDefenders.length === 0) {
    drawBoard(board);
    drawPaths();
    endGame("All defenders destroyed - Attackers win!");
    return;
  }

  // Redirect attackers that were targeting the destroyed defender
  for (let atk of attackers) {
    if (
      atk.baseTarget[0] === destroyedDefender[0] &&
      atk.baseTarget[1] === destroyedDefender[1]
    ) {
      // Find new target (nearest remaining defender)
      let newTarget = remainingDefenders[0];
      let minDist =
        Math.abs(newTarget[0] - atk.steppedPath[0][0]) +
        Math.abs(newTarget[1] - atk.steppedPath[0][1]);

      for (let def of remainingDefenders.slice(1)) {
        let dist =
          Math.abs(def[0] - atk.steppedPath[0][0]) +
          Math.abs(def[1] - atk.steppedPath[0][1]);
        if (dist < minDist) {
          minDist = dist;
          newTarget = def;
        }
      }

      // Regenerate path to new target
      atk.baseTarget = newTarget;
      let pathType = Math.random() < 0.5 ? "straight" : "curve";
      let fullPath =
        pathType === "straight"
          ? generateManhattanPath(
              atk.steppedPath[atk.currentIndex][0],
              atk.steppedPath[atk.currentIndex][1],
              newTarget[0],
              newTarget[1]
            )
          : generateManhattanCurvePath(
              atk.steppedPath[atk.currentIndex][0],
              atk.steppedPath[atk.currentIndex][1],
              newTarget[0],
              newTarget[1]
            );

      if (
        fullPath[fullPath.length - 1][0] !== newTarget[0] ||
        fullPath[fullPath.length - 1][1] !== newTarget[1]
      ) {
        fullPath.push(newTarget);
      }

      let steppedPath = [];
      for (let j = 0; j < fullPath.length; j += atk.speed) {
        steppedPath.push(fullPath[j]);
      }
      if (
        steppedPath[steppedPath.length - 1][0] !== newTarget[0] ||
        steppedPath[steppedPath.length - 1][1] !== newTarget[1]
      ) {
        steppedPath[steppedPath.length - 1] = newTarget;
      }

      atk.fullPath = fullPath;
      atk.steppedPath = steppedPath;
      atk.currentIndex = 0;
    }
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
  }
}

function nextTurn() {
<<<<<<< HEAD
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
=======
  if (gameOver) return;

  // Move all attackers
  let movedAttackers = [];
  let destroyedDefenders = [];

  for (let atk of attackers) {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      atk.currentIndex++;
      movedAttackers.push(atk);
    }
  }

  // Check for shot hits
  let remainingAttackers = [];
  for (let atk of movedAttackers) {
    let currentTile = atk.steppedPath[atk.currentIndex];
    let shotHit = shotTiles.some(
      (tile) => tile[0] === currentTile[0] && tile[1] === currentTile[1]
    );

    if (!shotHit) {
      remainingAttackers.push(atk);
    }
  }

  // Check for defender collisions (can destroy multiple defenders)
  let attackersAfterCollisions = [];
  for (let atk of remainingAttackers) {
    let currentTile = atk.steppedPath[atk.currentIndex];
    if (board[currentTile[0]][currentTile[1]] === 1) {
      // Defender destroyed
      board[currentTile[0]][currentTile[1]] = 0;
      destroyedDefenders.push([currentTile[0], currentTile[1]]);
      // This attacker is also destroyed (don't add to attackersAfterCollisions)
    } else {
      attackersAfterCollisions.push(atk);
    }
  }

  attackers = attackersAfterCollisions;
  shotTiles = [];

  // Handle any defender destructions
  if (destroyedDefenders.length > 0) {
    // Check if all defenders are gone
    if (countDefenders() === 0) {
      endGame("All defenders destroyed - Attackers win!");
      drawBoard(board);
      drawPaths();
      return;
    }

    // Redirect attackers that were targeting destroyed defenders
    for (let def of destroyedDefenders) {
      redirectAttackers(def);
    }
  }
  drawBoard(board);
  drawPaths();
  // Check win conditions
  if (attackers.length === 0) {
    if (countDefenders() > 0) {
      endGame("All attackers eliminated - Defenders win!");
    } else {
      endGame("All defenders destroyed - Attackers win!");
    }
    return;
  }
}
function getDefendersAlive() {
  return countDefenders();
}
canvas.addEventListener("mousemove", function (e) {
  if (gameOver) return;
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor(x / CELL_SIZE);
  let row = Math.floor(y / CELL_SIZE);
  hoveredCell = [row, col];
  drawBoard(board);
  drawPaths();
});
<<<<<<< HEAD

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
    
=======
canvas.addEventListener("click", function () {
  if (gameOver) return;
  if (hoveredCell) {
    if (board[hoveredCell[0]][hoveredCell[1]] === 1) return;
    for (let atk of attackers) {
      let current = atk.steppedPath[atk.currentIndex];
      if (current[0] === hoveredCell[0] && current[1] === hoveredCell[1])
        return;
    }
    let defendersAlive = getDefendersAlive();
    if (shotTiles.length < defendersAlive) {
      shotTiles.push(hoveredCell);
    }
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
    drawBoard(board);
    drawPaths();
  }
});
<<<<<<< HEAD

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
=======
newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);
>>>>>>> a61baa898a7368dc0b96f1b835eae0d7cf6cee70
