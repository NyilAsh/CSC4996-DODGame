let defenderImg, attackerImg;
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject("Failed to load " + src);
    img.src = "./" + src + "?v=" + Date.now();
  });
}
Promise.all([loadImage("Defender.png"), loadImage("Attacker.JPG")])
  .then((images) => {
    defenderImg = images[0];
    attackerImg = images[1];
    newGame();
  })
  .catch(() => {
    defenderImg = null;
    attackerImg = null;
    newGame();
  });

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const newGameBtn = document.getElementById("newGameBtn");
const nextTurnBtn = document.getElementById("nextTurnBtn");
const actionLogBtn = document.getElementById("actionLogBtn");
const togglePathsBtn = document.getElementById("togglePathsBtn");
const statusMessage = document.getElementById("statusMessage");
const actionLog = document.getElementById("actionLog");
const autoSelectBtn = document.getElementById("autoSelectBtn");
const generatePredictionsBtn = document.getElementById("generatePredictionsBtn");

const GRID_SIZE = 10;
const CELL_SIZE = 50;
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let shotTiles = [];
let gameOver = false;
let actions = [];
let attackerPositions = {}; // To track last 3 positions of each attacker
const MAX_POSITION_HISTORY = 3;
let showPaths = false; // Paths hidden by default

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
  boardArr[7][7] = 1;
}

function generateManhattanPath(r0, c0, r1, c1) {
  let path = [];
  let current = [r0, c0];
  path.push([r0, c0]);
  let stepCol = c1 > c0 ? 1 : -1;
  while (current[1] !== c1) {
    current = [current[0], current[1] + stepCol];
    path.push([current[0], current[1]]);
  }
  let stepRow = r1 > r0 ? 1 : -1;
  while (current[0] !== r1) {
    current = [current[0] + stepRow, current[1]];
    path.push([current[0], current[1]]);
  }
  return path;
}

function generateSmoothManhattanCurvePath(r0, c0, r1, c1) {
  let points = [];
  let midR = (r0 + r1) / 2 - 2;
  let midC = (c0 + c1) / 2;
  for (let t = 0; t <= 1; t += 0.05) {
    let x = (1 - t) * (1 - t) * c0 + 2 * (1 - t) * t * midC + t * t * c1;
    let y = (1 - t) * (1 - t) * r0 + 2 * (1 - t) * t * midR + t * t * r1;
    points.push([y, x]);
  }
  let smooth = [];
  for (let i = 1; i < points.length - 1; i++) {
    let avgRow = (points[i - 1][0] + points[i][0] + points[i + 1][0]) / 3;
    let avgCol = (points[i - 1][1] + points[i][1] + points[i + 1][1]) / 3;
    smooth.push([avgRow, avgCol]);
  }
  smooth.unshift(points[0]);
  smooth.push(points[points.length - 1]);
  let manhattanPath = [];
  manhattanPath.push([Math.round(smooth[0][0]), Math.round(smooth[0][1])]);
  for (let i = 1; i < smooth.length; i++) {
    let prev = manhattanPath[manhattanPath.length - 1];
    let cur = [Math.round(smooth[i][0]), Math.round(smooth[i][1])];
    let dr = cur[0] - prev[0];
    let dc = cur[1] - prev[1];
    if (dr !== 0 && dc !== 0) {
      if (Math.abs(dr) >= Math.abs(dc)) {
        cur = [prev[0] + (dr > 0 ? 1 : -1), prev[1]];
      } else {
        cur = [prev[0], prev[1] + (dc > 0 ? 1 : -1)];
      }
    }
    if (cur[0] !== prev[0] || cur[1] !== prev[1]) {
      manhattanPath.push(cur);
    }
  }
  let unique = [];
  for (let i = 0; i < manhattanPath.length; i++) {
    if (i === 0 || manhattanPath[i][0] !== manhattanPath[i - 1][0] || manhattanPath[i][1] !== manhattanPath[i - 1][1]) {
      unique.push(manhattanPath[i]);
    }
  }
  return unique;
}

function nearestDefender(spawn) {
  let def1 = [8, 2],
    def2 = [7, 7];
  let dist1 = Math.abs(def1[0] - spawn[0]) + Math.abs(def1[1] - spawn[1]);
  let dist2 = Math.abs(def2[0] - spawn[0]) + Math.abs(def2[1] - spawn[1]);
  return dist1 <= dist2 ? def1 : def2;
}

function placeAttackers() {
  attackers = [];
  let usedCols = [];
  while (usedCols.length < 3) {
    let randCol = Math.floor(Math.random() * GRID_SIZE);
    if (!usedCols.includes(randCol)) usedCols.push(randCol);
  }
  let pathColors = ["orange", "green", "purple"];
  let defenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (board[r][c] === 1) defenders.push([r, c]);
    }
  }
  for (let i = 0; i < defenders.length; i++) {
    let col = usedCols[i];
    let spawn = [0, col];
    let chosenTarget = defenders[i];
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath =
      pathType === "straight"
        ? generateManhattanPath(spawn[0], spawn[1], chosenTarget[0], chosenTarget[1])
        : generateSmoothManhattanCurvePath(spawn[0], spawn[1], chosenTarget[0], chosenTarget[1]);
    if (
      fullPath[fullPath.length - 1][0] !== chosenTarget[0] ||
      fullPath[fullPath.length - 1][1] !== chosenTarget[1]
    ) {
      fullPath.push(chosenTarget);
    }
    let steppedPath = [fullPath[0]];
    let currentIndex = 0;
    while (currentIndex < fullPath.length - 1) {
      let stepsRemaining = fullPath.length - 1 - currentIndex;
      let nextStep = Math.min(speed, stepsRemaining);
      currentIndex += nextStep;
      steppedPath.push(fullPath[currentIndex]);
    }
    attackers.push({
      id: i + 1,
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget
    });
  }
  for (let i = defenders.length; i < 3; i++) {
    let col = usedCols[i];
    let spawn = [0, col];
    let chosenTarget = defenders[Math.floor(Math.random() * defenders.length)];
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath =
      pathType === "straight"
        ? generateManhattanPath(spawn[0], spawn[1], chosenTarget[0], chosenTarget[1])
        : generateSmoothManhattanCurvePath(spawn[0], spawn[1], chosenTarget[0], chosenTarget[1]);
    if (
      fullPath[fullPath.length - 1][0] !== chosenTarget[0] ||
      fullPath[fullPath.length - 1][1] !== chosenTarget[1]
    ) {
      fullPath.push(chosenTarget);
    }
    let steppedPath = [fullPath[0]];
    let currentIndex = 0;
    while (currentIndex < fullPath.length - 1) {
      let stepsRemaining = fullPath.length - 1 - currentIndex;
      let nextStep = Math.min(speed, stepsRemaining);
      currentIndex += nextStep;
      steppedPath.push(fullPath[currentIndex]);
    }
    attackers.push({
      id: i + 1,
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget
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
  
  ctx.font = "14px Arial";
  ctx.fillStyle = "black";
  ctx.textAlign = "center";
  for (let c = 0; c < GRID_SIZE; c++) {
    ctx.fillText(String.fromCharCode(65 + c), (c * CELL_SIZE) + 25 + (CELL_SIZE / 2), 15);
  }
  ctx.textAlign = "right";
  for (let r = 0; r < GRID_SIZE; r++) {
    ctx.fillText((r + 1).toString(), 20, (r * CELL_SIZE) + 20 + (CELL_SIZE / 2) + 5);
  }
  
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "black";
      ctx.strokeRect(c * CELL_SIZE + 25, r * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
      let val = boardArr[r][c];
      if (val === 1) {
        if (defenderImg)
          ctx.drawImage(defenderImg, c * CELL_SIZE + 30, r * CELL_SIZE + 25, CELL_SIZE - 10, CELL_SIZE - 10);
      }
    }
  }
  
  for (let tile of shotTiles) {
    ctx.fillStyle = "rgba(255,0,0,0.3)";
    ctx.fillRect(tile[1] * CELL_SIZE + 25, tile[0] * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
  }
  
  if (!shotTiles.length && hoveredCell) {
    ctx.fillStyle = "rgba(0,255,0,0.3)";
    ctx.fillRect(hoveredCell[1] * CELL_SIZE + 25, hoveredCell[0] * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
  }
}

function drawPaths() {
  for (let atk of attackers) {
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = atk.pathColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < atk.fullPath.length; i++) {
      let pr = atk.fullPath[i][0];
      let pc = atk.fullPath[i][1];
      let x = (pc * CELL_SIZE) + 25 + (CELL_SIZE / 2);
      let y = (pr * CELL_SIZE) + 20 + (CELL_SIZE / 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.font = "16px Arial";
  ctx.fillStyle = "black";
  ctx.textAlign = "center";
  for (let atk of attackers) {
    for (let i = 1; i < atk.steppedPath.length; i++) {
      let pr = atk.steppedPath[i][0];
      let pc = atk.steppedPath[i][1];
      let x = (pc * CELL_SIZE) + 25 + (CELL_SIZE / 2) - 5;
      let y = (pr * CELL_SIZE) + 20 + (CELL_SIZE / 2) + 5;
      ctx.fillText(i.toString(), x, y);
    }
  }
}

function drawAttackers() {
  for (let atk of attackers) {
    let cr = atk.steppedPath[atk.currentIndex][0];
    let cc = atk.steppedPath[atk.currentIndex][1];
    if (attackerImg)
      ctx.drawImage(attackerImg, (cc * CELL_SIZE) + 30, (cr * CELL_SIZE) + 25, CELL_SIZE - 10, CELL_SIZE - 10);
  }
}

function drawBoardAndPaths() {
  drawBoard(board);
  drawAttackers();
  if (showPaths) drawPaths();
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
  actions = [];
  attackerPositions = {}; // Reset position tracking
  updateActionLog();
  for (let atk of attackers) { 
    atk.currentIndex = 0; 
    attackerPositions[atk.id] = []; // Initialize position history for each attacker
    recordAttackerPosition(atk); // Record initial position
  }
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoardAndPaths();
}

function endGame(reason) {
  gameOver = true;
  nextTurnBtn.disabled = true;
  statusMessage.textContent = reason;
  actions.push("Game ended: " + reason);
  updateActionLog();
}

function redirectAttackers(destroyedDefender) {
  const remainingDefenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (board[r][c] === 1 && (r !== destroyedDefender[0] || c !== destroyedDefender[1])) {
        remainingDefenders.push([r, c]);
      }
    }
  }
  if (remainingDefenders.length === 0) {
    drawBoardAndPaths();
    endGame("All defenders destroyed - Attackers win!");
    return;
  }
  for (let atk of attackers) {
    if (atk.baseTarget[0] === destroyedDefender[0] && atk.baseTarget[1] === destroyedDefender[1]) {
      let newTarget = remainingDefenders[0];
      let minDist = Math.abs(newTarget[0] - atk.steppedPath[atk.currentIndex][0]) + Math.abs(newTarget[1] - atk.steppedPath[atk.currentIndex][1]);
      for (let def of remainingDefenders.slice(1)) {
        let dist = Math.abs(def[0] - atk.steppedPath[atk.currentIndex][0]) + Math.abs(def[1] - atk.steppedPath[atk.currentIndex][1]);
        if (dist < minDist) { minDist = dist; newTarget = def; }
      }
      atk.baseTarget = newTarget;
      let fullPath = generateManhattanPath(
        atk.steppedPath[atk.currentIndex][0],
        atk.steppedPath[atk.currentIndex][1],
        newTarget[0],
        newTarget[1]
      );
      if (fullPath[fullPath.length - 1][0] !== newTarget[0] || fullPath[fullPath.length - 1][1] !== newTarget[1]) {
        fullPath.push(newTarget);
      }
      let steppedPath = [fullPath[0]];
      let currentIndex = 0;
      while (currentIndex < fullPath.length - 1) {
        let stepsRemaining = fullPath.length - 1 - currentIndex;
        let nextStep = Math.min(atk.speed, stepsRemaining);
        currentIndex += nextStep;
        steppedPath.push(fullPath[currentIndex]);
      }
      atk.fullPath = fullPath;
      atk.steppedPath = steppedPath;
      atk.currentIndex = 0;
    }
  }
}
function recordAttackerPosition(attacker) {
  let currentPos = attacker.steppedPath[attacker.currentIndex];
  if (!attackerPositions[attacker.id]) {
    attackerPositions[attacker.id] = [];
  }
  // Add current position to history
  attackerPositions[attacker.id].unshift([currentPos[0], currentPos[1]]);
  // Keep only the last MAX_POSITION_HISTORY positions
  if (attackerPositions[attacker.id].length > MAX_POSITION_HISTORY) {
    attackerPositions[attacker.id].pop();
  }
}
function autoSelectShots() {
  if (shotTiles.length >= countDefenders() || attackers.length === 0) return;

  // Find all attackers that have moved at least once
  let validAttackers = attackers.filter(atk => 
    attackerPositions[atk.id] && attackerPositions[atk.id].length > 0
  );

  if (validAttackers.length === 0) {
    // If no attackers have moved, pick random empty positions
    let emptyCells = [];
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        if (board[r][c] !== 1 && !attackers.some(a => {
          let pos = a.steppedPath[a.currentIndex];
          return pos[0] === r && pos[1] === c;
        })) {
          emptyCells.push([r, c]);
        }
      }
    }
    if (emptyCells.length > 0) {
      let randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
      shotTiles.push(randomCell);
      actions.push("Auto-selected random shot at " + 
        String.fromCharCode(65 + randomCell[1]) + (randomCell[0] + 1));
    }
    return;
  }

  // Sort attackers by proximity to defenders
  validAttackers.sort((a, b) => {
    let aPos = a.steppedPath[a.currentIndex];
    let bPos = b.steppedPath[b.currentIndex];
    let aDist = Infinity, bDist = Infinity;
    
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        if (board[r][c] === 1) {
          aDist = Math.min(aDist, Math.abs(aPos[0] - r) + Math.abs(aPos[1] - c));
          bDist = Math.min(bDist, Math.abs(bPos[0] - r) + Math.abs(bPos[1] - c));
        }
      }
    }
    return aDist - bDist;
  });

  // Try to predict positions for each attacker until we find a valid shot
  for (let atk of validAttackers) {
    let positions = attackerPositions[atk.id];
    let predictedPos;
    let attempts = 0;
    const maxAttempts = 3;
    
    do {
      if (positions.length >= 2) {
        // Calculate movement direction
        let dr = positions[0][0] - positions[1][0];
        let dc = positions[0][1] - positions[1][1];
        
        // Predict next position (50% chance to add small error)
        predictedPos = [
          positions[0][0] + dr,
          positions[0][1] + dc
        ];
        
        if (Math.random() < 0.9){
          predictedPos[0] += Math.floor(Math.random() * 3) - 1;
          predictedPos[1] += Math.floor(Math.random() * 3) - 1;
        }
      } else {
        // Fallback to current position with random offset
        predictedPos = atk.steppedPath[atk.currentIndex];
        predictedPos[0] += Math.floor(Math.random() * 3) - 1;
        predictedPos[1] += Math.floor(Math.random() * 3) - 1;
      }
      
      // Clamp to grid bounds
      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[1]));
      
      // Check if position is valid
      const isValid = board[predictedPos[0]][predictedPos[1]] !== 1 && 
                     !shotTiles.some(t => t[0] === predictedPos[0] && t[1] === predictedPos[1]) &&
                     !attackers.some(a => {
                       let pos = a.steppedPath[a.currentIndex];
                       return pos[0] === predictedPos[0] && pos[1] === predictedPos[1];
                     });
      
      if (isValid) {
        shotTiles.push([predictedPos[0], predictedPos[1]]);
        actions.push("Auto-selected shot targeting attacker " + atk.id + 
          " at " + String.fromCharCode(65 + predictedPos[1]) + (predictedPos[0] + 1));
        return;
      }
      
      attempts++;
    } while (attempts < maxAttempts);
  }
  
  // If all else fails, pick a random empty position
  let emptyCells = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (board[r][c] !== 1 && !attackers.some(a => {
        let pos = a.steppedPath[a.currentIndex];
        return pos[0] === r && pos[1] === c;
      })) {
        emptyCells.push([r, c]);
      }
    }
  }
  if (emptyCells.length > 0) {
    let randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
    shotTiles.push(randomCell);
    actions.push("Auto-selected random shot at " + 
      String.fromCharCode(65 + randomCell[1]) + (randomCell[0] + 1));
  }
}
function nextTurn() {
  if (gameOver) return;
  
  // Record current positions before moving
  for (let atk of attackers) {
    recordAttackerPosition(atk);
  }

  // Auto-select shots if none selected
  if (shotTiles.length === 0) {
    autoSelectShots();
  }
  actions.push("Turn advanced");
  let remainingAttackers = [];
  for (let atk of attackers) {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      let nextIndex = atk.currentIndex + 1;
      let nextTile = atk.steppedPath[nextIndex];
      let shotHit = shotTiles.some(
        (tile) => tile[0] === nextTile[0] && tile[1] === nextTile[1]
      );
      if (shotHit) {
        actions.push("Attacker " + atk.id + " was hit at " + String.fromCharCode(65 + nextTile[1]) + (nextTile[0] + 1));
        continue;
      } else {
        atk.currentIndex = nextIndex;
        if (atk.currentIndex === atk.steppedPath.length - 1 || (atk.speed === 2 && atk.currentIndex === atk.steppedPath.length - 2)) {
          board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
          actions.push("Attacker " + atk.id + " reached and destroyed defender at " + 
            String.fromCharCode(65 + atk.baseTarget[1]) + (atk.baseTarget[0] + 1) + 
            " and was destroyed in the process");
          // Attacker is destroyed when they kill a defender
          redirectAttackers(atk.baseTarget);
          continue; // Skip adding to remainingAttackers
        }
        remainingAttackers.push(atk);
      }
    } else {
      board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
      actions.push("Attacker " + atk.id + " destroyed defender at " + 
        String.fromCharCode(65 + atk.baseTarget[1]) + (atk.baseTarget[0] + 1) + 
        " and was destroyed in the process");
      redirectAttackers(atk.baseTarget);
      // Don't add to remainingAttackers since attacker is destroyed
    }
  }
  attackers = remainingAttackers;
  shotTiles = [];
  drawBoardAndPaths();
  if (attackers.length === 0) {
    if (countDefenders() > 0) endGame("All attackers eliminated - Defenders win!");
    else endGame("All defenders destroyed - Attackers win!");
  }
  updateActionLog();
}
function updateActionLog() {
  actionLog.innerHTML = actions.map(action => "<li>" + action + "</li>").join("");
}

function drawBoardAndPaths() {
  drawBoard(board);
  drawAttackers();
  if (showPaths) drawPaths();
}

function drawAttackers() {
  for (let atk of attackers) {
    let cr = atk.steppedPath[atk.currentIndex][0];
    let cc = atk.steppedPath[atk.currentIndex][1];
    if (attackerImg) ctx.drawImage(attackerImg, (cc * CELL_SIZE) + 30, (cr * CELL_SIZE) + 25, CELL_SIZE - 10, CELL_SIZE - 10);
  }
}

function newGame() {
  gameOver = false;
  statusMessage.textContent = "";
  nextTurnBtn.disabled = false;
  showPaths = false;
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  shotTiles = [];
  hoveredCell = null;
  actions = [];
  updateActionLog();
  for (let atk of attackers) { atk.currentIndex = 0; }
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoardAndPaths();
}

function endGame(reason) {
  gameOver = true;
  nextTurnBtn.disabled = true;
  statusMessage.textContent = reason;
  actions.push("Game ended: " + reason);
  updateActionLog();
}

canvas.addEventListener("mousemove", function(e) {
  if (gameOver) return;
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor((x - 25) / CELL_SIZE);
  let row = Math.floor((y - 20) / CELL_SIZE);
  if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) hoveredCell = [row, col];
  else hoveredCell = null;
  drawBoardAndPaths();
});

canvas.addEventListener("click", function(e) {
  if (gameOver) return;
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor((x - 25) / CELL_SIZE);
  let row = Math.floor((y - 20) / CELL_SIZE);
  if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
    hoveredCell = [row, col];
    if (board[hoveredCell[0]][hoveredCell[1]] === 1) return;
    for (let atk of attackers) {
      let current = atk.steppedPath[atk.currentIndex];
      if (current[0] === hoveredCell[0] && current[1] === hoveredCell[1]) return;
    }
    let defendersAlive = countDefenders();
    if (shotTiles.length < defendersAlive) {
      shotTiles.push(hoveredCell);
      actions.push("Selected shot at " + String.fromCharCode(65 + hoveredCell[0]) + (hoveredCell[1] + 1));
    } else {
      shotTiles.shift();
      shotTiles.push(hoveredCell);
      actions.push("Replaced shot with " + String.fromCharCode(65 + hoveredCell[0]) + (hoveredCell[1] + 1));
    }
    updateActionLog();
    drawBoardAndPaths();
  }
});

newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);
actionLogBtn.addEventListener("click", function() {
  actionLog.style.display = actionLog.style.display === "none" ? "block" : "none";
});
togglePathsBtn.addEventListener("click", function() {
  showPaths = !showPaths;
  drawBoardAndPaths();
});
autoSelectBtn.addEventListener("click", function() {
  if (gameOver) return;
  
  // Clear existing auto-selected shots (keep manually selected ones)
  shotTiles = shotTiles.filter(tile => {
    // Check if this tile was manually selected (not in action log as auto-selected)
    return !actions.some(action => 
      action.includes("Auto-selected shot at") && 
      action.includes(String.fromCharCode(65 + tile[1]) + (tile[0] + 1)
    ))
  });
  
  // Calculate how many shots we need to select
  const defendersAlive = countDefenders();
  const shotsToSelect = defendersAlive - shotTiles.length;
  
  if (shotsToSelect <= 0) return;
  
  // Use our existing algorithm to select the needed shots
  for (let i = 0; i < shotsToSelect; i++) {
    autoSelectShots();
  }
  
  updateActionLog();
  drawBoardAndPaths();
});
autoSelectBtn.addEventListener("click", function() {
  if (gameOver) return;
  
  const defendersAlive = countDefenders();
  
  // Clear all existing shots
  shotTiles = [];
  
  // Select shots equal to number of living defenders
  for (let i = 0; i < defendersAlive; i++) {
    autoSelectShots();
  }
  
  updateActionLog();
  drawBoardAndPaths();
});

function getCurrentDefenders() {
    let defenders = [];
    for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
            if (board[r][c] === 1) {
                defenders.push([r, c]);
            }
        }
    }
    return defenders;
}

function drawPredictionCanvas(canvas, board, attackers, defenders, shotTiles, turnOffset = 0) {
    const ctx = canvas.getContext('2d');
    const CELL_SIZE = 50;
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(25, 20, 500, 500);
    
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * CELL_SIZE + 25, 20);
        ctx.lineTo(i * CELL_SIZE + 25, 520);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(25, i * CELL_SIZE + 20);
        ctx.lineTo(525, i * CELL_SIZE + 20);
        ctx.stroke();
    }
    
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i < GRID_SIZE; i++) {
        ctx.fillText(String.fromCharCode(65 + i), i * CELL_SIZE + 50, 15);
        ctx.textAlign = 'right';
        ctx.fillText((i + 1).toString(), 20, i * CELL_SIZE + 45);
    }
    
    defenders.forEach(([r, c]) => {
        ctx.fillStyle = '#0066cc';
        ctx.strokeStyle = '#004d99';
        ctx.lineWidth = 2;
        ctx.fillRect(c * CELL_SIZE + 30, r * CELL_SIZE + 25, 40, 40);
        ctx.strokeRect(c * CELL_SIZE + 30, r * CELL_SIZE + 25, 40, 40);
    });
    
    if (shotTiles) {
        shotTiles.forEach(tile => {
            const [r, c] = tile;
            ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.fillRect(c * CELL_SIZE + 25, r * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
            ctx.strokeRect(c * CELL_SIZE + 25, r * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
        });
    }
    
    attackers.forEach(attacker => {
        let currentIndex = Math.min(
            attacker.currentIndex + turnOffset,
            attacker.steppedPath.length - 1
        );
        const [r, c] = attacker.steppedPath[currentIndex];
        
        ctx.beginPath();
        ctx.moveTo(c * CELL_SIZE + 45, r * CELL_SIZE + 25);
        ctx.lineTo(c * CELL_SIZE + 60, r * CELL_SIZE + 35);
        ctx.lineTo(c * CELL_SIZE + 45, r * CELL_SIZE + 45);
        ctx.lineTo(c * CELL_SIZE + 30, r * CELL_SIZE + 35);
        ctx.closePath();
        
        ctx.fillStyle = '#ff3333';
        ctx.fill();
        ctx.strokeStyle = '#cc0000';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

// Modify the generatePredictions function
function generatePredictions() {
    try {
        // Create canvases for each prediction state
        const states = ['current', 'next', 'plus-2', 'plus-3'];
        const turnOffsets = [0, 1, 2, 3];
        
        states.forEach((state, index) => {
            // Create a temporary canvas
            const canvas = document.createElement('canvas');
            canvas.width = 550;
            canvas.height = 550;

            drawPredictionCanvas(canvas, board, attackers, getCurrentDefenders(), shotTiles, turnOffsets[index]);
            
            // Convert to image and display
            const img = document.getElementById(`${state}-state`);
            if (img) {
                img.src = canvas.toDataURL();
            }
        });
        
        statusMessage.textContent = "Predictions generated successfully!";
    } catch (error) {
        console.error('Error generating predictions:', error);
        statusMessage.textContent = "Error generating predictions: " + error.message;
    }
}

generatePredictionsBtn.addEventListener('click', generatePredictions);