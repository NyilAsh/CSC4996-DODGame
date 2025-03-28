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

const GRID_SIZE = 10;
const CELL_SIZE = 50;
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let gameOver = false;
let actions = [];
let attackerPositions = {};
const MAX_POSITION_HISTORY = 3;
let showPaths = false;

// Track shots per defender
let defenderShots = {
  'A': [],  // Shots for Defender A
  'B': []   // Shots for Defender B
};

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
  boardArr[8][2] = 'A'; // Defender A (left)
  boardArr[7][7] = 'B'; // Defender B (right)
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
  let def1 = [8, 2, 'A'], def2 = [7, 7, 'B'];
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
  // Sort spawn columns left to right
  usedCols.sort((a, b) => a - b);
  
  let pathColors = ["orange", "green", "purple"];
  let defenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === 'string') defenders.push([r, c, board[r][c]]);
    }
  }

  // Assign IDs A, B, C based on left-to-right spawn position
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
    if (fullPath[fullPath.length - 1][0] !== chosenTarget[0] || fullPath[fullPath.length - 1][1] !== chosenTarget[1]) {
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
      id: String.fromCharCode(65 + i), // A, B, C based on sorted spawn order
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget
    });
  }

  // For remaining attackers (if less than 3 defenders)
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
    if (fullPath[fullPath.length - 1][0] !== chosenTarget[0] || fullPath[fullPath.length - 1][1] !== chosenTarget[1]) {
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
      id: String.fromCharCode(65 + i), // A, B, C based on sorted spawn order
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
      if (typeof board[r][c] === 'string') count++;
    }
  }
  return count;
}

function drawBoard(boardArr) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw grid with labels
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
  
  // Draw board pieces
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "black";
      ctx.strokeRect(c * CELL_SIZE + 25, r * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
      let val = boardArr[r][c];
      if (typeof val === 'string') {
        if (defenderImg)
          ctx.drawImage(defenderImg, c * CELL_SIZE + 30, r * CELL_SIZE + 25, CELL_SIZE - 10, CELL_SIZE - 10);
        // Draw defender label
        ctx.fillStyle = "white";
        ctx.font = "bold 16px Arial";
        ctx.textAlign = "center";
        ctx.fillText(val, c * CELL_SIZE + 25 + CELL_SIZE/2, r * CELL_SIZE + 20 + CELL_SIZE/2 + 5);
      }
    }
  }
  
  // Draw shot tiles with defender labels
  for (let defender in defenderShots) {
    for (let tile of defenderShots[defender]) {
      ctx.fillStyle = defender === 'A' ? "rgba(255,0,0,0.3)" : "rgba(0,0,255,0.3)";
      ctx.fillRect(tile[1] * CELL_SIZE + 25, tile[0] * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
      // Draw defender label on shot
      ctx.fillStyle = "black";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "center";
      ctx.fillText(defender, tile[1] * CELL_SIZE + 25 + CELL_SIZE/2, tile[0] * CELL_SIZE + 20 + CELL_SIZE/2 + 5);
    }
  }
  
  // Draw hovered cell if no shots selected
  let totalShots = defenderShots['A'].length + defenderShots['B'].length;
  if (totalShots === 0 && hoveredCell) {
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
    // Draw attacker label
    ctx.fillStyle = "white";
    ctx.font = "bold 16px Arial";
    ctx.textAlign = "center";
    ctx.fillText(atk.id, (cc * CELL_SIZE) + 25 + CELL_SIZE/2, (cr * CELL_SIZE) + 20 + CELL_SIZE/2 + 5);
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
  showPaths = false;
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  defenderShots = { 'A': [], 'B': [] }; // Initialize defender shots
  hoveredCell = null;
  actions = [];
  attackerPositions = {};
  updateActionLog();
  for (let atk of attackers) { 
    atk.currentIndex = 0; 
    attackerPositions[atk.id] = [];
    recordAttackerPosition(atk);
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
      if (typeof board[r][c] === 'string' && board[r][c] !== destroyedDefender[2]) {
        remainingDefenders.push([r, c, board[r][c]]);
      }
    }
  }
  if (remainingDefenders.length === 0) {
    drawBoardAndPaths();
    endGame("All defenders destroyed - Attackers win!");
    return;
  }
  for (let atk of attackers) {
    if (atk.baseTarget[2] === destroyedDefender[2]) {
      let newTarget = remainingDefenders[0];
      let minDist = Math.abs(newTarget[0] - atk.steppedPath[atk.currentIndex][0]) + 
                   Math.abs(newTarget[1] - atk.steppedPath[atk.currentIndex][1]);
      for (let def of remainingDefenders.slice(1)) {
        let dist = Math.abs(def[0] - atk.steppedPath[atk.currentIndex][0]) + 
                  Math.abs(def[1] - atk.steppedPath[atk.currentIndex][1]);
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
  attackerPositions[attacker.id].unshift([currentPos[0], currentPos[1]]);
  if (attackerPositions[attacker.id].length > MAX_POSITION_HISTORY) {
    attackerPositions[attacker.id].pop();
  }
}

function isValidShotPosition(row, col) {
  // Position is valid if:
  // 1. Not occupied by a defender
  // 2. Not occupied by an attacker's current position
  // 3. Not already selected as a shot
  return board[row][col] === 0 && 
         !attackers.some(a => {
           let pos = a.steppedPath[a.currentIndex];
           return pos[0] === row && pos[1] === col;
         }) &&
         !Object.values(defenderShots).flat().some(t => t[0] === row && t[1] === col);
}
function autoSelectShots() {
  // Clear existing shots
  defenderShots = { 'A': [], 'B': [] };
  
  // Get all living defenders
  let livingDefenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === 'string') {
        livingDefenders.push(board[r][c]);
      }
    }
  }

  // Assign one shot per living defender
  for (let defender of livingDefenders) {
    // Find the closest attacker to this defender
    let closestAttacker = null;
    let minDistance = Infinity;

    for (let atk of attackers) {
      if (atk.baseTarget[2] !== defender) continue;
      
      let currentPos = atk.steppedPath[atk.currentIndex];
      let distance = Math.abs(currentPos[0] - atk.baseTarget[0]) + 
                   Math.abs(currentPos[1] - atk.baseTarget[1]);
      
      if (distance < minDistance) {
        minDistance = distance;
        closestAttacker = atk;
      }
    }

    if (!closestAttacker) continue;

    // Get last positions of the closest attacker
    let positions = attackerPositions[closestAttacker.id] || [];
    let predictedPos;

    // Try to predict next position (50% chance to offset)
    if (positions.length >= 2) {
      // Calculate movement vector
      let dr = positions[0][0] - positions[1][0];
      let dc = positions[0][1] - positions[1][1];
      
      // Predict next position
      predictedPos = [
        positions[0][0] + dr,
        positions[0][1] + dc
      ];
      
      // 50% chance to offset
      if (Math.random() < 0.5) {
        const directions = [[0,1],[1,0],[0,-1],[-1,0]];
        const randomDir = directions[Math.floor(Math.random() * directions.length)];
        predictedPos[0] += randomDir[0];
        predictedPos[1] += randomDir[1];
      }
      
      // Clamp to grid
      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE-1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE-1, predictedPos[1]));
      
      if (isValidShotPosition(predictedPos[0], predictedPos[1])) {
        defenderShots[defender].push([predictedPos[0], predictedPos[1]]);
        actions.push(`Defender ${defender} auto-shot at ${String.fromCharCode(65+predictedPos[1])}${predictedPos[0]+1}`);
        continue;
      }
    }

    // Fallback to current position (with 50% offset)
    let currentPos = closestAttacker.steppedPath[closestAttacker.currentIndex];
    predictedPos = [currentPos[0], currentPos[1]];
    
    if (Math.random() < 0.5) {
      const directions = [[0,1],[1,0],[0,-1],[-1,0]];
      const randomDir = directions[Math.floor(Math.random() * directions.length)];
      predictedPos[0] += randomDir[0];
      predictedPos[1] += randomDir[1];
      
      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE-1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE-1, predictedPos[1]));
    }
    
    if (isValidShotPosition(predictedPos[0], predictedPos[1])) {
      defenderShots[defender].push([predictedPos[0], predictedPos[1]]);
      actions.push(`Defender ${defender} auto-shot at ${String.fromCharCode(65+predictedPos[1])}${predictedPos[0]+1}`);
      continue;
    }

    // Final fallback - random valid position
    let emptyCells = [];
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        if (isValidShotPosition(r, c)) {
          emptyCells.push([r, c]);
        }
      }
    }
    if (emptyCells.length > 0) {
      let randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
      defenderShots[defender].push(randomCell);
      actions.push(`Defender ${defender} random shot at ${String.fromCharCode(65+randomCell[1])}${randomCell[0]+1}`);
    }
  }
}

function nextTurn() {
  if (gameOver) return;

  // Auto-select shots at start of turn
  autoSelectShots();
  actions.push("Auto-selected shots for this turn");
  updateActionLog();
  drawBoardAndPaths();

  // Record current positions before moving
  for (let atk of attackers) {
    recordAttackerPosition(atk);
  }

  actions.push("Turn advanced");
  let remainingAttackers = [];
  let defendersDestroyedThisTurn = [];

  // First move all attackers
  for (let atk of attackers) {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      atk.currentIndex += 1;
      let newPos = atk.steppedPath[atk.currentIndex];
      actions.push(`Attacker ${atk.id} moved to ${String.fromCharCode(65 + newPos[1])}${newPos[0] + 1}`);
    }
  }

  // Then check for hits after all have moved
  for (let atk of attackers) {
    let currentPos = atk.steppedPath[atk.currentIndex];
    let wasHit = false;

    // Check if attacker was hit by any defender's shot
    for (let defender in defenderShots) {
      if (defenderShots[defender].some(
        tile => tile[0] === currentPos[0] && tile[1] === currentPos[1]
      )) {
        actions.push(`Defender ${defender}'s shot hit Attacker ${atk.id} at ${String.fromCharCode(65 + currentPos[1])}${currentPos[0] + 1}`);
        wasHit = true;
        break;
      }
    }

    if (!wasHit) {
      // Check if attacker reached defender
      if (atk.currentIndex === atk.steppedPath.length - 1 || 
          (atk.speed === 2 && atk.currentIndex === atk.steppedPath.length - 2)) {
        let defender = board[atk.baseTarget[0]][atk.baseTarget[1]];
        board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
        defendersDestroyedThisTurn.push(atk.baseTarget);
        actions.push(`Attacker ${atk.id} destroyed defender ${defender} at ${String.fromCharCode(65 + atk.baseTarget[1])}${atk.baseTarget[0] + 1}`);
      } else {
        remainingAttackers.push(atk);
      }
    }
  }

  // Redirect attackers if defenders were destroyed
  for (let destroyedDefender of defendersDestroyedThisTurn) {
    redirectAttackers(destroyedDefender);
  }

  attackers = remainingAttackers;
  
  // Clear all shots at end of turn
  defenderShots = { 'A': [], 'B': [] };
  drawBoardAndPaths();
  
  // Check win conditions
  if (attackers.length === 0) {
    endGame("All attackers eliminated - Defenders win!");
  } else if (countDefenders() === 0) {
    endGame("All defenders destroyed - Attackers win!");
  }
  
  updateActionLog();
}
function updateActionLog() {
  actionLog.innerHTML = actions.map(action => "<li>" + action + "</li>").join("");
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
    if (typeof board[hoveredCell[0]][hoveredCell[1]] === 'string') return;
    for (let atk of attackers) {
      let current = atk.steppedPath[atk.currentIndex];
      if (current[0] === hoveredCell[0] && current[1] === hoveredCell[1]) return;
    }
    
    // Determine which defender gets this shot (alternate between defenders)
    let defender;
    if (defenderShots['A'].length <= defenderShots['B'].length) {
      defender = 'A';
    } else {
      defender = 'B';
    }
    
    if (defenderShots[defender].length < 1) { // Each defender gets 1 shot
      defenderShots[defender].push(hoveredCell);
      actions.push("Defender " + defender + " selected shot at " + 
        String.fromCharCode(65 + hoveredCell[1]) + (hoveredCell[0] + 1));
    } else {
      defenderShots[defender][0] = hoveredCell; // Replace existing shot
      actions.push("Defender " + defender + " replaced shot with " + 
        String.fromCharCode(65 + hoveredCell[1]) + (hoveredCell[0] + 1));
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
  autoSelectShots();
  updateActionLog();
  drawBoardAndPaths();
});