let defenderImg = new Image();
let attackerImg = new Image();

function loadGameImages() {
    console.log("Starting to load images...");
    
    defenderImg.onload = function() {
        console.log("Defender image loaded successfully");
    };
    
    defenderImg.onerror = function() {
        console.error("Error loading Defender image");
    };
    
    attackerImg.onload = function() {
        console.log("Attacker image loaded successfully");
    };
    
    attackerImg.onerror = function() {
        console.error("Error loading Attacker image");
    };

    // Set the source paths
    defenderImg.src = "./Defender.png";
    attackerImg.src = "./Attacker.JPG";
}

// Call loadGameImages after window loads
window.onload = function() {
    loadGameImages();
    newGame();
};

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
let gameOver = false;
let actions = [];
let showPaths = false;
const autoPlayBtn = document.getElementById('autoPlayBtn');
let autoPlayActive = false;
const MIN_TURN_DELAY = 50;
let attackerHistory = {};

let defenderShotHistory = {
  A: [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], // [current, prev1, prev2, prev3]
  B: [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
};
function toggleAutoPlay() {
  autoPlayActive = !autoPlayActive;
  autoPlayBtn.textContent = autoPlayActive ? 'Stop Auto Play' : 'Auto Play';
  autoPlayBtn.style.backgroundColor = autoPlayActive ? '#f44336' : '#4CAF50';
  
  // Disable other controls during auto-play
  [newGameBtn, nextTurnBtn, autoSelectBtn].forEach(btn => {
    btn.disabled = autoPlayActive;
  });

  if (autoPlayActive) {
    autoPlayLoop();
  }
}

function autoPlayLoop() {
  if (!autoPlayActive || gameOver) {
    stopAutoPlay();
    return;
  }

  nextTurn();
  
  // Use requestAnimationFrame for smooth rendering
  if (!gameOver) {
    requestAnimationFrame(() => {
      setTimeout(autoPlayLoop, MIN_TURN_DELAY); 
    });
  }
}
let defenderShots = {
  A: [], // Current shot for Defender A
  B: [], // Current shot for Defender B
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
  boardArr[1][2] = "A"; // Defender A (left)
  boardArr[2][7] = "B"; // Defender B (right)
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
    if (
      i === 0 ||
      manhattanPath[i][0] !== manhattanPath[i - 1][0] ||
      manhattanPath[i][1] !== manhattanPath[i - 1][1]
    ) {
      unique.push(manhattanPath[i]);
    }
  }
  return unique;
}

function nearestDefender(spawn) {
  let def1 = [8, 2, "A"],
    def2 = [7, 7, "B"];
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
      if (typeof board[r][c] === "string") defenders.push([r, c, board[r][c]]);
    }
  }

  // Assign IDs A, B, C based on left-to-right spawn position
  for (let i = 0; i < defenders.length; i++) {
    let col = usedCols[i];
    let spawn = [GRID_SIZE - 1, col]; // Spawn at top row (row 9)
    let chosenTarget = defenders[i];
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
        : generateSmoothManhattanCurvePath(
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
      baseTarget: chosenTarget,
    });
    // Initialize attacker history
    attackerHistory[String.fromCharCode(65 + i)] = [
      [-1, -1], [-1, -1], [-1, -1], [-1, -1] // [current, prev1, prev2, prev3]
    ];
  }

  // For remaining attackers (if less than 3 defenders)
  for (let i = defenders.length; i < 3; i++) {
    let col = usedCols[i];
    let spawn = [GRID_SIZE - 1, col]; // Spawn at top row (row 9)
    let chosenTarget = defenders[Math.floor(Math.random() * defenders.length)];
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
        : generateSmoothManhattanCurvePath(
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
      baseTarget: chosenTarget,
    });
    // Initialize attacker history
    attackerHistory[String.fromCharCode(65 + i)] = [
      [-1, -1], [-1, -1], [-1, -1], [-1, -1] // [current, prev1, prev2, prev3]
    ];
  }
}

function countDefenders() {
  let count = 0;
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === "string") count++;
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
    ctx.fillText(c.toString(), c * CELL_SIZE + 25 + CELL_SIZE / 2, 15);
  }
  ctx.textAlign = "right";
  for (let r = 0; r < GRID_SIZE; r++) {
    ctx.fillText((GRID_SIZE - 1 - r).toString(), 20, r * CELL_SIZE + 20 + CELL_SIZE / 2 + 5);
  }
  
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "black";
      ctx.strokeRect(c * CELL_SIZE + 25, r * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
      let val = boardArr[GRID_SIZE - 1 - r][c];
      if (typeof val === "string") {
        try {
          if (defenderImg.complete && defenderImg.naturalHeight !== 0) {
            ctx.drawImage(defenderImg, c * CELL_SIZE + 30, r * CELL_SIZE + 25, 40, 40);
            console.log("Drew defender at", r, c);
          } else {
            console.warn("Defender image not ready");
            ctx.fillStyle = val === "A" ? "#ff9999" : "#9999ff";
            ctx.fillRect(c * CELL_SIZE + 30, r * CELL_SIZE + 25, 40, 40);
          }
        } catch (e) {
          console.error("Error drawing defender:", e);
          ctx.fillStyle = val === "A" ? "#ff9999" : "#9999ff";
          ctx.fillRect(c * CELL_SIZE + 30, r * CELL_SIZE + 25, 40, 40);
        }
        
        ctx.fillStyle = "white";
        ctx.font = "bold 16px Arial";
        ctx.textAlign = "center";
        ctx.fillText(val, c * CELL_SIZE + 50, r * CELL_SIZE + 45);
      }
    }
  }

  for (let defender in defenderShots) {
    for (let tile of defenderShots[defender]) {
      ctx.fillStyle = defender === "A" ? "rgba(255,0,0,0.3)" : "rgba(0,0,255,0.3)";
      ctx.fillRect(
        tile[1] * CELL_SIZE + 25,
        (GRID_SIZE - 1 - tile[0]) * CELL_SIZE + 20,
        CELL_SIZE,
        CELL_SIZE
      );
      ctx.fillStyle = "black";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        defender,
        tile[1] * CELL_SIZE + 50,
        (GRID_SIZE - 1 - tile[0]) * CELL_SIZE + 45
      );
    }
  }
  
  if (defenderShots["A"].length + defenderShots["B"].length === 0 && hoveredCell) {
    ctx.fillStyle = "rgba(0,255,0,0.3)";
    ctx.fillRect(
      hoveredCell[1] * CELL_SIZE + 25,
      (GRID_SIZE - 1 - hoveredCell[0]) * CELL_SIZE + 20,
      CELL_SIZE,
      CELL_SIZE
    );
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
      let x = pc * CELL_SIZE + 25 + CELL_SIZE / 2;
      let y = (GRID_SIZE - 1 - pr) * CELL_SIZE + 20 + CELL_SIZE / 2; 
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
      let x = pc * CELL_SIZE + 25 + CELL_SIZE / 2 - 5;
      let y = (GRID_SIZE - 1 - pr) * CELL_SIZE + 20 + CELL_SIZE / 2 + 5; 
      ctx.fillText(i.toString(), x, y);
    }
  }
}

function drawAttackers() {
  for (let atk of attackers) {
    let cr = atk.steppedPath[atk.currentIndex][0];
    let cc = atk.steppedPath[atk.currentIndex][1];
    
    try {
      if (attackerImg.complete && attackerImg.naturalHeight !== 0) {
        ctx.drawImage(
          attackerImg,
          cc * CELL_SIZE + 30,
          (GRID_SIZE - 1 - cr) * CELL_SIZE + 25,
          40,
          40
        );
        console.log("Drew attacker at", cr, cc);
      } else {
        console.warn("Attacker image not ready");
        const x = cc * CELL_SIZE + 25 + CELL_SIZE / 2;
        const y = (GRID_SIZE - 1 - cr) * CELL_SIZE + 20 + CELL_SIZE / 2;
        ctx.beginPath();
        ctx.moveTo(x, y - 20);
        ctx.lineTo(x + 20, y);
        ctx.lineTo(x, y + 20);
        ctx.lineTo(x - 20, y);
        ctx.closePath();
        ctx.fillStyle = "#ff3333";
        ctx.fill();
      }
    } catch (e) {
      console.error("Error drawing attacker:", e);
      const x = cc * CELL_SIZE + 25 + CELL_SIZE / 2;
      const y = (GRID_SIZE - 1 - cr) * CELL_SIZE + 20 + CELL_SIZE / 2;
      ctx.beginPath();
      ctx.moveTo(x, y - 20);
      ctx.lineTo(x + 20, y);
      ctx.lineTo(x, y + 20);
      ctx.lineTo(x - 20, y);
      ctx.closePath();
      ctx.fillStyle = "#ff3333";
      ctx.fill();
    }
    
    ctx.fillStyle = "white";
    ctx.font = "bold 16px Arial";
    ctx.textAlign = "center";
    ctx.fillText(atk.id, cc * CELL_SIZE + 50, (GRID_SIZE - 1 - cr) * CELL_SIZE + 45);
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
  defenderShots = { A: [], B: [] };
  hoveredCell = null;
  actions = [];
  stopAutoPlay(); 
  autoPlayBtn.disabled = false;
  
  // Initialize histories
  defenderShotHistory = {
    A: [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    B: [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
  };
  
  // Initialize attacker history with starting positions
  attackerHistory = {};
  for (let atk of attackers) { 
    let startPos = atk.steppedPath[0];
    attackerHistory[atk.id] = [
      [startPos[0], startPos[1]],
      [-1, -1],
      [-1, -1],
      [-1, -1]
    ];
  }
  
  // Auto-select initial shots
  autoSelectShots();
  updateDefenderShotHistory();
  
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoardAndPaths();

  // Hide prediction container when starting new game
  const predictionContainer = document.getElementById('prediction-container');
  if (predictionContainer) {
    predictionContainer.style.display = 'none';
  }
}

function endGame(reason) {
  gameOver = true;
  stopAutoPlay();
  nextTurnBtn.disabled = true;
  statusMessage.textContent = reason;
  actions.push("Game ended: " + reason);
  updateActionLog();
}

function redirectAttackers(destroyedDefender) {
  const remainingDefenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (
        typeof board[r][c] === "string" &&
        board[r][c] !== destroyedDefender[2]
      ) {
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
      let minDist =
        Math.abs(newTarget[0] - atk.steppedPath[atk.currentIndex][0]) +
        Math.abs(newTarget[1] - atk.steppedPath[atk.currentIndex][1]);
      for (let def of remainingDefenders.slice(1)) {
        let dist =
          Math.abs(def[0] - atk.steppedPath[atk.currentIndex][0]) +
          Math.abs(def[1] - atk.steppedPath[atk.currentIndex][1]);
        if (dist < minDist) {
          minDist = dist;
          newTarget = def;
        }
      }
      atk.baseTarget = newTarget;
      let fullPath = generateManhattanPath(
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

function isValidShotPosition(row, col) {
    // Position is valid if:
    // 1. Within grid bounds
    // 2. Not occupied by a defender
    // 3. Not occupied by another shot from the same defender
    
    // Check bounds
    if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) {
        return false;
    }
    
    // Check if position has a defender
    if (typeof board[row][col] === "string") {
        return false;
    }
    
    return true;
}

function autoSelectShots() {
  // Clear existing shots
  defenderShots = { A: [], B: [] };

  // Get all living defenders
  let livingDefenders = [];
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === "string") {
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
      // Skip if attacker isn't targeting this defender
      if (atk.baseTarget[2] !== defender) continue;

      let currentPos = atk.steppedPath[atk.currentIndex];
      let distance =
        Math.abs(currentPos[0] - atk.baseTarget[0]) +
        Math.abs(currentPos[1] - atk.baseTarget[1]);

      if (distance < minDistance) {
        minDistance = distance;
        closestAttacker = atk;
      }
    }

    // If no attackers targeting this defender, pick any attacker
    if (!closestAttacker) {
      for (let atk of attackers) {
        let currentPos = atk.steppedPath[atk.currentIndex];
        let distance =
          Math.abs(currentPos[0] - atk.baseTarget[0]) +
          Math.abs(currentPos[1] - atk.baseTarget[1]);

        if (distance < minDistance) {
          minDistance = distance;
          closestAttacker = atk;
        }
      }
    }

    if (!closestAttacker) continue;

    // Get position history of the closest attacker
    let positions = attackerHistory[closestAttacker.id] || [];

    // Predict next position based on movement pattern
    let predictedPos;
    if (positions.length >= 2 && positions[0][0] !== -1 && positions[1][0] !== -1) {
      // Calculate movement vector from last 2 positions
        let dr = positions[0][0] - positions[1][0];
        let dc = positions[0][1] - positions[1][1];
        
      // Predict next position by continuing the movement
        predictedPos = [
          positions[0][0] + dr,
          positions[0][1] + dc
        ];
        
      // 50% chance to offset the prediction by 1 in a random direction
      if (Math.random() < 0.5) {
        const directions = [
          [0, 1], [1, 0], [0, -1], [-1, 0] 
        ];
        const randomDir = directions[Math.floor(Math.random() * directions.length)];
        predictedPos[0] += randomDir[0];
        predictedPos[1] += randomDir[1];
      }
      
      // Ensure predicted position is within bounds
      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[1]));
      
      // Only use if valid position
      if (isValidShotPosition(predictedPos[0], predictedPos[1])) {
        defenderShots[defender].push([predictedPos[0], predictedPos[1]]);
        actions.push(
          `Defender ${defender} predicted shot at ${String.fromCharCode(
            65 + predictedPos[1]
          )}${predictedPos[0] + 1}`
        );
        continue;
      }
    }

    // Fallback to current position (with 50% offset chance)
    let currentPos = closestAttacker.steppedPath[closestAttacker.currentIndex];
    predictedPos = [currentPos[0], currentPos[1]];
    
    // 50% chance to offset current position
    if (Math.random() < 1) {
      const directions = [
        [0, 1], [1, 0], [0, -1], [-1, 0]
      ];
      const randomDir = directions[Math.floor(Math.random() * directions.length)];
      predictedPos[0] += randomDir[0];
      predictedPos[1] += randomDir[1];
      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[1]));
    }
    
    if (isValidShotPosition(predictedPos[0], predictedPos[1])) {
      defenderShots[defender].push([predictedPos[0], predictedPos[1]]);
      actions.push(
        `Defender ${defender} shot at ${String.fromCharCode(
          65 + predictedPos[1]
        )}${predictedPos[0] + 1}`
      );
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
      actions.push(
        `Defender ${defender} random shot at ${String.fromCharCode(
          65 + randomCell[1]
        )}${randomCell[0] + 1}`
      );
    }
  }
}

function nextTurn() {
  if (gameOver) return;
  
  // 1. Save PRE-move state for history
  const preMoveState = {
    attackers: {},
    defenders: JSON.parse(JSON.stringify(defenderShots))
  };
  
  attackers.forEach(atk => {
    preMoveState.attackers[atk.id] = [...atk.steppedPath[atk.currentIndex]];
  });

  // 2. Process attacker movement
  const movedAttackers = [];
  attackers.forEach(atk => {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      atk.currentIndex++;
      movedAttackers.push(atk);
      actions.push(`Attacker ${atk.id} moved to ${
        String.fromCharCode(65 + atk.steppedPath[atk.currentIndex][1])
      }${atk.steppedPath[atk.currentIndex][0] + 1}`);
    }
  });

  // 3. Process hits and defender destruction
  const remainingAttackers = [];
  const destroyedDefenders = [];
  
  attackers.forEach(atk => {
    const currentPos = atk.steppedPath[atk.currentIndex];
    let wasHit = false;

    // Check hits against defender shots
    Object.entries(defenderShots).forEach(([defender, shots]) => {
      if (shots.some(shot => shot[0] === currentPos[0] && shot[1] === currentPos[1])) {
        actions.push(`Defender ${defender} hit Attacker ${atk.id} at ${
          String.fromCharCode(65 + currentPos[1])
        }${currentPos[0] + 1}`);
        wasHit = true;
      }
    });

    if (!wasHit) {
      // Check if reached defender
      if (atk.currentIndex >= atk.steppedPath.length - (atk.speed === 2 ? 2 : 1)) {
        const defenderPos = atk.baseTarget;
        const defender = board[defenderPos[0]][defenderPos[1]];
        if (typeof defender === 'string') {
          board[defenderPos[0]][defenderPos[1]] = 0;
          destroyedDefenders.push(defenderPos);
          actions.push(`Attacker ${atk.id} destroyed Defender ${defender}`);
        }
      } else {
        remainingAttackers.push(atk);
      }
    }
  });

  // 4. Update attacker history AFTER all movement and hits
  updateAttackerHistory();

  // 5. Handle destroyed defenders
  destroyedDefenders.forEach(defenderPos => {
    redirectAttackers(defenderPos);
  });

  // 6. Update game state
  attackers = remainingAttackers;
  defenderShots = { A: [], B: [] }; // Clear old shots

  // 7. Select new shots for NEXT turn
  autoSelectShots();

  // 8. Update defender history with NEW shots
  updateDefenderShotHistory();

  // 9. Draw final state
  drawBoardAndPaths();

  // 10. Log FINAL positions/selections
  logHistoryToCSV();

  // Check win conditions
  if (attackers.length === 0) endGame("Defenders win!");
  if (countDefenders() === 0) endGame("Attackers win!");

  updateActionLog();
}

function updateAttackerHistory() {
  // Update living attackers
  attackers.forEach(atk => {
    const currentPos = atk.steppedPath[atk.currentIndex];
    if (!attackerHistory[atk.id]) {
      attackerHistory[atk.id] = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]];
    }
    attackerHistory[atk.id].pop();
    attackerHistory[atk.id].unshift([currentPos[0], currentPos[1]]);
  });

  // Mark dead attackers
  Object.keys(attackerHistory).forEach(atkId => {
    if (!attackers.some(a => a.id === atkId)) {
      attackerHistory[atkId].pop();
      attackerHistory[atkId].unshift([-1, -1]);
    }
  });
}

function updateDefenderShotHistory() {
  Object.keys(defenderShots).forEach(defender => {
    if (!defenderShotHistory[defender]) {
      defenderShotHistory[defender] = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]];
    }
    defenderShotHistory[defender].pop();
    defenderShotHistory[defender].unshift(
      defenderShots[defender][0] || [-1, -1]
    );
  });
}

function logHistoryToCSV() {
  // Prepare data in the exact requested format
  const csvData = {
    attackerA: attackerHistory['A'] || [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    attackerB: attackerHistory['B'] || [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    attackerC: attackerHistory['C'] || [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    defenderA: defenderShotHistory['A'] || [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    defenderB: defenderShotHistory['B'] || [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
  };

  // Convert to CSV row
  const csvRow = [
    // Attacker A history (prev1, prev2, prev3)
    csvData.attackerA[1][1], csvData.attackerA[1][0],
    csvData.attackerA[2][1], csvData.attackerA[2][0],
    csvData.attackerA[3][1], csvData.attackerA[3][0],
    // Attacker B history
    csvData.attackerB[1][1], csvData.attackerB[1][0],
    csvData.attackerB[2][1], csvData.attackerB[2][0],
    csvData.attackerB[3][1], csvData.attackerB[3][0],
    // Attacker C history
    csvData.attackerC[1][1], csvData.attackerC[1][0],
    csvData.attackerC[2][1], csvData.attackerC[2][0],
    csvData.attackerC[3][1], csvData.attackerC[3][0],
    // Defender A history
    csvData.defenderA[1][1], csvData.defenderA[1][0],
    csvData.defenderA[2][1], csvData.defenderA[2][0],
    csvData.defenderA[3][1], csvData.defenderA[3][0],
    // Defender B history
    csvData.defenderB[1][1], csvData.defenderB[1][0],
    csvData.defenderB[2][1], csvData.defenderB[2][0],
    csvData.defenderB[3][1], csvData.defenderB[3][0],
    // Current positions
    (attackerHistory['A'] && attackerHistory['A'][0][1]) || -1,
    (attackerHistory['A'] && attackerHistory['A'][0][0]) || -1,
    (attackerHistory['B'] && attackerHistory['B'][0][1]) || -1,
    (attackerHistory['B'] && attackerHistory['B'][0][0]) || -1,
    (attackerHistory['C'] && attackerHistory['C'][0][1]) || -1,
    (attackerHistory['C'] && attackerHistory['C'][0][0]) || -1,
    (defenderShotHistory['A'] && defenderShotHistory['A'][0][1]) || -1,
    (defenderShotHistory['A'] && defenderShotHistory['A'][0][0]) || -1,
    (defenderShotHistory['B'] && defenderShotHistory['B'][0][1]) || -1,
    (defenderShotHistory['B'] && defenderShotHistory['B'][0][0]) || -1
  ];

  // Send to Python backend
  fetch('http://localhost:5000/log_history', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(csvRow),
  })
  .catch((error) => {
    console.error('Error logging history:', error);
  });
}

function updateActionLog() {
  actionLog.innerHTML = actions
    .map((action) => "<li>" + action + "</li>")
    .join("");
}

canvas.addEventListener("mousemove", function (e) {
  if (gameOver) return;
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor((x - 25) / CELL_SIZE);
  let row = GRID_SIZE - 1 - Math.floor((y - 20) / CELL_SIZE); 
  if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE)
    hoveredCell = [row, col];
  else hoveredCell = null;
  drawBoardAndPaths();
});

canvas.addEventListener("click", function (e) {
  if (gameOver) return;
  
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor((x - 25) / CELL_SIZE);
  let row = GRID_SIZE - 1 - Math.floor((y - 20) / CELL_SIZE); 

  if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
    hoveredCell = [row, col];
    
    // Check if the clicked position is valid
    if (!isValidShotPosition(row, col)) {
      return;
    }

    // Determine which defender gets this shot (alternate between defenders)
    let defender;
    if (defenderShots["A"].length <= defenderShots["B"].length) {
      defender = "A";
    } else {
      defender = "B";
    }

    if (defenderShots[defender].length < 1) {
      // Each defender gets 1 shot
      defenderShots[defender].push([row, col]);
      actions.push(
        "Defender " +
        defender +
        " selected shot at " +
        String.fromCharCode(65 + col) +
        (row + 1)
      );
    } else {
      // Replace existing shot
      defenderShots[defender][0] = [row, col];
      actions.push(
        "Defender " +
        defender +
        " replaced shot with " +
        String.fromCharCode(65 + col) +
        (row + 1)
      );
    }

    updateActionLog();
    drawBoardAndPaths();
  }
});

function startAutoPlay() {
  if (gameOver) return;
  
  autoPlayBtn.textContent = 'Stop Auto Play';
  autoPlayBtn.style.backgroundColor = '#f44336'; // Red when active
  
  // Disable other buttons during auto-play
  newGameBtn.disabled = true;
  nextTurnBtn.disabled = true;
  autoSelectBtn.disabled = true;
  
  autoPlayInterval = setInterval(() => {
    nextTurn();
    if (gameOver) {
      stopAutoPlay();
    }
  }, TURN_DELAY_MS);
}

function stopAutoPlay() {
  autoPlayActive = false;
  autoPlayBtn.textContent = 'Auto Play';
  autoPlayBtn.style.backgroundColor = '#4CAF50';
  
  // Re-enable controls
  [newGameBtn, nextTurnBtn, autoSelectBtn].forEach(btn => {
    btn.disabled = gameOver;
  });
}

autoPlayBtn.addEventListener('click', function() {
  if (autoPlayInterval) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
});

newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);
actionLogBtn.addEventListener("click", function () {
  actionLog.style.display =
    actionLog.style.display === "none" ? "block" : "none";
});
togglePathsBtn.addEventListener("click", function () {
  showPaths = !showPaths;
  drawBoardAndPaths();
});
autoSelectBtn.addEventListener("click", function () {
  if (gameOver) return;
  autoSelectShots();
  updateActionLog();
  drawBoardAndPaths();
});
autoPlayBtn.addEventListener('click', toggleAutoPlay);

function getCurrentDefenders() {
    let defenders = [];
    for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
            if (typeof board[r][c] === "string") {
                defenders.push([r, c, board[r][c]]);
            }
        }
    }
    return defenders;
}

function drawPredictionCanvas(canvas, board, attackers, defenders, defenderShots, turnOffset = 0) {
    const ctx = canvas.getContext('2d');
    const PRED_CELL_SIZE = 36; 
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(25, 20, PRED_CELL_SIZE * GRID_SIZE, PRED_CELL_SIZE * GRID_SIZE);
    
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * PRED_CELL_SIZE + 25, 20);
        ctx.lineTo(i * PRED_CELL_SIZE + 25, PRED_CELL_SIZE * GRID_SIZE + 20);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(25, i * PRED_CELL_SIZE + 20);
        ctx.lineTo(PRED_CELL_SIZE * GRID_SIZE + 25, i * PRED_CELL_SIZE + 20);
        ctx.stroke();
    }
    
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i < GRID_SIZE; i++) {
        ctx.fillText(i.toString(), i * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2, 15);
        ctx.textAlign = 'right';
        ctx.fillText((GRID_SIZE - 1 - i).toString(), 20, i * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2);
    }
    
    for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
            if (typeof board[r][c] === "string") {
                try {
                    if (defenderImg.complete && defenderImg.naturalHeight !== 0) {
                        ctx.drawImage(
                            defenderImg,
                            c * PRED_CELL_SIZE + 27,
                            (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                            PRED_CELL_SIZE - 4,
                            PRED_CELL_SIZE - 4
                        );
                    } else {
                        ctx.fillStyle = board[r][c] === "A" ? "#ff9999" : "#9999ff";
                        ctx.fillRect(
                            c * PRED_CELL_SIZE + 27,
                            (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                            PRED_CELL_SIZE - 4,
                            PRED_CELL_SIZE - 4
                        );
                    }
                    
                    ctx.fillStyle = "white";
                    ctx.font = "bold 14px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText(
                        board[r][c],
                        c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                        (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 5
                    );
                } catch (e) {
                    console.error("Error drawing defender in prediction:", e);
                }
            }
        }
    }
    
    Object.entries(defenderShots).forEach(([defender, shots]) => {
        shots.forEach(shot => {
            const [r, c] = shot;
            ctx.fillStyle = defender === 'A' ? 'rgba(255,0,0,0.3)' : 'rgba(0,0,255,0.3)';
            ctx.fillRect(
                c * PRED_CELL_SIZE + 25,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20,
                PRED_CELL_SIZE,
                PRED_CELL_SIZE
            );
            
            ctx.fillStyle = 'black';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(
                defender,
                c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 4
            );
        });
    });
    
    attackers.forEach(attacker => {
        let predictedIndex = Math.min(attacker.currentIndex + turnOffset, attacker.steppedPath.length - 1);
        let [r, c] = attacker.steppedPath[predictedIndex];
        
        try {
            if (attackerImg.complete && attackerImg.naturalHeight !== 0) {
                ctx.drawImage(
                    attackerImg,
                    c * PRED_CELL_SIZE + 27,
                    (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                    PRED_CELL_SIZE - 4,
                    PRED_CELL_SIZE - 4
                );
            } else {
                const x = c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2;
                const y = (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2;
                ctx.beginPath();
                ctx.moveTo(x, y - 12);
                ctx.lineTo(x + 12, y);
                ctx.lineTo(x, y + 12);
                ctx.lineTo(x - 12, y);
                ctx.closePath();
                ctx.fillStyle = '#ff3333';
                ctx.fill();
            }
            
            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(
                attacker.id,
                c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 5
            );
        } catch (e) {
            console.error("Error drawing attacker in prediction:", e);
        }
    });
}

function generatePossibleShots() {
    const possibleShots = [];
    
    // Get current defender and attacker positions
    const defenders = getCurrentDefenders();
    const currentAttackerPositions = attackers.map(atk => ({
        id: atk.id,
        pos: atk.steppedPath[atk.currentIndex],
        nextPos: atk.steppedPath[Math.min(atk.currentIndex + 1, atk.steppedPath.length - 1)]
    }));

    // Best prediction: Try to shoot where attackers will be next turn
    const bestPrediction = { A: [], B: [] };
    defenders.forEach((defender, idx) => {
        if (currentAttackerPositions[idx]) {
            bestPrediction[defender[2]] = [currentAttackerPositions[idx].nextPos];
        }
    });
    possibleShots.push(bestPrediction);

    // Second best prediction: Shoot one square ahead of current attacker positions
    const secondBestPrediction = { A: [], B: [] };
    defenders.forEach((defender, idx) => {
        if (currentAttackerPositions[idx]) {
            const currentPos = currentAttackerPositions[idx].pos;
            const nextPos = [
                Math.min(currentPos[0] + 1, GRID_SIZE - 1),
                currentPos[1]
            ];
            secondBestPrediction[defender[2]] = [nextPos];
        }
    });
    possibleShots.push(secondBestPrediction);

    return possibleShots;
}

function evaluatePrediction(shots) {
    let score = 0;
    
    // Simple scoring for demonstration
    attackers.forEach(atk => {
        const nextPos = atk.steppedPath[Math.min(atk.currentIndex + 1, atk.steppedPath.length - 1)];
        
        // Check shots from both defenders
        Object.entries(shots).forEach(([defender, defenderShots]) => {
            defenderShots.forEach(shot => {
                // Direct hit
                if (shot[0] === nextPos[0] && shot[1] === nextPos[1]) {
                    score += 100;
                }
                // Near miss
                else if (Math.abs(shot[0] - nextPos[0]) + Math.abs(shot[1] - nextPos[1]) === 1) {
                    score += 50;
                }
            });
        });
    });
    
    return score;
}

function generatePredictions() {
    try {
        console.log("Starting prediction generation...");
        
        const predictionContainer = document.getElementById('prediction-container');
        if (predictionContainer) {
            predictionContainer.style.display = 'block';
        }
        
        const bestCanvas = document.getElementById('best-option');
        const secondBestCanvas = document.getElementById('second-best');
        
        if (!bestCanvas || !secondBestCanvas) {
            console.error('Prediction canvases not found');
            return;
        }

        // Generate predictions
        const possibleShots = generatePossibleShots();
        console.log("Generated predictions:", possibleShots);

        // Score predictions
        const scoredPredictions = possibleShots.map(shots => ({
            shots,
            score: evaluatePrediction(shots)
        }));

        if (scoredPredictions.length > 0) {
            drawPredictionCanvas(
                bestCanvas,
                board,
                attackers,
                getCurrentDefenders(),
                scoredPredictions[0].shots,
                0
            );

            // Update best score
            const bestScore = document.getElementById('best-score');
            if (bestScore) {
                bestScore.textContent = `Score: ${scoredPredictions[0].score.toFixed(2)}`;
            }
        }

        if (scoredPredictions.length > 1) {
            drawPredictionCanvas(
                secondBestCanvas,
                board,
                attackers,
                getCurrentDefenders(),
                scoredPredictions[1].shots,
                0
            );

            // Update second best score
            const secondScore = document.getElementById('second-score');
            if (secondScore) {
                secondScore.textContent = `Score: ${scoredPredictions[1].score.toFixed(2)}`;
            }
        }

        console.log("Prediction generation complete");

    } catch (error) {
        console.error('Error generating predictions:', error);
        statusMessage.textContent = "Error generating predictions";
    }
}

// Add event listener for generate predictions button
generatePredictionsBtn.addEventListener('click', generatePredictions);