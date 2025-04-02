let defenderImg = new Image();
let attackerImg = new Image();
let redSquareImg = new Image();
let blueSquareImg = new Image();

function loadGameImages() {
    console.log("Loading game images...");
    
    // Create a promise for each image load
    const defenderPromise = new Promise((resolve, reject) => {
        defenderImg.onload = () => {
            console.log("Defender image loaded successfully");
            resolve();
        };
        defenderImg.onerror = () => {
            console.error("Error loading Defender image");
            reject();
        };
    });

    const attackerPromise = new Promise((resolve, reject) => {
        attackerImg.onload = () => {
            console.log("Attacker image loaded successfully");
            resolve();
        };
        attackerImg.onerror = () => {
            console.error("Error loading Attacker image");
            reject();
        };
    });

    redSquareImg.onload = function() {
        console.log("Red square image loaded successfully");
    };
    
    blueSquareImg.onload = function() {
        console.log("Blue square image loaded successfully");
    };

    redSquareImg.onerror = function() {
        console.error("Error loading red square image");
    };
    
    blueSquareImg.onerror = function() {
        console.error("Error loading blue square image");
    };

    defenderImg.src = "Defender.png";
    attackerImg.src = "Attacker.png";

    return Promise.all([defenderPromise, attackerPromise]);
}

// Call loadGameImages when window loads
window.onload = function() {
    loadGameImages().then(() => {
        console.log("All images loaded, starting game");
        newGame();
    }).catch(error => {
        console.error("Error loading images:", error);
        newGame(); 
    });
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
const autoPlayBtn = document.getElementById("autoPlayBtn");
const generatePredictionsBtn = document.getElementById("generatePredictionsBtn");

const GRID_SIZE = 10;
const CELL_SIZE = 70;
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let gameOver = false;
let actions = [];
let showPaths = false;
let autoPlayActive = false;
let autoPlayInterval = null;
const TURN_DELAY_MS = 100;
let attackerHistory = {};

let defenderShotHistory = {
  A: [
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
  ],
  B: [
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
  ],
};

function toggleAutoPlay() {
  if (autoPlayInterval) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
}

function startAutoPlay() {
  autoPlayActive = true;
  autoPlayBtn.textContent = 'Stop Auto Play';
  autoPlayBtn.style.backgroundColor = '#f44336'; 
  
  newGameBtn.disabled = true;
  nextTurnBtn.disabled = true;
  autoSelectBtn.disabled = true;
  
  autoPlayInterval = setInterval(() => {
    if (gameOver) {
      newGame(); // Start a new game automatically if current game is over
    } else {
      nextTurn();
    }
  }, TURN_DELAY_MS);
}

function stopAutoPlay() {
  autoPlayActive = false;
  autoPlayBtn.textContent = 'Auto Play';
  autoPlayBtn.style.backgroundColor = '#333';
  
  newGameBtn.disabled = false;
  nextTurnBtn.disabled = false;
  autoSelectBtn.disabled = false;
  
  if (autoPlayInterval) {
    clearInterval(autoPlayInterval);
    autoPlayInterval = null;
  }
}

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
  boardArr[1][2] = "A";
  boardArr[2][7] = "B";
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
  usedCols.sort((a, b) => a - b);

  let pathColors = ["#0ff", "#f0f", "#ff0"];
  let defenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === "string") defenders.push([r, c, board[r][c]]);
    }
  }

  for (let i = 0; i < defenders.length; i++) {
    let col = usedCols[i];
    let spawn = [GRID_SIZE - 1, col];
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
      id: String.fromCharCode(65 + i),
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget,
    });
    attackerHistory[String.fromCharCode(65 + i)] = [
      [-1, -1],
      [-1, -1],
      [-1, -1],
      [-1, -1],
    ];
  }

  for (let i = defenders.length; i < 3; i++) {
    let col = usedCols[i];
    let spawn = [GRID_SIZE - 1, col];
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
      id: String.fromCharCode(65 + i),
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget,
    });
    attackerHistory[String.fromCharCode(65 + i)] = [
      [-1, -1],
      [-1, -1],
      [-1, -1],
      [-1, -1],
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
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#444";
  ctx.textAlign = "center";
  
  for (let c = 0; c < GRID_SIZE; c++) {
    ctx.fillText(c.toString(), c * CELL_SIZE + 25 + CELL_SIZE / 2, 15);
  }
  
  ctx.textAlign = "right";
  for (let r = 0; r < GRID_SIZE; r++) {
    ctx.fillText(
      (GRID_SIZE - 1 - r).toString(),
      20,
      r * CELL_SIZE + 20 + CELL_SIZE / 2 + 5
    );
  }
  
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "#444";
      ctx.strokeRect(
        c * CELL_SIZE + 25,
        r * CELL_SIZE + 20,
        CELL_SIZE,
        CELL_SIZE
      );
      
      if (typeof boardArr[GRID_SIZE - 1 - r][c] === "string") {
        try {
          ctx.drawImage(
            defenderImg,
            c * CELL_SIZE + 30,
            r * CELL_SIZE + 25,
            CELL_SIZE - 10,
            CELL_SIZE - 10
          );
          
          ctx.fillStyle = "#fff";
          ctx.font = "bold 16px Arial";
          ctx.textAlign = "center";
          ctx.fillText(
            boardArr[GRID_SIZE - 1 - r][c],
            c * CELL_SIZE + 25 + CELL_SIZE/2,
            r * CELL_SIZE + 20 + CELL_SIZE/2 + 5
          );
        } catch (e) {
          console.error("Error drawing defender:", e);
        }
      }
    }
  }
  
  if (defenderShots["A"].length + defenderShots["B"].length > 0) {
    for (let defender in defenderShots) {
      defenderShots[defender].forEach(shot => {
        let [r, c] = shot;
        ctx.fillStyle = defender === "A" ? "rgba(255,0,0,0.3)" : "rgba(0,0,255,0.3)";
        ctx.fillRect(
          c * CELL_SIZE + 25,
          (GRID_SIZE - 1 - r) * CELL_SIZE + 20,
          CELL_SIZE,
          CELL_SIZE
        );
        
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 20px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          defender,
          c * CELL_SIZE + 25 + CELL_SIZE/2,
          (GRID_SIZE - 1 - r) * CELL_SIZE + 20 + CELL_SIZE/2 + 5
        );
      });
    }
  }
}

function drawPaths() {
  for (let atk of attackers) {
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = atk.pathColor;
    ctx.lineWidth = 2;
    ctx.shadowColor = atk.pathColor;
    ctx.shadowBlur = 10;
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
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }
  ctx.setLineDash([]);
  ctx.font = "16px Arial";
  ctx.fillStyle = "#fff";
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
    let [r, c] = atk.steppedPath[atk.currentIndex];
    try {
      ctx.drawImage(
        attackerImg,
        c * CELL_SIZE + 30,
        (GRID_SIZE - 1 - r) * CELL_SIZE + 25,
        CELL_SIZE - 10,
        CELL_SIZE - 10
      );
      
      ctx.fillStyle = "#fff";
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        atk.id,
        c * CELL_SIZE + 25 + CELL_SIZE/2,
        (GRID_SIZE - 1 - r) * CELL_SIZE + 20 + CELL_SIZE/2 + 5
      );
    } catch (e) {
      console.error("Error drawing attacker:", e);
    }
  }
}

function drawBoardAndPaths() {
  drawBoard(board);
  drawAttackers();
  if (showPaths) drawPaths();
}

function newGame() {
  actions = [];
  actionLog.innerHTML = "";
  
  gameOver = false;
  statusMessage.textContent = "";
  nextTurnBtn.disabled = false;
  newGameBtn.disabled = false;
  autoSelectBtn.disabled = false;
  showPaths = false;
  shotToggle = 0;
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  defenderShots = { A: [], B: [] };
  hoveredCell = null;
  
  defenderShotHistory = {
    A: [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    B: [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
  };

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
  
  autoSelectShots();
  updateDefenderShotHistory();
  
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoardAndPaths();

  const predictionContainer = document.getElementById('prediction-container');
  if (predictionContainer) {
    predictionContainer.style.display = 'none';
  }
}

function endGame(reason) {
  gameOver = true;
  statusMessage.textContent = reason;
  actions.push("Game ended: " + reason);
  updateActionLog();
  
  if (!autoPlayActive) {
    nextTurnBtn.disabled = true;
  }
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
  return (
    board[row][col] === 0 &&
    !attackers.some((a) => {
      let pos = a.steppedPath[a.currentIndex];
      return pos[0] === row && pos[1] === col;
    }) &&
    !Object.values(defenderShots)
      .flat()
      .some((t) => t[0] === row && t[1] === col)
  );
}

function autoSelectShots() {
  defenderShots = { A: [], B: [] };

  let livingDefenders = [];
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === "string") {
        livingDefenders.push(board[r][c]);
      }
    }
  }

  for (let defender of livingDefenders) {
    let closestAttacker = null;
    let minDistance = Infinity;

    for (let atk of attackers) {
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

    let positions = attackerHistory[closestAttacker.id] || [];

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
        const randomDir =
          directions[Math.floor(Math.random() * directions.length)];
        predictedPos[0] += randomDir[0];
        predictedPos[1] += randomDir[1];
      }

      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[1]));
      
      if (isValidShotPosition(predictedPos[0], predictedPos[1])) {
        defenderShots[defender].push([predictedPos[0], predictedPos[1]]);
        actions.push(
          `Defender ${defender} predicted shot at (${predictedPos[1]},${predictedPos[0]})`
        );
        continue;
      }
    }

    let currentPos = closestAttacker.steppedPath[closestAttacker.currentIndex];
    predictedPos = [currentPos[0], currentPos[1]];

    if (Math.random() < 1) {
      const directions = [
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
      ];
      const randomDir =
        directions[Math.floor(Math.random() * directions.length)];
      predictedPos[0] += randomDir[0];
      predictedPos[1] += randomDir[1];
      predictedPos[0] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[0]));
      predictedPos[1] = Math.max(0, Math.min(GRID_SIZE - 1, predictedPos[1]));
    }

    if (isValidShotPosition(predictedPos[0], predictedPos[1])) {
      defenderShots[defender].push([predictedPos[0], predictedPos[1]]);
      actions.push(
        `Defender ${defender} shot at (${predictedPos[1]},${predictedPos[0]})`
      );
      continue;
    }

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
        `Defender ${defender} random shot at (${randomCell[1]},${randomCell[0]})`
      );
    }
  }
}

function nextTurn() {
  if (gameOver) return;
  
  // 1. Save PRE-move state for history
  const preMoveState = {
    attackers: {},
    defenders: JSON.parse(JSON.stringify(defenderShots)),
  };

  attackers.forEach((atk) => {
    preMoveState.attackers[atk.id] = [...atk.steppedPath[atk.currentIndex]];
  });

  const movedAttackers = [];
  attackers.forEach((atk) => {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      atk.currentIndex++;
      movedAttackers.push(atk);
      actions.push(
        `Attacker ${atk.id} moved to (${atk.steppedPath[atk.currentIndex][1]},${atk.steppedPath[atk.currentIndex][0]})`
      );
    }
  });

  const remainingAttackers = [];
  const destroyedDefenders = [];

  attackers.forEach((atk) => {
    const currentPos = atk.steppedPath[atk.currentIndex];
    let wasHit = false;

    Object.entries(defenderShots).forEach(([defender, shots]) => {
      if (
        shots.some(
          (shot) => shot[0] === currentPos[0] && shot[1] === currentPos[1]
        )
      ) {
        actions.push(
          `Defender ${defender} hit Attacker ${atk.id} at (${currentPos[1]},${currentPos[0]})`
        );
        wasHit = true;
      }
    });

    if (!wasHit) {
      if (
        atk.currentIndex >=
        atk.steppedPath.length - (atk.speed === 2 ? 2 : 1)
      ) {
        const defenderPos = atk.baseTarget;
        const defender = board[defenderPos[0]][defenderPos[1]];
        if (typeof defender === "string") {
          board[defenderPos[0]][defenderPos[1]] = 0;
          destroyedDefenders.push(defenderPos);
          actions.push(`Attacker ${atk.id} destroyed Defender ${defender}`);
      }
    } else {
        remainingAttackers.push(atk);
      }
    }
  });
  console.log("before",attackerHistory)
  updateAttackerHistory();
  console.log("after",attackerHistory)

  destroyedDefenders.forEach((defenderPos) => {
    redirectAttackers(defenderPos);
  });

  attackers = remainingAttackers;
  defenderShots = { A: [], B: [] };

  autoSelectShots();

  console.log("before",defenderShotHistory)
  updateDefenderShotHistory();
  console.log("after",defenderShotHistory)

  drawBoardAndPaths();

  if (attackers.length === 0) endGame("Defenders win!");
  if (countDefenders() === 0) endGame("Attackers win!");

  updateActionLog();
}

function updateAttackerHistory() {
  // First create array of active attacker IDs
  const activeAttackerIds = attackers.map(atk => atk.id);

  // Update history for all possible attackers (A, B, C)
  ['A', 'B', 'C'].forEach(id => {
    if (!attackerHistory[id]) {
      attackerHistory[id] = [
        [-1, -1], [-1, -1], [-1, -1], [-1, -1]
      ];
    }

    // If attacker is active, update their position
    if (activeAttackerIds.includes(id)) {
      const attacker = attackers.find(atk => atk.id === id);
      const currentPos = attacker.steppedPath[attacker.currentIndex];
      
      // Only update if position changed
      if (currentPos[0] !== attackerHistory[id][0][0] || 
          currentPos[1] !== attackerHistory[id][0][1]) {
        attackerHistory[id].pop();
        attackerHistory[id].unshift([currentPos[0], currentPos[1]]);
      }
    } 
    // If attacker is not active, mark as destroyed
    else if (attackerHistory[id][0][0] !== -1 || attackerHistory[id][0][1] !== -1) {
      attackerHistory[id].pop();
      attackerHistory[id].unshift([-1, -1]);
    }
  });
}

function updateDefenderShotHistory() {
  Object.keys(defenderShots).forEach((defender) => {
    if (!defenderShotHistory[defender]) {
      defenderShotHistory[defender] = [
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
      ];
    }
    defenderShotHistory[defender].pop();
    defenderShotHistory[defender].unshift(
      defenderShots[defender][0] || [-1, -1]
    );
  });
}

function logHistoryToBoth() {
  // Helper function to validate and format coordinates
  const formatCoord = (val, max) => {
    val = parseInt(val);
    return (val >= 0 && val < max) ? val : -1;
  };

  // Ensure all attackers (A, B, C) have history
  ['A', 'B', 'C'].forEach(id => {
    if (!attackerHistory[id]) {
      attackerHistory[id] = [
        [-1, -1], [-1, -1], [-1, -1], [-1, -1]
      ];
    }
  });

  // Ensure both defenders have history
  ['A', 'B'].forEach(id => {
    if (!defenderShotHistory[id]) {
      defenderShotHistory[id] = [
        [-1, -1], [-1, -1], [-1, -1], [-1, -1]
      ];
    }
  });

  const csvRow = [
    // CURRENT POSITIONS FIRST (A_Cur, B_cur, C_cur, SA_cur, SB_Cur)
    formatCoord(attackerHistory['A'][0][1], GRID_SIZE), // A_Cur x
    formatCoord(attackerHistory['A'][0][0], GRID_SIZE), // A_Cur y
    formatCoord(attackerHistory['A'][1][1], GRID_SIZE), // AA2 x (prev1)
    formatCoord(attackerHistory['A'][1][0], GRID_SIZE), // AA2 y (prev1)
    formatCoord(attackerHistory['A'][2][1], GRID_SIZE), // AA1 x (prev2)
    formatCoord(attackerHistory['A'][2][0], GRID_SIZE), // AA1 y (prev2)
    
    // Attacker B - Current + 2 most recent positions
    formatCoord(attackerHistory['B'][0][1], GRID_SIZE), // B_Cur x
    formatCoord(attackerHistory['B'][0][0], GRID_SIZE), // B_Cur y
    formatCoord(attackerHistory['B'][1][1], GRID_SIZE), // AB2 x (prev1)
    formatCoord(attackerHistory['B'][1][0], GRID_SIZE), // AB2 y (prev1)
    formatCoord(attackerHistory['B'][2][1], GRID_SIZE), // AB1 x (prev2)
    formatCoord(attackerHistory['B'][2][0], GRID_SIZE), // AB1 y (prev2),
    
    // Attacker C - Current + 2 most recent positions
    formatCoord(attackerHistory['C'][0][1], GRID_SIZE), // C_Cur x
    formatCoord(attackerHistory['C'][0][0], GRID_SIZE), // C_Cur y
    formatCoord(attackerHistory['C'][1][1], GRID_SIZE), // AC2 x (prev1)
    formatCoord(attackerHistory['C'][1][0], GRID_SIZE), // AC2 y (prev1)
    formatCoord(attackerHistory['C'][2][1], GRID_SIZE), // AC1 x (prev2)
    formatCoord(attackerHistory['C'][2][0], GRID_SIZE), // AC1 y (prev2),
    
    // Defender A Shots - Current + 2 most recent
    formatCoord(defenderShotHistory['A'][0][1], GRID_SIZE), // SA_Cur x
    formatCoord(defenderShotHistory['A'][0][0], GRID_SIZE), // SA_Cur y
    formatCoord(defenderShotHistory['A'][1][1], GRID_SIZE), // SA2 x (prev1)
    formatCoord(defenderShotHistory['A'][1][0], GRID_SIZE), // SA2 y (prev1)
    formatCoord(defenderShotHistory['A'][2][1], GRID_SIZE), // SA1 x (prev2)
    formatCoord(defenderShotHistory['A'][2][0], GRID_SIZE), // SA1 y (prev2),
    
    // Defender B Shots - Current + 2 most recent
    formatCoord(defenderShotHistory['B'][0][1], GRID_SIZE), // SB_Cur x
    formatCoord(defenderShotHistory['B'][0][0], GRID_SIZE), // SB_Cur y
    formatCoord(defenderShotHistory['B'][1][1], GRID_SIZE), // SB2 x (prev1)
    formatCoord(defenderShotHistory['B'][1][0], GRID_SIZE), // SB2 y (prev1)
    formatCoord(defenderShotHistory['B'][2][1], GRID_SIZE), // SB1 x (prev2)
    formatCoord(defenderShotHistory['B'][2][0], GRID_SIZE)
  ];

  console.log("Logging positions to server:", csvRow);

  // Send data to servers
  fetch('http://localhost:5000/log_history', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(csvRow),
  })
  .then(response => response.json())
  .then(data => console.log("CSV logger response:", data))
  .catch((error) => {
    console.error('Error logging history to CSV:', error);
  });

  fetch('http://localhost:5001/log_data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(csvRow),
  })
  .then(response => response.json())
  .then(data => console.log("Extra processing response:", data))
  .catch((error) => {
    console.error("Error sending data to extra processing server:", error);
  });
}

function createSeparator(character) {
  const separator = document.createElement('hr');
  separator.style.border = 'none';
  separator.style.borderTop = '1px dashed #444';
  separator.style.margin = '5px 0';
  
  if (character === '~') {
    separator.style.borderTop = '1px wavy #666';
  }
  
  return separator;
}

function updateActionLog() {
  actionLog.innerHTML = '';
  
  let currentGameStart = true;
  
  for (let i = 0; i < actions.length; i++) {
    const action = actions[i];
    
    // Add game separator if this is the first action of a new game
    if (currentGameStart) {
      actionLog.appendChild(createSeparator('~'));
      currentGameStart = false;
    }
    
    // Add action item
    const li = document.createElement('li');
    li.textContent = action;
    actionLog.appendChild(li);
    
    // Add turn separator if next action is from a different turn
    if (i < actions.length - 1 && 
        actions[i+1].includes("Defender") && 
        !action.includes("Defender")) {
      actionLog.appendChild(createSeparator('-'));
    }
    
    // Detect new game
    if (action.includes("Game ended")) {
      currentGameStart = true;
    }
  }
}

let shotToggle = 0;

canvas.addEventListener("click", function(e) {
  if (gameOver || autoPlayActive) return;
  
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor((x - 25) / CELL_SIZE);
  let row = GRID_SIZE - 1 - Math.floor((y - 20) / CELL_SIZE);
  
  if (col < 0 || col >= GRID_SIZE || row < 0 || row >= GRID_SIZE) return;
  
    hoveredCell = [row, col];
  if (!isValidShotPosition(row, col)) return;
  
  let defender = (shotToggle % 2 === 0) ? "A" : "B";
  shotToggle++;
  
  defenderShots[defender] = [[row, col]];
  
  actions.push(
    "Defender " + defender + 
    " selected shot at " + 
    String.fromCharCode(65 + col) + 
    (row + 1)
  );
  
    updateActionLog();
    drawBoardAndPaths();
});
document.getElementById('makePredictionsBtn').addEventListener('click', logHistoryToBoth);
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
  if (gameOver || autoPlayActive) return;
  autoSelectShots();
  updateActionLog();
  drawBoardAndPaths();
});
autoPlayBtn.addEventListener("click", toggleAutoPlay);

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
    
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = '#444';
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
                    ctx.drawImage(
                        defenderImg,
                        c * PRED_CELL_SIZE + 27,
                        (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                        PRED_CELL_SIZE - 4,
                        PRED_CELL_SIZE - 4
                    );
                    
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
            ctx.fillStyle = defender === "A" ? "rgba(255,0,0,0.3)" : "rgba(0,0,255,0.3)";
            ctx.fillRect(
                c * PRED_CELL_SIZE + 25,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20,
                PRED_CELL_SIZE,
                PRED_CELL_SIZE
            );
            
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "bold 16px Arial";
            ctx.textAlign = "center";
            ctx.fillText(
                defender,
                c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 5
            );
        });
    });
    
    attackers.forEach(attacker => {
        let predictedIndex = Math.min(attacker.currentIndex + turnOffset, attacker.steppedPath.length - 1);
        let [r, c] = attacker.steppedPath[predictedIndex];
        
        try {
            ctx.drawImage(
                attackerImg,
                c * PRED_CELL_SIZE + 27,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                PRED_CELL_SIZE - 4,
                PRED_CELL_SIZE - 4
            );
            
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
    let maxPossibleScore = attackers.length * 100; // Perfect score if all shots are direct hits
    
    attackers.forEach(atk => {
        const nextPos = atk.steppedPath[Math.min(atk.currentIndex + 1, atk.steppedPath.length - 1)];
        
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
                // Distance penalty
                else {
                    score -= 2 * (Math.abs(shot[0] - nextPos[0]) + Math.abs(shot[1] - nextPos[1]));
                }
            });
        });
    });
    
    // Convert to score out of 10
    return Math.max(0, Math.min(10, (score / maxPossibleScore) * 10));
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

        // Generate and score predictions
        const possibleShots = generatePossibleShots();
        console.log("Generated predictions:", possibleShots);

        const scoredPredictions = possibleShots.map(shots => ({
            shots,
            score: evaluatePrediction(shots)
        })).sort((a, b) => b.score - a.score);

        // Draw best prediction
        if (scoredPredictions.length > 0) {
            drawPredictionCanvas(
                bestCanvas,
                board,
                attackers,
                getCurrentDefenders(),
                scoredPredictions[0].shots,
                0
            );

            const bestScore = document.getElementById('best-score');
            if (bestScore) {
                bestScore.textContent = `Score: ${scoredPredictions[0].score.toFixed(1)}/10`;
            }
        }

        // Draw second best prediction
        if (scoredPredictions.length > 1) {
            drawPredictionCanvas(
                secondBestCanvas,
                board,
                attackers,
                getCurrentDefenders(),
                scoredPredictions[1].shots,
                0
            );

            const secondScore = document.getElementById('second-score');
            if (secondScore) {
                secondScore.textContent = `Score: ${scoredPredictions[1].score.toFixed(1)}/10`;
            }
        }

        console.log("Prediction generation complete");

    } catch (error) {
        console.error('Error generating predictions:', error);
        statusMessage.textContent = "Error generating predictions";
    }
}


