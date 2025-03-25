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
const statusMessage = document.getElementById("statusMessage");
const GRID_SIZE = 10;
const CELL_SIZE = 50;
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let shotTiles = [];
let gameOver = false;

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

function placeAttackers() {
  attackers = [];
  let usedCols = [];
  while (usedCols.length < 3) {
    let randCol = Math.floor(Math.random() * GRID_SIZE);
    if (!usedCols.includes(randCol)) usedCols.push(randCol);
  }
  let pathColors = ["orange", "green", "purple"];

  // Get all defender positions
  let defenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (board[r][c] === 1) defenders.push([r, c]);
    }
  }

  // First assign one attacker to each defender
  for (let i = 0; i < defenders.length; i++) {
    let col = usedCols[i];
    let spawn = [0, col];
    let chosenTarget = defenders[i];
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath = pathType === "straight"
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
      id: i + 1,
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget
    });
  }

  // Assign remaining attackers randomly
  for (let i = defenders.length; i < 3; i++) {
    let col = usedCols[i];
    let spawn = [0, col];
    let chosenTarget = defenders[Math.floor(Math.random() * defenders.length)];
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath = pathType === "straight"
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
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "black";
      ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      let val = boardArr[r][c];
      if (val === 1) {
        if (defenderImg) ctx.drawImage(defenderImg, c * CELL_SIZE + 5, r * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10);
      }
    }
  }
  for (let tile of shotTiles) {
    ctx.fillStyle = "rgba(255,0,0,0.3)";
    ctx.fillRect(tile[1] * CELL_SIZE, tile[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
  }
  if (!shotTiles.length && hoveredCell) {
    ctx.fillStyle = "rgba(0,255,0,0.3)";
    ctx.fillRect(hoveredCell[1] * CELL_SIZE, hoveredCell[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
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
      let x = pc * CELL_SIZE + CELL_SIZE / 2;
      let y = pr * CELL_SIZE + CELL_SIZE / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.font = "16px Arial";
  ctx.fillStyle = "black";
  for (let atk of attackers) {
    for (let i = 1; i < atk.steppedPath.length; i++) {
      let pr = atk.steppedPath[i][0];
      let pc = atk.steppedPath[i][1];
      let x = pc * CELL_SIZE + CELL_SIZE / 2 - 5;
      let y = pr * CELL_SIZE + CELL_SIZE / 2 + 5;
      ctx.fillText(i.toString(), x, y);
    }
    let cr = atk.steppedPath[atk.currentIndex][0];
    let cc = atk.steppedPath[atk.currentIndex][1];
    if (attackerImg) ctx.drawImage(attackerImg, cc * CELL_SIZE + 5, cr * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10);
  }
}

function drawBoardAndPaths() {
  drawBoard(board);
  drawPaths();
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
  for (let atk of attackers) { atk.currentIndex = 0; }
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoardAndPaths();
}

function endGame(reason) {
  gameOver = true;
  nextTurnBtn.disabled = true;
  statusMessage.textContent = reason;
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

function nextTurn() {
  if (gameOver) return;
  
  let remainingAttackers = [];
  for (let atk of attackers) {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      let nextIndex = atk.currentIndex + 1;
      let nextTile = atk.steppedPath[nextIndex];
      let shotHit = shotTiles.some(
        (tile) => tile[0] === nextTile[0] && tile[1] === nextTile[1]
      );
      if (shotHit) {
        continue;
      } else {
        atk.currentIndex = nextIndex;
        if (atk.currentIndex === atk.steppedPath.length - 1) {
          board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
          redirectAttackers(atk.baseTarget);
        }
        remainingAttackers.push(atk);
      }
    } else {
      board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
      redirectAttackers(atk.baseTarget);
    }
  }
  
  attackers = remainingAttackers;
  shotTiles = [];
  drawBoardAndPaths();
  
  if (attackers.length === 0) {
    endGame("All attackers eliminated - Defenders win!");
  } else if (countDefenders() === 0) {
    endGame("All defenders destroyed - Attackers win!");
  }
}

canvas.addEventListener("mousemove", function(e) {
  if (gameOver) return;
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor(x / CELL_SIZE);
  let row = Math.floor(y / CELL_SIZE);
  hoveredCell = [row, col];
  drawBoardAndPaths();
});

canvas.addEventListener("click", function() {
  if (gameOver) return;
  if (hoveredCell) {
    // Prevent shooting defenders or attackers
    if (board[hoveredCell[0]][hoveredCell[1]] === 1) return;
    for (let atk of attackers) {
      let current = atk.steppedPath[atk.currentIndex];
      if (current[0] === hoveredCell[0] && current[1] === hoveredCell[1]) return;
    }
    
    // Check if this cell is already selected
    let alreadySelected = shotTiles.some(tile => 
      tile[0] === hoveredCell[0] && tile[1] === hoveredCell[1]
    );
    
    if (alreadySelected) {
      // Remove the shot if clicking on an already selected cell
      shotTiles = shotTiles.filter(tile => 
        !(tile[0] === hoveredCell[0] && tile[1] === hoveredCell[1])
      );
    } else {
      let defendersAlive = countDefenders();
      if (shotTiles.length < defendersAlive) {
        // Add new shot if under limit
        shotTiles.push(hoveredCell);
      } else {
        // Replace oldest shot if at limit (rolling selection)
        shotTiles.shift(); // Remove oldest shot
        shotTiles.push(hoveredCell); // Add new shot
      }
    }
    drawBoardAndPaths();
  }
});

newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);