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

const GRID_SIZE = 10;
const CELL_SIZE = 50;
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let shotTiles = [];
let gameOver = false;
let actions = [];
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
      if (val === 1) {
        if (defenderImg)
          ctx.drawImage(defenderImg, c * CELL_SIZE + 30, r * CELL_SIZE + 25, CELL_SIZE - 10, CELL_SIZE - 10);
      }
    }
  }
  
  // Draw shot tiles
  for (let tile of shotTiles) {
    ctx.fillStyle = "rgba(255,0,0,0.3)";
    ctx.fillRect(tile[1] * CELL_SIZE + 25, tile[0] * CELL_SIZE + 20, CELL_SIZE, CELL_SIZE);
  }
  
  // Draw hovered cell if no shots selected
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
  actions.push("Turn advanced");
  let remainingAttackers = [];
  for (let atk of attackers) {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      let nextIndex = atk.currentIndex + 1;
      let nextTile = atk.steppedPath[nextIndex];
      let shotHit = shotTiles.some(tile => tile[0] === nextTile[0] && tile[1] === nextTile[1]);
      if (shotHit) { actions.push("Attacker " + atk.id + " hit at " + String.fromCharCode(65 + nextTile[0]) + (nextTile[1] + 1)); continue; }
      else {
        atk.currentIndex = nextIndex;
        if (atk.currentIndex === atk.steppedPath.length - 1 || (atk.speed === 2 && atk.currentIndex === atk.steppedPath.length - 2)) {
          board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
          actions.push("Attacker " + atk.id + " reached defender at " + String.fromCharCode(65 + atk.baseTarget[0]) + (atk.baseTarget[1] + 1));
          drawBoardAndPaths();
          setTimeout(() => { endGame("A defender was destroyed!"); }, 500);
          updateActionLog();
          return;
        }
        remainingAttackers.push(atk);
      }
    } else {
      board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
      actions.push("Attacker " + atk.id + " destroyed defender at " + String.fromCharCode(65 + atk.baseTarget[0]) + (atk.baseTarget[1] + 1));
      drawBoardAndPaths();
      setTimeout(() => { endGame("A defender was destroyed!"); }, 500);
      updateActionLog();
      return;
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
