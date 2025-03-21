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
  .then(images => {
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
let shotTile = null;
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

// Function to calculate Euclidean distance between two points
function calculateDistance(r1, c1, r2, c2) {
  return Math.sqrt(Math.pow(r2 - r1, 2) + Math.pow(c2 - c1, 2));
}

function generatePathStraight(r0, c0, r1, c1) {
  let path = [];
  let currentR = r0;
  let currentC = c0;

  while (currentR !== r1 || currentC !== c1) {
    if (currentR < r1) {
      currentR++;
    } else if (currentR > r1) {
      currentR--;
    } else if (currentC < c1) {
      currentC++;
    } else if (currentC > c1) {
      currentC--;
    }
    path.push([currentR, currentC]);
  }

  return path;
}

function generatePathCurve(r0, c0, r1, c1) {
  let curvePoints = [];
  let midR = (r0 + r1) / 2 - 2;
  let midC = (c0 + c1) / 2;

  // Generate the curve points
  for (let t = 0; t <= 1.0001; t += 0.1) {
    let x = (1 - t) * (1 - t) * c0 + 2 * (1 - t) * t * midC + t * t * c1;
    let y = (1 - t) * (1 - t) * r0 + 2 * (1 - t) * t * midR + t * t * r1;
    let rx = Math.round(x);
    let ry = Math.round(y);
    curvePoints.push([ry, rx]);
  }

  // Simplify the path to only horizontal or vertical steps
  let uniquePath = [];
  let prevR = r0;
  let prevC = c0;
  for (let i = 0; i < curvePoints.length; i++) {
    let rr = curvePoints[i][0];
    let cc = curvePoints[i][1];

    if (rr !== prevR && cc !== prevC) {
      // Move horizontally first, then vertically
      if (cc !== prevC) {
        uniquePath.push([prevR, cc]);
      }
      if (rr !== prevR) {
        uniquePath.push([rr, cc]);
      }
    } else {
      uniquePath.push([rr, cc]);
    }
    prevR = rr;
    prevC = cc;
  }

  return uniquePath;
}

function placeAttackers() {
  attackers = [];
  let usedCols = [];
  while (usedCols.length < 3) {
    let randCol = Math.floor(Math.random() * GRID_SIZE);
    if (!usedCols.includes(randCol)) usedCols.push(randCol);
  }
  let pathColors = ["orange", "green", "purple"];
  let defenderPositions = [[8, 2], [7, 7]]; // Positions of defenders

  for (let i = 0; i < 3; i++) {
    let col = usedCols[i];
    let startRow = 0;
    let startCol = col;

    // Find the closest defender
    let closestDefender = defenderPositions[0];
    let minDistance = calculateDistance(startRow, startCol, closestDefender[0], closestDefender[1]);

    for (let j = 1; j < defenderPositions.length; j++) {
      let distance = calculateDistance(startRow, startCol, defenderPositions[j][0], defenderPositions[j][1]);
      if (distance < minDistance) {
        closestDefender = defenderPositions[j];
        minDistance = distance;
      }
    }

    let baseTarget = closestDefender;
    let dRow = baseTarget[0] - startRow;
    let dCol = baseTarget[1] - startCol;
    let signRow = dRow >= 0 ? 1 : -1;
    let signCol = dCol >= 0 ? 1 : -1;
    let chosenTarget = [baseTarget[0] - signRow, baseTarget[1] - signCol];

    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath = pathType === "straight" 
      ? generatePathStraight(startRow, startCol, chosenTarget[0], chosenTarget[1])
      : generatePathCurve(startRow, startCol, chosenTarget[0], chosenTarget[1]);

    let steppedPath = [];
    for (let j = 0; j < fullPath.length; j += speed) {
      steppedPath.push(fullPath[j]);
    }

    attackers.push({
      fullPath,
      steppedPath,
      speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget
    });
  }
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
  if (shotTile) {
    ctx.fillStyle = "rgba(255,0,0,0.3)";
    ctx.fillRect(shotTile[1] * CELL_SIZE, shotTile[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
  } else if (hoveredCell) {
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

function newGame() {
  gameOver = false;
  statusMessage.textContent = "";
  nextTurnBtn.disabled = false;
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  shotTile = null;
  hoveredCell = null;
  for (let atk of attackers) {
    atk.currentIndex = 0;
  }
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoard(board);
  drawPaths();
}

function endGame(reason) {
  gameOver = true;
  nextTurnBtn.disabled = true;
  statusMessage.textContent = reason;
}

function nextTurn() {
  if (gameOver) return;
  let remainingAttackers = [];
  for (let atk of attackers) {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      let nextIndex = atk.currentIndex + 1;
      let nextTile = atk.steppedPath[nextIndex];
      if (shotTile && nextTile[0] === shotTile[0] && nextTile[1] === shotTile[1]) {
        continue;
      } else {
        atk.currentIndex = nextIndex;
        if (atk.currentIndex === atk.steppedPath.length - 1) {
          let [defRow, defCol] = atk.baseTarget;
          board[defRow][defCol] = 0;
          drawBoard(board);
          drawPaths();
          endGame("A defender was destroyed!");
          return;
        }
        remainingAttackers.push(atk);
      }
    } else {
      let [defRow, defCol] = atk.baseTarget;
      board[defRow][defCol] = 0;
      drawBoard(board);
      drawPaths();
      endGame("A defender was destroyed!");
      return;
    }
  }
  attackers = remainingAttackers;
  shotTile = null;
  drawBoard(board);
  drawPaths();
}

canvas.addEventListener("mousemove", function(e) {
  if (gameOver) return;
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor(x / CELL_SIZE);
  let row = Math.floor(y / CELL_SIZE);
  hoveredCell = [row, col];
  drawBoard(board);
  drawPaths();
});

canvas.addEventListener("click", function() {
  if (gameOver) return;
  if (hoveredCell) {
    if (board[hoveredCell[0]][hoveredCell[1]] === 1) return;
    for (let atk of attackers) {
      let current = atk.steppedPath[atk.currentIndex];
      if (current[0] === hoveredCell[0] && current[1] === hoveredCell[1]) return;
    }
    shotTile = hoveredCell;
    drawBoard(board);
    drawPaths();
  }
});

newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);