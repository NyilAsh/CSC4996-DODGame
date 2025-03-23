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
function generateManhattanCurvePath(r0, c0, r1, c1) {
  let detour = Math.floor((r1 - r0) / 2);
  let intermediate = [r0 + detour + (Math.random() < 0.5 ? 1 : -1), c0];
  let part1 = generateManhattanPath(r0, c0, intermediate[0], intermediate[1]);
  let part2 = generateManhattanPath(intermediate[0], intermediate[1], r1, c1);
  return part1.concat(part2.slice(1));
}
function nearestDefender(spawn) {
  let def1 = [8,2], def2 = [7,7];
  let dist1 = Math.abs(def1[0]-spawn[0]) + Math.abs(def1[1]-spawn[1]);
  let dist2 = Math.abs(def2[0]-spawn[0]) + Math.abs(def2[1]-spawn[1]);
  return dist1 <= dist2 ? def1 : def2;
}
function placeAttackers() {
  attackers = [];
  let usedCols = [];
  while (usedCols.length < 3) {
    let randCol = Math.floor(Math.random()*GRID_SIZE);
    if(!usedCols.includes(randCol)) usedCols.push(randCol);
  }
  let pathColors = ["orange","green","purple"];
  for(let i=0;i<3;i++){
    let col = usedCols[i];
    let spawn = [0, col];
    let chosenTarget = nearestDefender(spawn);
    let pathType = Math.random()<0.5 ? "straight" : "curve";
    let speed = Math.random()<0.5 ? 1 : 2;
    let fullPath = pathType==="straight" ? generateManhattanPath(spawn[0],spawn[1],chosenTarget[0],chosenTarget[1])
                                          : generateManhattanCurvePath(spawn[0],spawn[1],chosenTarget[0],chosenTarget[1]);
    if(fullPath[fullPath.length-1][0]!==chosenTarget[0] || fullPath[fullPath.length-1][1]!==chosenTarget[1]){
      fullPath.push(chosenTarget);
    }
    let steppedPath = [];
    for(let j=0;j<fullPath.length;j+=speed){
      steppedPath.push(fullPath[j]);
    }
    if(steppedPath[steppedPath.length-1][0]!==chosenTarget[0] || steppedPath[steppedPath.length-1][1]!==chosenTarget[1]){
      steppedPath[steppedPath.length-1] = chosenTarget;
    }
    attackers.push({fullPath, steppedPath, speed, pathColor: pathColors[i], currentIndex:0, baseTarget: chosenTarget});
  }
}
function countDefenders() {
  let count = 0;
  for(let r=0; r<GRID_SIZE; r++){
    for(let c=0; c<GRID_SIZE; c++){
      if(board[r][c]===1) count++;
    }
  }
  return count;
}
function drawBoard(boardArr) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let r=0;r<GRID_SIZE;r++){
    for(let c=0;c<GRID_SIZE;c++){
      ctx.strokeStyle="black";
      ctx.strokeRect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE);
      let val = boardArr[r][c];
      if(val===1){
        if(defenderImg) ctx.drawImage(defenderImg, c*CELL_SIZE+5, r*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10);
      }
    }
  }
  for(let tile of shotTiles){
    ctx.fillStyle="rgba(255,0,0,0.3)";
    ctx.fillRect(tile[1]*CELL_SIZE, tile[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE);
  }
  if(!shotTiles.length && hoveredCell){
    ctx.fillStyle="rgba(0,255,0,0.3)";
    ctx.fillRect(hoveredCell[1]*CELL_SIZE, hoveredCell[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE);
  }
}
function drawPaths() {
  for(let atk of attackers){
    ctx.setLineDash([5,5]);
    ctx.strokeStyle = atk.pathColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for(let i=0;i<atk.fullPath.length;i++){
      let pr = atk.fullPath[i][0];
      let pc = atk.fullPath[i][1];
      let x = pc*CELL_SIZE+CELL_SIZE/2;
      let y = pr*CELL_SIZE+CELL_SIZE/2;
      if(i===0) ctx.moveTo(x,y);
      else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.font="16px Arial";
  ctx.fillStyle="black";
  for(let atk of attackers){
    for(let i=1;i<atk.steppedPath.length;i++){
      let pr = atk.steppedPath[i][0];
      let pc = atk.steppedPath[i][1];
      let x = pc*CELL_SIZE+CELL_SIZE/2-5;
      let y = pr*CELL_SIZE+CELL_SIZE/2+5;
      ctx.fillText(i.toString(), x,y);
    }
    let cr = atk.steppedPath[atk.currentIndex][0];
    let cc = atk.steppedPath[atk.currentIndex][1];
    if(attackerImg) ctx.drawImage(attackerImg, cc*CELL_SIZE+5, cr*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10);
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
  for(let atk of attackers){ atk.currentIndex = 0; }
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
  if(gameOver) return;
  let remainingAttackers = [];
  for(let atk of attackers){
    if(atk.currentIndex < atk.steppedPath.length - 1){
      let nextIndex = atk.currentIndex + 1;
      let nextTile = atk.steppedPath[nextIndex];
      let shotHit = shotTiles.some(tile => tile[0]===nextTile[0] && tile[1]===nextTile[1]);
      if(shotHit){
        continue;
      } else {
        atk.currentIndex = nextIndex;
        if(atk.currentIndex === atk.steppedPath.length - 1){
          board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
          drawBoard(board);
          drawPaths();
          endGame("A defender was destroyed!");
          return;
        }
        remainingAttackers.push(atk);
      }
    } else {
      board[atk.baseTarget[0]][atk.baseTarget[1]] = 0;
      drawBoard(board);
      drawPaths();
      endGame("A defender was destroyed!");
      return;
    }
  }
  attackers = remainingAttackers;
  shotTiles = [];
  drawBoard(board);
  drawPaths();
}
function getDefendersAlive() {
  return countDefenders();
}
canvas.addEventListener("mousemove", function(e){
  if(gameOver) return;
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let col = Math.floor(x/CELL_SIZE);
  let row = Math.floor(y/CELL_SIZE);
  hoveredCell = [row, col];
  drawBoard(board);
  drawPaths();
});
canvas.addEventListener("click", function(){
  if(gameOver) return;
  if(hoveredCell){
    if(board[hoveredCell[0]][hoveredCell[1]]===1) return;
    for(let atk of attackers){
      let current = atk.steppedPath[atk.currentIndex];
      if(current[0]===hoveredCell[0] && current[1]===hoveredCell[1]) return;
    }
    let defendersAlive = getDefendersAlive();
    if(shotTiles.length < defendersAlive){
      shotTiles.push(hoveredCell);
    }
    drawBoard(board);
    drawPaths();
  }
});
newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);
