import { recoverSessionVariables } from "./utility/session.js";
const CAPTURE_SIZE = [50, 37.5];
const MAX_WIDTH_STATIVO = 144;
const MAX_HEIGHT_STATIVO = 126;
function letterToNumber(letter) {
  return letter.toLowerCase().charCodeAt(0) - "a".charCodeAt(0) + 1;
}

function numberToLetter(number) {
  if (number >= 1 && number <= 26) {
    return String.fromCharCode("A".charCodeAt(0) + number - 1); // Returns uppercase letter
  }
  return ""; // Return empty string if number is out of range
}

function convertToLabel(row, col) {
  return `${numberToLetter(row)}${col}`;
}

function convertToNumber(label) {
  const row = letterToNumber(label[0]) - 1;
  const col = parseInt(label.slice(1) - 1);
  // console.log(row, col);
  return [row, col];
}

document.addEventListener("DOMContentLoaded", function () {
  const sessionVars = recoverSessionVariables();
  const stepUp = document.getElementById("step-position-up");
  const stepDown = document.getElementById("step-position-down");
  const stepLeft = document.getElementById("step-position-left");
  const stepRight = document.getElementById("step-position-right");
  stepUp.addEventListener("click", function (e) {
    e.preventDefault();
    disableAllButtons();
    fetch("/api/capturer/move/step/w", {}).then((res) => {
      handleMovement(res);
    });
  });
  stepDown.addEventListener("click", function (e) {
    e.preventDefault();
    disableAllButtons();
    fetch("/api/capturer/move/step/s", {}).then((res) => {
      handleMovement(res);
    });
  });
  stepLeft.addEventListener("click", function (e) {
    e.preventDefault();
    disableAllButtons();
    fetch("/api/capturer/move/step/a", {}).then((res) => {
      handleMovement(res);
    });
  });
  stepRight.addEventListener("click", function (e) {
    e.preventDefault();
    disableAllButtons();
    fetch("/api/capturer/move/step/d", {}).then((res) => {
      handleMovement(res);
    });
  });
  function handleFreePosition(el, direction = "w") {
    //disable the grid
    const grid = document.getElementById("capturer-art-piece-pos");
    const gridLabel = document.getElementById("camera-position-label");
    gridLabel.classList.add("hide");
    grid.classList.add("hide");
    // //console.log(el, direction);
    const playIcon = el.querySelector(".play-icon");
    const pauseIcon = el.querySelector(".pause-icon");
    let isMove = false;
    if (playIcon.classList.contains("hide")) {
      isMove = true;
    }
    let url = `/api/capturer/move/free/${direction}`;
    if (isMove) {
      url = `/api/capturer/move/stop/${direction}`;
    }

    fetch(url, {}).then((res) => {
      if (res.status == 200) {
        res.json().then((data) => {
          if (data.move.state) {
            // is moving --> stop icon
            pauseIcon.classList.remove("hide");
            playIcon.classList.add("hide");
          } else {
            // not moving --> start icon
            pauseIcon.classList.add("hide");
            playIcon.classList.remove("hide");
          }
        });
      }
    });
  }
  const freeUp = document.getElementById("free-position-up");
  const freeDown = document.getElementById("free-position-down");
  const freeLeft = document.getElementById("free-position-left");
  const freeRight = document.getElementById("free-position-right");
  freeUp.addEventListener("click", function (e) {
    e.preventDefault();
    handleFreePosition(this, "w");
  });
  freeDown.addEventListener("click", function (e) {
    e.preventDefault();
    handleFreePosition(this, "s");
  });
  freeLeft.addEventListener("click", function (e) {
    e.preventDefault();
    handleFreePosition(this, "a");
  });
  freeRight.addEventListener("click", function (e) {
    e.preventDefault();
    handleFreePosition(this, "d");
  });

  function disableAllButtons() {
    //console.log("disableAllButtons");
    stepUp.disabled = true;
    stepDown.disabled = true;
    stepLeft.disabled = true;
    stepRight.disabled = true;
  }

  function enableAllButtons() {
    //console.log("enableAllButtons");
    stepUp.disabled = false;
    stepDown.disabled = false;
    stepLeft.disabled = false;
    stepRight.disabled = false;
  }
  function handleMovement(res) {
    enableAllButtons();
    if (res.status == 200) {
      //console.log(res);
      res.json().then((data) => {
        const pos = data.move.position;
        const coord = convertToNumber(pos);
        moveTargetToCell(coord[0], coord[1]);
      });
    }
  }

  //GRID
  let currentRows = sessionVars.nshoty;
  let currentColumns = sessionVars.nshotx;
  function generateGrid() {
    const rows = currentRows - 1;
    const columns = currentColumns - 1;
    const gridContainer = document.getElementById("grid");
    const positionRect =
      document.getElementsByClassName("position-rectangle")[0];
    const gridWidth = gridContainer.offsetWidth;
    const gridHeight = gridContainer.offsetHeight;
    //  MAX_WIDTH_STATIVO/CAPTURE_SIZE[0] = gridWidth/x
    //  MAX_HEIGHT_STATIVO/CAPTURE_SIZE[1] = gridHeight/y

    const x = (gridWidth * CAPTURE_SIZE[0]) / MAX_WIDTH_STATIVO;
    const y = (gridHeight * CAPTURE_SIZE[1]) / MAX_HEIGHT_STATIVO;
    positionRect.style.width = x + "px";
    positionRect.style.height = y + "px";
    gridContainer.innerHTML = "";

    // Set CSS custom properties for responsive font sizing
    gridContainer.style.setProperty("--rows", rows);
    gridContainer.style.setProperty("--columns", columns);

    // Set grid template
    gridContainer.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    gridContainer.style.gridTemplateRows = `repeat(${rows}, 1fr)`;

    // Create positition cells
    for (let i = 0; i < rows * columns; i++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      gridContainer.appendChild(cell);
    }

    // Generate shot grid
    const shotGrid = document.getElementById("shot-grid");
    // Clear existing shot grid
    shotGrid.innerHTML = "";

    // Calculate cell sizes
    const gridCellWidth = gridWidth / columns; // Size of each gridContainer cell
    const gridCellHeight = gridHeight / rows; // Size of each gridContainer cell
    const shotCellWidth = x; // Size of positionRect (width)
    const shotCellHeight = y; // Size of positionRect (height)
    // Set shot grid to relative positioning to allow absolute positioning of cells
    // shotGridContainer.style.position = "relative";

    // Create shot grid cells
    for (let row = 0; row < rows + 1; row++) {
      for (let col = 0; col < columns + 1; col++) {
        const cell = document.createElement("div");
        cell.className = "shot-grid-cell";
        // Calculate center position for this cell
        const centerX = col * gridCellWidth;
        const centerY = row * gridCellHeight;
        // Position the cell so its center aligns with the grid center
        const leftPos = centerX - shotCellWidth / 2;
        const topPos = centerY - shotCellHeight / 2;
        // Style the cell
        cell.style.position = "absolute";
        cell.style.left = leftPos + "px";
        cell.style.top = topPos + "px";
        cell.style.width = shotCellWidth + "px";
        cell.style.height = shotCellHeight + "px";
        cell.style.border = "1px solid #c49a73";
        cell.style.boxSizing = "border-box";
        cell.textContent = convertToLabel(rows - row + 1, col + 1);
        cell.style.fontSize =
          "calc(min(600px /" + columns + ", 400px /" + rows + ") * 0.15)";
        shotGrid.appendChild(cell);
      }
    }
    const gridContainerDIV = document.getElementById("capturer-art-piece-pos");
    gridContainerDIV.style.paddingTop = y / 2 + 5 + "px";
    gridContainerDIV.style.paddingLeft = x / 2 + 5 + "px";
    gridContainerDIV.style.paddingRight = x / 2 + "px";
    gridContainerDIV.style.paddingBottom = y / 2 + "px";

    const totalWidth = gridCellWidth * columns + shotCellWidth + 10;
    const totalHeight = gridCellHeight * rows + shotCellHeight + 10;
    // console.log(totalWidth, totalHeight);
    gridContainerDIV.style.width = totalWidth + "px"; //x + "px";
    gridContainerDIV.style.height = totalHeight + "px";
  }

  // Generate initial grid
  generateGrid();

  function moveTargetToCell(row, col) {
    //console.log(row, col);
    const target = document.getElementById("target");
    const grid = document.getElementById("grid");
    const gridWidth = grid.offsetWidth;
    const gridHeight = grid.offsetHeight;
    // Calculate cell size
    const cellWidth = gridWidth / (currentColumns - 1);
    const cellHeight = gridHeight / (currentRows - 1);
    //console.log(currentColumns, currentRows);
    // Calculate position (center of the cell)
    //console.log(cellWidth, cellHeight);
    // const x = col * cellWidth + cellWidth;
    // const y = gridHeight - (row * cellHeight + cellHeight);
    const x = col * cellWidth;
    const y = gridHeight - row * cellHeight;
    //console.log(x, y);
    target.style.left = x + "px";
    target.style.top = y + "px";
    const camera = document.getElementById("camera-position-label");
    camera.innerText =
      "Camera position x:" +
      col * sessionVars.stepx +
      " mm ; y:" +
      row * sessionVars.stepy +
      " mm";
  }

  moveTargetToCell(0, 0);

  //LOCKS and UNLOCKS
  const lockOpen = document.getElementById("icon-lock-open");
  const lockClose = document.getElementById("icon-lock-close");
  //the content to disable
  const freePosition = document.getElementById("free-position-group");
  const freeLabel = document.getElementById("free-position-label");
  // Initially hide the closed lock icon
  lockOpen.style.display = "none";
  freePosition.classList.add("disable");
  freeLabel.classList.add("disable");
  function toggleLockIcons() {
    if (lockOpen.style.display === "block") {
      //close
      lockOpen.style.display = "none";
      lockClose.style.display = "block";
      lockClose.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      freePosition.classList.add("disable");
      freeLabel.classList.add("disable");
    } else {
      lockClose.style.display = "none";
      lockOpen.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      lockOpen.style.display = "block";
      freePosition.classList.remove("disable");
      freeLabel.classList.remove("disable");
    }
  }

  lockOpen.addEventListener("click", toggleLockIcons);
  lockClose.addEventListener("click", toggleLockIcons);

  //checking vabration
  const vibrationButton = document.getElementById("vibration-btn");

  async function handleVibration() {
    const spinner = document.getElementsByClassName("spinner")[0];
    const vibrationOk = document.getElementById("vibration-ok");
    const vibrationOkHelp = document.getElementById("vibration-ok-help");
    const vibratingLabel = document.getElementById("vibrating-label");
    spinner.style.display = "block";
    vibrationOk.style.display = "none";
    vibrationOkHelp.style.display = "none";
    vibratingLabel.style.display = "none";
    vibrationButton.innerText = "Checking...";
    vibrationButton.background = "#F1DBC3";
    fetch("/api/capturer/checkMovements")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        spinner.style.display = "none";
        const vibrations = data.vibratios;
        // //console.log(vibrations);
        vibrationButton.innerText = "Check again";

        if (vibrations) {
          vibratingLabel.style.display = "block";
          vibrationOk.style.display = "none";
          vibrationOkHelp.style.display = "none";
        } else {
          vibrationOk.style.display = "block";
          vibrationOkHelp.style.display = "block";
          vibratingLabel.style.display = "none";
        }

        //move target
        const pos = convertToNumber(data.move.position) ?? [0, 0];
        //console.log(pos);
        moveTargetToCell(pos[0], pos[1]);
      })
      .catch((error) => {
        console.error(error);
        spinner.style.display = "none";
        vibrationButton.innerText = "Check again";
        vibratingLabel.style.display = "none";
        vibrationOk.style.display = "none";
        vibrationOkHelp.style.display = "none";
        //window.pywebview.api.alert_window(error.message);
        alert(`Error checking vibrations: ${error}`);
      });
  }

  vibrationButton.addEventListener("click", handleVibration);

  //intial call
  handleVibration();

  const phocusLabel = document.getElementById("phocus-label");
  phocusLabel.addEventListener("click", function (e) {
    e.preventDefault();
    fetch("/api/capturer/phocus", {}).then((res) => {});
  });
});
