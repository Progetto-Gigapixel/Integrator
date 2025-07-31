import { handleBoldTextArea } from "./components/input.js";
const MAX_WIDTH_STATIVO = 1440;
const MAX_HEIGHT_STATIVO = 1260;
document.addEventListener("DOMContentLoaded", function () {
  const editableDiv = document.getElementById("editable-title-author");
  handleBoldTextArea(editableDiv);
  editableDiv.addEventListener("focusout", function () {
    const resText = handleBoldTextArea(this);
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: resText[0],
        artist: resText[1],
      }),
    });
  });
  const editableDiv2 = document.getElementById("editable-location-address");
  handleBoldTextArea(editableDiv2);
  editableDiv2.addEventListener("focusout", function () {
    const resText = handleBoldTextArea(this);
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: resText[0],
        address: resText[1],
      }),
    });
  });
  const verticalButton = document.getElementById("new-stand-vertical-button");
  const horizontalButton = document.getElementById(
    "new-stand-horizontal-button"
  );
  const displacementFrame = document.getElementById("displacement-frame");
  const stepDisplacementFrame = document.getElementById(
    "step-displacement-frame"
  );
  const shotNumberFrame = document.getElementById("shot-number-frame");
  const vibrationTimeFrame = document.getElementById("vibration-time-frame");
  verticalButton.addEventListener("click", function (e) {
    e.preventDefault();
    if (verticalButton.classList.contains("active")) {
      return;
    }
    const data = { stand_type: "vertical" };
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        horizontalButton.classList.remove("active");
        verticalButton.classList.toggle("active");
        displacementFrame.classList.remove("hide");
        stepDisplacementFrame.classList.remove("hide");
        shotNumberFrame.classList.remove("hide");
        vibrationTimeFrame.classList.remove("hide");
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });

  horizontalButton.addEventListener("click", function (e) {
    e.preventDefault();
    if (horizontalButton.classList.contains("active")) {
      return;
    }
    const data = { stand_type: "horizontal" };
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        verticalButton.classList.remove("active");
        horizontalButton.classList.toggle("active");
        displacementFrame.classList.add("hide");
        stepDisplacementFrame.classList.add("hide");
        shotNumberFrame.classList.add("hide");
        vibrationTimeFrame.classList.add("hide");
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });

  //LOCKS and UNLOCKS
  const lockOpen = document.getElementById("icon-lock-open");
  const lockClose = document.getElementById("icon-lock-close");
  //the content to disable
  const stepLabel = document.getElementById("step-displacement-label");
  const stepInner = document.getElementById("step-displacement-inner");
  // Initially hide the closed lock icon
  lockClose.style.display = "none";
  // Toggle between the two icons on click (either one)
  function toggleLockIcons() {
    if (lockOpen.style.display === "none") {
      lockOpen.style.display = "block";
      lockOpen.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      lockClose.style.display = "none";
      stepLabel.classList.add("disable");
      stepInner.classList.add("disable");
    } else {
      lockOpen.style.display = "none";
      lockClose.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      lockClose.style.display = "block";
      stepLabel.classList.remove("disable");
      stepInner.classList.remove("disable");
    }
  }

  lockOpen.addEventListener("click", toggleLockIcons);
  lockClose.addEventListener("click", toggleLockIcons);

  //MUTATION OBESERVER
  const sensivityDiv = document.getElementById("new-sensivity-dropdown");
  const outputDiv = document.getElementById("new-output-dropdown");
  const validateButton = document.getElementById("validate-button");
  const checkValidateButton = () => {
    const sensivityValue = sensivityDiv.textContent.trim();
    const outputValue = outputDiv.textContent.trim();
    if (sensivityValue == "Sensitivity" || outputValue == "Output type") {
      validateButton.classList.add("disable");
    } else {
      validateButton.classList.remove("disable");
    }
  };
  //check at first
  checkValidateButton();
  const observer = new MutationObserver(checkValidateButton);
  observer.observe(sensivityDiv, {
    childList: true,
    subtree: true,
    characterData: true,
  });
  observer.observe(outputDiv, {
    childList: true,
    subtree: true,
    characterData: true,
  });
  //vibration input
  const vibrationInput = document.getElementById("vibration-time");
  vibrationInput.addEventListener("change", () => {
    const value = vibrationInput.value;
    const data = { vibration_check_time: +value };
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then((response) => {
      response.json().then((data) => {});
    });
  });

  //calculate automatically the shot numbers
  const displacementXInput = document.getElementById("displacement-x");
  const displacementYInput = document.getElementById("displacement-y");
  const stepDistanceXInput = document.getElementById("step-distance-x");
  const stepDistanceYInput = document.getElementById("step-distance-y");
  const shotNumberXInput = document.getElementById("shot-number-x");
  const shotNumberYInput = document.getElementById("shot-number-y");
  displacementXInput.addEventListener("change", () => {
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ width: +displacementXInput.value }),
    });
  });

  displacementYInput.addEventListener("change", () => {
    fetch("/session-var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ height: +displacementYInput.value }),
    });
  });
  stepDistanceXInput.addEventListener("change", () => {
    const xValue = MAX_WIDTH_STATIVO; //fisso dello stativo
    const stepXValue = +stepDistanceXInput.value;
    if (xValue && stepXValue) {
      const xShotNumber = Math.floor(xValue / stepXValue);
      shotNumberXInput.value = xShotNumber;
      generateGrid();
    }
  });

  stepDistanceYInput.addEventListener("change", () => {
    const yValue = MAX_HEIGHT_STATIVO; //fisso dello stativo
    const stepYValue = +stepDistanceYInput.value;
    if (yValue && stepYValue) {
      const yShotNumber = Math.floor(yValue / stepYValue);
      shotNumberYInput.value = yShotNumber;
      generateGrid();
    }
  });

  //generate grid
  function generateGrid() {
    const shotNumberXInput = document.getElementById("shot-number-x");
    const shotNumberYInput = document.getElementById("shot-number-y");
    const stepXValue = +stepDistanceXInput.value;
    const stepYValue = +stepDistanceYInput.value;
    const columns = parseInt(shotNumberXInput.value);
    const rows = parseInt(shotNumberYInput.value);
    const gridContainer = document.getElementById("grid");

    //update the backend sessione
    fetch("/session-var", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        nshoty: rows,
        nshotx: columns,
        stepx: stepXValue,
        stepy: stepYValue,
      }),
    });
    // Clear existing grid
    gridContainer.innerHTML = "";

    // Set CSS custom properties for responsive font sizing
    gridContainer.style.setProperty("--rows", rows);
    gridContainer.style.setProperty("--columns", columns);

    // Set grid template
    gridContainer.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    gridContainer.style.gridTemplateRows = `repeat(${rows}, 1fr)`;

    // Create grid cells
    for (let i = 0; i < rows * columns; i++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";

      gridContainer.appendChild(cell);
    }
  }
  // Generate initial grid
  const xValue = MAX_WIDTH_STATIVO;
  const stepXValue = +stepDistanceXInput.value;
  const yValue = MAX_HEIGHT_STATIVO;
  const stepYValue = +stepDistanceYInput.value;
  if (yValue && stepYValue && xValue && stepXValue) {
    const yShotNumber = Math.floor(yValue / stepYValue);
    const xShotNumber = Math.floor(xValue / stepXValue);
    shotNumberYInput.value = yShotNumber;
    shotNumberXInput.value = xShotNumber;
    generateGrid();
  }
});
