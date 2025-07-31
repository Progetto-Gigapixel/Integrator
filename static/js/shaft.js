import { recoverSessionVariables } from "./utility/session.js";
import { pathPicker, checkBox } from "./components/input.js";
import { progressBar } from "./components/progressBar.js";
document.addEventListener("DOMContentLoaded", function () {
  const sessionVars = recoverSessionVariables();
  const checkerBoardFrame = document.getElementById("checker-board-frame");
  const developmentFrame = document.getElementById("development-frame");
  pathPicker(
    "shaft_colorchecker_path",
    "colorchecker-path",
    "colorchecker-path-label",
    false
  );
  pathPicker(
    "shaft_flatfield_file_path",
    "flatfield-path",
    "flatfield-path-label",
    false
  );
  pathPicker(
    "shaft_savein_path",
    "shaft-savein-path",
    "shaft-savein-path-label"
  );
  pathPicker(
    "shaft_develop_folder_path",
    "shaft-develop-folder-path",
    "shaft-develop-folder-path-label"
  );

  pathPicker(
    "shaft_output_path",
    "shaft-output-path",
    "shaft-output-path-label"
  );
  checkBox("shaft_process_subfolders", "shaft-process-subfolders");
  checkBox("shaft_overwrite", "shaft-overwrite");
  checkBox("shaft_light_balance", "shaft-light-balance");
  checkBox("shaft_sharpen", "shaft-sharpen");
  // progress bar
  const analyzeColorBtn = document.getElementById("analyze-shaft-btn");
  const shaftAnalyzing = document.getElementById("shaft-analyzing");
  const analysisResults = document.getElementById("analysis-results");
  const shaftPauseBtn = document.getElementById("shaft-pause-btn");
  const bar = progressBar("shaft-analysis", {
    showSlider: true,
  });
  const barEl = document.getElementById("barEl");
  const nextShaft = document.getElementById("next-shaft");
  let eventSource;
  barEl.style.display = "none";
  bar.setProgress(0);
  analyzeColorBtn.addEventListener("click", (e) => {
    eventSource = new EventSource("/api/shaft/colorchecker");
    e.preventDefault();
    analyzeColorBtn.style.display = "none";
    shaftAnalyzing.style.display = "flex";
    barEl.style.display = "flex";
    eventSource.onmessage = function (event) {
      // //console.log(event);
      if (event.data === "done") {
        bar.setProgress(100);
        eventSource.close();
        shaftAnalyzing.style.display = "none";
        analysisResults.style.display = "flex";
        nextShaft.style.display = "flex";
      } else {
        if (event.data.includes("Color correction")) {
          analysisResults.getElementsByTagName("span")[0].innerText =
            event.data;
        } else {
          if (event.data.includes("ERROR")) {
            console.error(event.data);
            eventSource.close();
            alert("Error \n\n" + event.data);
            bar.setProgress(0);
            analyzeColorBtn.style.display = "flex";
            shaftAnalyzing.style.display = "none";
            barEl.style.display = "none";
          } else {
            const value = parseInt(event.data);
            bar.setProgress(value);
          }
        }
      }
    };
    eventSource.onerror = function () {
      console.error("EventSource failed.");
      eventSource.close();
    };
    bar.setProgress(0);
  });
  shaftPauseBtn.addEventListener("click", (e) => {
    e.preventDefault();
    shaftAnalyzing.style.display = "none";
    analyzeColorBtn.style.display = "flex";
    eventSource.close();
    analysisResults.style.display = "none";
    barEl.style.display = "none";
    bar.setProgress(0);
  });

  nextShaft.addEventListener("click", (e) => {
    e.preventDefault();
    shaftAnalyzing.style.display = "none";
    analysisResults.style.display = "none";
    barEl.style.display = "none";
    bar.setProgress(0);
    analyzeColorBtn.style.display = "flex";
    checkerBoardFrame.classList.add("opace");
    developmentFrame.classList.remove("opace");
    // Send to server
    fetch("/session-var", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        is_shaft_developed: true,
      }),
    });
  });

  //development
  const developmentBar = progressBar("shaft-development", {
    showSlider: true,
  });
  const devBarFrame = document.getElementById("shaft-devBar-frame");
  const developColorBtn = document.getElementById("development-shaft-btn");
  const developingShaft = document.getElementById("shaft-developing");
  const developingPauseBtn = document.getElementById("shaft-pause-dev-btn");
  const developmentResults = document.getElementById(
    "shaft-development-results"
  );
  devBarFrame.style.display = "none";
  developingShaft.style.display = "none";
  developmentResults.style.display = "none";
  let devEventSource;
  // console.log(sessionVars.is_shaft_developed);

  if (sessionVars.is_shaft_developed) {
    checkerBoardFrame.classList.add("opace");
    developmentFrame.classList.remove("opace");
  }
  // developmentFrame.classList.remove("opace");
  developColorBtn.addEventListener("click", (e) => {
    e.preventDefault();
    //console.log("developColorBtn clicked");
    developColorBtn.style.display = "none";
    developingShaft.style.display = "flex";
    devBarFrame.style.display = "flex";
    // Initialize with options
    developmentBar.setProgress(0);

    devEventSource = new EventSource("/api/shaft/development/stream");
    devEventSource.onmessage = function (event) {
      if (event.data === "done") {
        developmentBar.setProgress(100);
        devEventSource.close();
        developingShaft.style.display = "none";
        developmentResults.style.display = "flex";
      } else {
        if (event.data.includes("ERROR")) {
          console.error(event.data);
          eventSource.close();
          alert("Error \n\n" + event.data);
          developmentBar.setProgress(0);
          developColorBtn.style.display = "flex";
          developingShaft.style.display = "none";
          devBarFrame.style.display = "none";
        } else {
          const value = parseInt(event.data);
          developmentBar.setProgress(value);
        }
      }
    };
    devEventSource.onerror = function () {
      console.error("EventSource failed.");
      devEventSource.close();
    };
  });

  developingPauseBtn.addEventListener("click", (e) => {
    e.preventDefault();
    devEventSource.close();
    developingShaft.style.display = "none";
    developColorBtn.style.display = "flex";
    developmentResults.style.display = "none";
    devBarFrame.style.display = "none";
    developmentBar.setProgress(0);
  });
});
