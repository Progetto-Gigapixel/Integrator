import { recoverSessionVariables } from "./utility/session.js";
import { pathPicker, checkBox, textField } from "./components/input.js";
import { progressBar } from "./components/progressBar.js";
let files;

const nameText = document.getElementById("modelviewer-name");
document.addEventListener("DOMContentLoaded", function () {
  const sessionVars = recoverSessionVariables();
  //LOCKS and UNLOCKS
  const lockOpen = document.getElementById("icon-lock-open");
  const lockClose = document.getElementById("icon-lock-close");
  const infoFrame = document.getElementById("modelviewer-info-frame");
  lockOpen.style.display = "none";
  infoFrame.classList.add("opace");
  function toggleLockIcons() {
    if (lockOpen.style.display === "none") {
      lockOpen.style.display = "block";
      lockOpen.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      lockClose.style.display = "none";
      infoFrame.classList.remove("opace");
    } else {
      lockOpen.style.display = "none";
      lockClose.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      lockClose.style.display = "block";
      infoFrame.classList.add("opace");
    }
  }
  //the content to disable
  lockClose.addEventListener("click", toggleLockIcons);
  lockOpen.addEventListener("click", toggleLockIcons);
  //TEXT FIELDS
  textField("title", "modelviewer-name");
  textField("artist", "modelviewer-author");
  textField("height", "modelviewer-lenght");
  textField("width", "modelviewer-width");

  //PATH PICKER
  pathPicker(
    "modelviewer_artwork_dir_path",
    "modelviewer-artwork-dir-path",
    "modelviewer-artwork-dir-path-label"
  );
  pathPicker(
    "modelviewer_template_dir_path",
    "modelviewer-template-dir-path",
    "modelviewer-template-dir-path-label"
  );
  const bar = progressBar("modelviewer-templating", {
    showSlider: true,
  });
  pathPicker(
    "modelviewer_output_path",
    "modelviewer-output-path",
    "modelviewer-output-path-label"
  );
  const barDiv = document.getElementById("modelviewer-templating-bar-frame");
  const inputElement = document.getElementById("save-modelviewer");
  const results = document.getElementById("modelviewer-templating-results");
  barDiv.style.display = "none";
  results.style.display = "none";
  let eventSource;
  inputElement.addEventListener("click", function (e) {
    inputElement.style.display = "none";
    barDiv.style.display = "flex";
    bar.setProgress(0);
    eventSource = new EventSource("/api/modelviewer/unity/template/stream");
    eventSource.onmessage = function (event) {
      // //console.log(event);
      if (event.data === "done") {
        bar.setProgress(100);
        eventSource.close();
        // barDiv.style.display = "none";
        inputElement.style.display = "none";
        results.style.display = "flex";
      } else {
        const value = parseInt(event.data);
        bar.setProgress(value);
      }
    };
    eventSource.onerror = function () {
      console.error("EventSource failed.");
      eventSource.close();
    };
  });

  const unityLabel = document.getElementById("unity-label");
  unityLabel.addEventListener("click", function (e) {
    e.preventDefault();
    fetch("/api/modelviewer/unity/open", {}).then((res) => {});
  });
});
