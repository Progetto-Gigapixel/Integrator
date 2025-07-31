import { recoverSessionVariables } from "./utility/session.js";
import { pathPicker, checkBox } from "./components/input.js";
import { progressBar } from "./components/progressBar.js";
document.addEventListener("DOMContentLoaded", function () {
  const sessionVars = recoverSessionVariables();
  pathPicker(
    "imagematcher_maps_path",
    "imagematcher-maps-path",
    "imagematcher-maps-path-label"
  );
  pathPicker(
    "imagematcher_tex_out_path",
    "imagematcher-tex-out-path",
    "imagematcher-tex-out-path-label"
  );

  // pathPicker(
  //   "imagematcherReflectionPath",
  //   "reflection-path",
  //   "reflection-path-label"
  // );
  // pathPicker("imagematcherNormalPath", "normal-path", "normal-path-label");
  // pathPicker("imagematcherMeshPath", "mesh-path", "mesh-path-label");
  // pathPicker("albedo-save-path", "albedo-save-path", "albedo-save-path-label");
  // pathPicker(
  //   "imagematcherReflectionSavePath",
  //   "reflection-save-path",
  //   "reflection-save-path-label"
  // );
  // pathPicker(
  //   "imagematcherNormalSavePath",
  //   "normal-save-path",
  //   "normal-save-path-label"
  // );
  // pathPicker(
  //   "imagematcherMeshSavePath",
  //   "mesh-save-path",
  //   "mesh-save-path-label"
  // );
  const matcherLocked = document.getElementById("matcher-locked");
  const matcherValidateBtn = document.getElementById("matcher-validate-btn");
  const selectMapsFrame = document.getElementById("matcher-selectmaps-frame");
  const alignMapsFrame = document.getElementById("matcher-alignmaps-frame");
  matcherValidateBtn.addEventListener("click", function (e) {
    e.preventDefault();
    selectMapsFrame.classList.add("opace");
    matcherLocked.style.display = "flex";
    alignMapsFrame.classList.remove("opace");
  });
  matcherLocked.style.display = "none";
  matcherLocked.addEventListener("click", function (e) {
    selectMapsFrame.classList.remove("opace");
    matcherLocked.style.display = "none";
    alignMapsFrame.classList.add("opace");
  });

  //ALIGNMENT
  //le barre di progresso
  const alignBar = progressBar("imagematcher-alignment-bar", {
    showSlider: true,
  });
  const alignBarDiv = document.getElementById(
    "imagematcher-alignmaps-bar-frame"
  );
  alignBarDiv.style.display = "none";
  alignBar.setProgress(0);

  const mergeBar = progressBar("imagematcher-merge-bar", {
    showSlider: true,
  });
  const mergeBarDiv = document.getElementById("imagematcher-merge-bar-frame");
  mergeBarDiv.style.display = "none";
  mergeBar.setProgress(0);
  //allineamento
  const matcherAligmentBtn = document.getElementById("matcher-alignmaps-btn");
  const matcherManualBtn = document.getElementById(
    "matcher-alignmaps-manual-btn"
  );
  const matcherAgainBtn = document.getElementById(
    "matcher-alignmaps-again-btn"
  );
  const matcherAligningBtn = document.getElementById(
    "matcher-alignmaps-aligning-btn"
  );
  const matcherMergeBtn = document.getElementById("matcher-merge-btn");
  matcherMergeBtn.style.display = "none";
  matcherAgainBtn.style.display = "none";
  matcherManualBtn.style.display = "none";
  matcherAligningBtn.style.display = "none";

  const handleAlignment = (e) => {
    e.preventDefault();
    let eventSource;
    matcherAligmentBtn.style.display = "none";
    matcherAligningBtn.style.display = "flex";
    matcherAgainBtn.style.display = "none";
    matcherManualBtn.style.display = "none";
    matcherMergeBtn.style.display = "none";
    alignBarDiv.style.display = "flex";
    //progress fasulli
    alignBar.setProgress(0);
    eventSource = new EventSource("/api/imagematcher/alignment/stream");
    eventSource.onmessage = function (event) {
      //console.log(event.data);
      if (event.data === "done") {
        //FINE
        eventSource.close();
        alignBar.setProgress(100);
        matcherAligningBtn.style.display = "none";
        matcherMergeBtn.style.display = "flex";
        matcherManualBtn.style.display = "flex";
        // matcherAgainBtn.style.display = "flex";
      } else {
        if (event.data.includes("ERROR")) {
          console.error(event.data);
          eventSource.close();
          alert("Error \n\n" + event.data);
          alignBar.setProgress(0);
        } else {
          const value = parseInt(event.data);
          alignBar.setProgress(value);
        }
      }
    };
    eventSource.onerror = function (event) {
      console.error("EventSource failed.");
      eventSource.close();
    };
  };
  matcherAligmentBtn.addEventListener("click", handleAlignment);
  matcherAgainBtn.addEventListener("click", handleAlignment);
  matcherManualBtn.addEventListener("click", (e) => {
    fetch("/api/imagematcher/alignment/manual", {}).then((res) => {
      if (res.status == 200) {
        //console.log(res);
      } else {
        //console.log(res);
      }
    });
  });

  const handleMerge = (e) => {
    e.preventDefault();
    let eventSource;
    matcherManualBtn.style.display = "none";
    matcherMergeBtn.style.display = "none";
    alignBarDiv.style.display = "none";
    mergeBarDiv.style.display = "flex";
    //progress fasulli
    mergeBar.setProgress(0);
    eventSource = new EventSource("/api/imagematcher/merge/stream");
    eventSource.onmessage = function (event) {
      //console.log(event.data);
      if (event.data === "done") {
        //FINE
        eventSource.close();
        mergeBar.setProgress(100);
        // matcherAgainBtn.style.display = "flex";
      } else {
        if (event.data.includes("ERROR")) {
          console.error(event.data);
          eventSource.close();
          alert("Error \n\n" + event.data);
          alignBar.setProgress(0);
          matcherManualBtn.style.display = "flex";
          matcherMergeBtn.style.display = "flex";
          mergeBarDiv.style.display = "none";
        } else {
          const value = parseInt(event.data);
          alignBar.setProgress(value);
        }
      }
    };
    eventSource.onerror = function (event) {
      console.error("EventSource failed.");
      eventSource.close();
    };
  };

  matcherMergeBtn.addEventListener("click", handleMerge);
});
