import { recoverSessionVariables } from "./utility/session.js";
import { pathPicker, checkBox } from "./components/input.js";
import { progressBar } from "./components/progressBar.js";

const trashSVG = `<svg class="trash-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M20 6H16V5C16 4.20435 15.6839 3.44129 15.1213 2.87868C14.5587 2.31607 13.7956 2 13 2H11C10.2044 2 9.44129 2.31607 8.87868 2.87868C8.31607 3.44129 8 4.20435 8 5V6H4C3.73478 6 3.48043 6.10536 3.29289 6.29289C3.10536 6.48043 3 6.73478 3 7C3 7.26522 3.10536 7.51957 3.29289 7.70711C3.48043 7.89464 3.73478 8 4 8H5V19C5 19.7956 5.31607 20.5587 5.87868 21.1213C6.44129 21.6839 7.20435 22 8 22H16C16.7956 22 17.5587 21.6839 18.1213 21.1213C18.6839 20.5587 19 19.7956 19 19V8H20C20.2652 8 20.5196 7.89464 20.7071 7.70711C20.8946 7.51957 21 7.26522 21 7C21 6.73478 20.8946 6.48043 20.7071 6.29289C20.5196 6.10536 20.2652 6 20 6ZM10 5C10 4.73478 10.1054 4.48043 10.2929 4.29289C10.4804 4.10536 10.7348 4 11 4H13C13.2652 4 13.5196 4.10536 13.7071 4.29289C13.8946 4.48043 14 4.73478 14 5V6H10V5ZM17 19C17 19.2652 16.8946 19.5196 16.7071 19.7071C16.5196 19.8946 16.2652 20 16 20H8C7.73478 20 7.48043 19.8946 7.29289 19.7071C7.10536 19.5196 7 19.2652 7 19V8H17V19Z" fill="black"/>
</svg>`;
function renderDirectionImages(directionImagesArray) {
  const singlelightsList = document.getElementById(
    "nlights-singlelights-list-inner"
  );
  singlelightsList.innerHTML = "";
  // //console.log(directionImagesArray);
  if (!directionImagesArray) return;
  if (directionImagesArray.length === 1 && directionImagesArray[0] === "")
    return;
  for (let i = 0; i < directionImagesArray.length; i++) {
    const span = document.createElement("span");
    span.className = "label-light";
    span.innerHTML = `<span data-index="${i}">${trashSVG}</span>${directionImagesArray[i]}`;

    // Add click event to trash icon
    span.querySelector(".trash-icon").addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent event bubbling
      const index = parseInt(e.currentTarget.getAttribute("data-index"));
      directionImagesArray.splice(index, 1); // Remove the element
      const directionImagesString = directionImagesArray.join(",");
      fetch("/session-var", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ["nlights_direction_images_dir"]: directionImagesString,
        }),
      }).then(() => {
        renderDirectionImages(directionImagesArray);
      });
    });

    singlelightsList.appendChild(span);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const sessionVars = recoverSessionVariables();
  //picker single direction images
  //initial render of images
  let directionImagesArray = sessionVars.directionimages.split(",");
  // //console.log(directionImages);
  renderDirectionImages(directionImagesArray);

  pathPicker(
    "nlights_all_lights_image_dir",
    "nlights-all-lights-image-dir",
    "nlights-all-lights-image-dir-label",
    false
  );

  const singlelightsInput = document.getElementById(
    "nlights-direction-images-dir"
  );

  singlelightsInput.addEventListener("click", function (e) {
    const dialog = window.pywebview.api.open_file_dialog();
    dialog.then((resPath) => {
      try {
        const path = Array.isArray(resPath) ? resPath[0] : resPath;
        if (!path) {
          return;
        }
        // Update UI
        // directionImagesArray.push('"' + path + '"');
        if (directionImagesArray[0] === "") directionImagesArray.shift();
        directionImagesArray.push(path);
        const directionImagesString = directionImagesArray.join(",");
        // Send to server
        fetch("/session-var", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ["nlights_direction_images_dir"]: directionImagesString,
          }),
        }).then(() => {
          renderDirectionImages(directionImagesArray);
        });
      } catch (error) {
        console.error("Error updating session variable:", error);
        alert("Error saving file: " + error.message);
      }
    });
  });

  pathPicker(
    "nlights_output_directory",
    "nlights-output-directory",
    "nlights-output-directory-label"
  );

  //compute maps
  const computeMapsBtn = document.getElementById("compute-nlights");
  const nlightsComputing = document.getElementById("nlights-computing");
  const computingResults = document.getElementById("nlights-computing-results");
  const nlightsPauseBtn = document.getElementById(
    "nlights-computing-pause-btn"
  );
  const bar = progressBar("nlights-computation", {
    showSlider: true,
  });
  const barEl = document.getElementById("nlights-computing-bar-frame");
  nlightsComputing.style.display = "none";
  nlightsPauseBtn.style.display = "none";
  computingResults.style.display = "none";
  barEl.style.display = "none";
  bar.setProgress(0);
  let eventSource;
  computeMapsBtn.addEventListener("click", (e) => {
    e.preventDefault();
    computeMapsBtn.style.display = "none";
    nlightsComputing.style.display = "flex";
    nlightsPauseBtn.style.display = "flex";
    barEl.style.display = "flex";
    // Initialize with options
    bar.setProgress(0);
    eventSource = new EventSource("/api/nlights/compute/stream");
    eventSource.onmessage = function (event) {
      if (event.data === "done") {
        //FINE
        bar.setProgress(100);
        nlightsComputing.style.display = "none";
        computingResults.style.display = "flex";
        nlightsPauseBtn.style.display = "none";
      } else {
        if (event.data.includes("ERROR")) {
          console.error(event.data);
          eventSource.close();
          alert("Error computing maps: " + event.data);
          bar.setProgress(0);

          computeMapsBtn.style.display = "flex";
          nlightsComputing.style.display = "none";
          nlightsPauseBtn.style.display = "none";
          barEl.style.display = "none";
        } else {
          const value = parseInt(event.data);
          bar.setProgress(value);
        }
      }
    };
    eventSource.onerror = function (event) {
      // //console.log(event);
      console.error("EventSource failed.");
      eventSource.close();
    };
  });

  nlightsPauseBtn.addEventListener("click", (e) => {
    e.preventDefault();
    nlightsComputing.style.display = "none";
    computeMapsBtn.style.display = "flex";
    computingResults.style.display = "none";
    barEl.style.display = "none";
    bar.setProgress(0);
    eventSource.close();
  });
});
