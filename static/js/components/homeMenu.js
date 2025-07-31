let selectedFileData = null;

document.addEventListener(
  "DOMContentLoaded",
  () => {
    window.addEventListener("pywebviewready", function () {
      document
        .getElementById("close-button")
        .addEventListener("click", function (e) {
          print("close");
          window.pywebview.api.destroy_window();
        });
      if (!document.getElementById("home-menu-save-project")) return;
      document
        .getElementById("home-menu-save-project")
        .addEventListener("click", function (e) {
          window.pywebview.api.save_file_dialog().then((resPath) => {
            try {
              //console.log(resPath);
              const path = Array.isArray(resPath) ? resPath[0] : resPath;

              const response = fetch("/api/save_session", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  path: path,
                }),
              });
            } catch (error) {
              alert("Error saving file: " + error.message);
            }
          });
        });
    });

    const openProjectButton = document.getElementById("home-menu-open-project");
    if (openProjectButton) {
      openProjectButton.addEventListener("click", function (e) {
        document.getElementById("json-upload").click();
      });
    }

    document
      .getElementById("json-upload")
      .addEventListener("change", function (e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
          try {
            selectedFileData = JSON.parse(e.target.result);
            //console.log(selectedFileData);
            processSelectedFile();
          } catch (error) {
            alert("Invalid JSON file");
          }
        };
        reader.readAsText(file);
      });

    async function processSelectedFile() {
      if (!selectedFileData) return;

      try {
        const response = await fetch("/api/projects", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(selectedFileData),
        });

        const result = await response.json();
        // document.getElementById("sync-status").innerHTML = `
        //     <p style="color: green">Successfully processed project!</p>
        //     <p>ID: ${result.id}</p>
        // `;
        //console.log(result);

        // Optionally trigger full sync
        await syncWithDatabase();
        await fetch("/api/load_session", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(selectedFileData),
        });
        //redirect to /project
        window.location.href = `/projectpage`;
      } catch (error) {}
    }

    async function syncWithDatabase() {
      try {
        const response = await fetch("/api/projects/sync", {
          method: "POST",
        });
        const result = await response.json();
        //console.log("Sync result:", result);
      } catch (error) {
        console.error("Sync failed:", error);
      }
    }
  },
  { once: true }
);
