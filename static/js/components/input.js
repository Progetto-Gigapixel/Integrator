export function pathPicker(
  sessionVarName,
  inputId,
  labelId = null,
  folder = true,
  customFunction = null
) {
  const inputElement = document.getElementById(inputId);
  const labelElement = labelId ? document.getElementById(labelId) : null;
  inputElement.addEventListener("click", function (e) {
    let dialog;
    if (folder) {
      dialog = window.pywebview.api.open_folder_dialog();
    } else {
      dialog = window.pywebview.api.open_file_dialog();
    }
    dialog.then((resPath) => {
      try {
        const path = Array.isArray(resPath) ? resPath[0] : resPath;
        if (!path) {
          return;
        }
        // Update UI
        inputElement.value = path;
        if (labelElement) {
          labelElement.innerText = path;
        }
        // //console.log(path);

        // Send to server
        fetch("/session-var", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            [sessionVarName]: path,
          }),
        });
        if (customFunction) {
          customFunction(path);
        }
      } catch (error) {
        console.error("Error updating session variable:", error);
        alert("Error saving file: " + error.message);
      }
    });
  });
}

export function checkBox(sessionVarName, inputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.addEventListener("change", function (e) {
    e.preventDefault();
    try {
      const isChecked = inputElement.checked;
      // //console.log(`${sessionVarName}: ${isChecked}`);
      // Send to server
      fetch("/session-var", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          [sessionVarName]: isChecked,
        }),
      });
    } catch (error) {
      console.error("Error updating session variable:", error);
      alert("Error saving file: " + error.message);
    }
  });
}

/**
 * Given an HTML element, this function takes the innerText of the element
 * and splits it into two parts, where the first part is everything before the
 * first comma, and the second part is everything after the first comma.
 * It then creates a new string with the first part in bold and the second part
 * normal, and sets this as the innerHTML of the element.
 * Finally, it returns an array with the two parts.
 *
 * @param {HTMLElement} element The element to manipulate
 * @returns {Array<string>} The two parts of the text
 */
export function handleBoldTextArea(element) {
  const text = element.innerText;
  const firstSpaceIndex = text.indexOf(",");
  const bold = text.substring(0, firstSpaceIndex);
  const normal = text.substring(firstSpaceIndex + 1);
  element.innerHTML = `<span class="label-bold">${bold}<span class="label">, ${normal}</span></span>`;
  return [bold, normal];
}

export function textField(sessionVarName, inputId, customFunction = null) {
  const inputElement = document.getElementById(inputId);
  inputElement.addEventListener("change", function (e) {
    e.preventDefault();
    try {
      const value = inputElement.value;

      fetch("/session-var", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          [sessionVarName]: value,
        }),
      });
      if (customFunction) {
        customFunction(value);
      }
    } catch (error) {
      console.error("Error updating session variable:", error);
    }
  });
}
