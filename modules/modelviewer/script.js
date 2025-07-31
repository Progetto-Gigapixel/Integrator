addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);

  const setParam = (val, id) => (document.getElementById(id).value = val);

  setParam(params.get("name") ?? "Name", "name");
  setParam(params.get("author") ?? "Author", "author");
  setParam(params.get("w") ?? 0, "width");
  setParam(params.get("l") ?? 0, "length");
  document.getElementById("art-img").src = params.get("img") ?? "";

  const lock = document.getElementById("lock");
  lock.addEventListener("click", onLockToggle);

  const fileHandler = new FileHandler();
  const submit = document.getElementById("submit-btn");
  const upload = document.getElementById("zip-upload");
  const artwork = document.getElementById("art-upload");
  submit.addEventListener("click", fileHandler.onSubmit.bind(fileHandler));
  upload.addEventListener("click", fileHandler.onZipUpload.bind(fileHandler));
  artwork.addEventListener("click", fileHandler.onArtUpload.bind(fileHandler));
});

function onLockToggle() {
  const lock = document.getElementById("lock");
  lock.src = `assets/${
    lock.src.endsWith("/lock.svg") ? "unlock.svg" : "lock.svg"
  }`;
  const fields = document.getElementsByClassName("field");
  for (const field of fields) {
    field.disabled = !field.disabled;
  }
}

class FileHandler {
  constructor() {
    this.files = null;
    this.artworkHandler = null;
  }

  async onSubmit() {
    const dirName = this.newDirectoryName();
    const handle = await window.showDirectoryPicker();
    const newDir = await handle.getDirectoryHandle(dirName, {
      create: true,
    });

    if (this.artworkHandler === null) {
      alert("You must select an artwork directory first!");
      return;
    }

    await this.artworkHandler.requestPermission({ mode: "read" });
    await this.createInfo(newDir);
    await this.unzipTo(newDir);
    let subDirHandle = newDir;
    for (const subDir of ["Assets", "Artwork", "Maps"]) {
      subDirHandle = await subDirHandle.getDirectoryHandle(subDir, {
        create: true,
      });
    }
    await this.copyTextures(this.artworkHandler, subDirHandle);
    this.appendFeedback();
  }

  async copyTextures(sourceHandle, destHandle) {
    for await (const entry of sourceHandle.values()) {
      if (entry.kind === "file") {
        const sourceFile = await entry.getFile();
        const destFileHandle = await destHandle.getFileHandle(entry.name, {
          create: true,
        });
        const writable = await destFileHandle.createWritable();
        await writable.write(await sourceFile.arrayBuffer());
        await writable.close();
      } else if (entry.kind === "directory") {
        const newDestDir = await destHandle.getDirectoryHandle(entry.name, {
          create: true,
        });
        await this.copyTextures(entry, newDestDir);
      }
    }
  }

  async onArtUpload() {
    const handle = await window.showDirectoryPicker();
    this.artworkHandler = handle;
  }

  async createInfo(rootHandler) {
    const fileHandle = await rootHandler.getFileHandle("info.txt", {
      create: true,
    });
    const writable = await fileHandle.createWritable();
    const fields = document.getElementsByClassName("field");
    let file_content = "";
    for (const field of fields) {
      file_content = file_content.concat(`${field.value}\n`);
    }
    await writable.write(file_content);
    await writable.close();
  }

  async unzipTo(rootHandler) {
    if (this.files === null) {
      alert("Upload the template first!");
      return;
    }
    for (const file of this.files.filter((x) => !x.dir)) {
      const subDir = file.name.split("/");
      const fileName = subDir.pop();

      let currentDir = rootHandler;
      for (const dir of subDir) {
        currentDir = await currentDir.getDirectoryHandle(dir, {
          create: true,
        });
      }

      const fileHandle = await currentDir.getFileHandle(fileName, {
        create: true,
      });
      const writable = await fileHandle.createWritable();
      await writable.write(await file.async("blob"));
      await writable.close();
    }
  }

  async onZipUpload() {
    const zipName = document.getElementById("zip-types").value;
    if (zipName === null || zipName === "") {
      alert("You must select a template first!");
      return;
    }

    const handle = await window.showDirectoryPicker();
    let zip;
    let file;
    try {
      zip = await handle.getFileHandle(zipName);
      file = await zip.getFile();
    } catch {
      alert("Invalid folder selected!");
      return;
    }
    const reader = new FileReader();
    reader.onload = async (buffer) => {
      const zipper = new JSZip();
      const zip = await zipper.loadAsync(buffer.target.result);
      this.files = Object.values(zip.files);
    };

    reader.readAsArrayBuffer(file);
  }

  appendFeedback() {
    const feedbackDiv = document.getElementById("create-feedback");
    feedbackDiv.innerHTML = "";
    const openUnityHub = document.createElement("a");
    openUnityHub.href = "unityhub://";
    openUnityHub.innerHTML = "<b>Open in Unity Hub.</b>";
    const successParagraph = document.createElement("p");
    successParagraph.innerHTML = "Template was successfully generated. ";
    successParagraph.appendChild(openUnityHub);
    feedbackDiv.appendChild(successParagraph);
  }

  newDirectoryName() {
    const padStr = (x) => String(x).padStart(2, "0");
    const now = new Date();
    return [
      now.getFullYear(),
      padStr(now.getMonth() + 1),
      padStr(now.getDate()),
      padStr(now.getHours()),
      padStr(now.getMinutes()),
      padStr(now.getSeconds()),
      document.getElementById("name").value,
      document.getElementById("author").value,
    ]
      .join("_")
      .replaceAll(" ", "_");
  }
}
