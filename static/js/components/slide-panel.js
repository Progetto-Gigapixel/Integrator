let panelOpen = false;

function togglePanel() {
  if (panelOpen) {
    closePanel();
  } else {
    openPanel();
  }
}

function openPanel() {
  const panel = document.getElementById("slidingPanel");
  const overlay = document.getElementById("panelOverlay");

  panel.classList.add("active");
  overlay.classList.add("active");
  panelOpen = true;

  // Prevent body scroll when panel is open on mobile
  if (window.innerWidth <= 768) {
    document.body.style.overflow = "hidden";
  }
}

function closePanel() {
  const panel = document.getElementById("slidingPanel");
  const overlay = document.getElementById("panelOverlay");

  panel.classList.remove("active");
  overlay.classList.remove("active");
  panelOpen = false;

  // Restore body scroll
  document.body.style.overflow = "";
}

// Close panel with Escape key
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape" && panelOpen) {
    closePanel();
  }
});

// Handle window resize
window.addEventListener("resize", function () {
  if (window.innerWidth > 768) {
    document.getElementById("panelOverlay").classList.remove("active");
    document.body.style.overflow = "";
  } else if (panelOpen) {
    document.getElementById("panelOverlay").classList.add("active");
    document.body.style.overflow = "hidden";
  }
});
