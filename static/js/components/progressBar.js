export function progressBar(id, options = {}) {
  const {
    initialValue = 0,
    animateOnLoad = true,
    animationDelay = 500,
    animationDuration = 1000,
    targetValue = 0,
    showPercentage = true,
    showSlider = false,
  } = options;

  let currentProgress = initialValue;
  const progressBar = document.getElementById(id + "-progress-bar");
  const percentageDisplay = document.getElementById(id + "-percentage");
  const label = document.getElementById(id + "-label");

  function setProgress(value) {
    currentProgress = Math.max(0, Math.min(100, value));
    updateProgressBar();
  }

  function updateProgressBar() {
    if (progressBar) progressBar.style.width = currentProgress + "%";
    if (percentageDisplay && showPercentage) {
      percentageDisplay.textContent = Math.round(currentProgress) + "%";
    }

    if (label) {
      if (currentProgress >= 100) {
        label.textContent = "Complete";
        progressBar?.classList.add("complete");
      } else {
        label.textContent = "In progress";
        progressBar?.classList.remove("complete");
      }
    }
  }

  function animateProgress(target = targetValue, duration = animationDuration) {
    let start = null;
    const initialProgress = currentProgress;
    const progressDiff = target - initialProgress;

    function step(timestamp) {
      if (!start) start = timestamp;
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      const newValue = initialProgress + progress * progressDiff;

      setProgress(newValue);

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  function simulateLoading() {
    let progress = currentProgress;
    const interval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
      }
      setProgress(progress);
    }, 200);
  }

  // Initialize
  setProgress(initialValue);

  if (animateOnLoad) {
    window.addEventListener("load", () => {
      setTimeout(() => animateProgress(targetValue), animationDelay);
    });
  }

  return {
    setProgress,
    animateProgress,
    simulateLoading,
    getCurrentProgress: () => currentProgress,
  };
}
