// static/js/dropdown.js
document.addEventListener(
  "DOMContentLoaded",
  () => {
    const dropdowns = document.querySelectorAll(".custom-dropdown");

    dropdowns.forEach((dropdown) => {
      const dropdownBtn = dropdown.querySelector(".dropdown-btn");
      const dropdownContent = dropdown.querySelector(".dropdown-content");
      const dropdownSelected = dropdown.querySelector(".dropdown-selected");
      const dropdownItems = dropdown.querySelectorAll(".dropdown-item");
      // Toggle dropdown visibility
      dropdownBtn.addEventListener("click", function (e) {
        e.stopPropagation();

        dropdownBtn.classList.toggle("active");
        if (dropdownBtn.classList.contains("active")) {
          expand(dropdownContent);
        } else {
          collapse(dropdownContent);
        }
      });

      // Handle item selection
      dropdownItems.forEach((item) => {
        item.addEventListener("click", function (e) {
          e.preventDefault();
          const value = item.getAttribute("data-value");
          const label = item.getAttribute("data-label");
          const text = item.textContent;
          const url = item.getAttribute("url");
          const name = item.getAttribute("name-value");
          const data = { [name]: value };
          const method = item.getAttribute("method");
          const redirect = item.getAttribute("redirect");

          if (redirect) {
            window.location.href = url;
            return;
          }

          fetch(url, {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: method === "POST" ? JSON.stringify(data) : null,
          });

          dropdownSelected.textContent = label;
          dropdownSelected.setAttribute("data-value", value);
          dropdownBtn.classList.remove("active");

          collapse(dropdownContent);
        });
      });
    });

    // Close all dropdowns when clicking outside
    document.addEventListener("click", function (e) {
      document.querySelectorAll(".dropdown").forEach((dropdown) => {
        const dropdownBtn = dropdown.querySelector(".dropdown-btn");
        const dropdownContent = dropdown.querySelector(".dropdown-content");
        if (
          !dropdownBtn.contains(e.target) &&
          !dropdownContent.contains(e.target)
        ) {
          dropdownBtn.classList.remove("active");

          collapse(dropdownContent);
        }
      });
    });
  },
  { once: true }
);

function expand(element) {
  // Show the element first
  element.style.display = "block";

  // Get the full height of the content
  const fullHeight = element.scrollHeight + "px";

  // Create and play the expand animation
  element.animate(
    { maxHeight: ["0", fullHeight] },
    { duration: 300, fill: "forwards" }
  );
}

function collapse(element) {
  // Get the current height
  const currentHeight = element.scrollHeight + "px";

  // Create and play the collapse animation
  const animation = element.animate(
    { maxHeight: [currentHeight, "0"] },
    { duration: 300, fill: "forwards" }
  );

  // Hide the element after animation completes
  animation.onfinish = () => {
    element.style.display = "none";
  };
}
