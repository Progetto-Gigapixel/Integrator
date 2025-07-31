document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".project-card.default").forEach((card) => {
    const id = card.getAttributeNode("data-id").value;
    card.addEventListener("click", function (e) {
      window.location.href = `/projectpage/${id}`;
    });
  });
  function hideCard(value) {
    document.querySelectorAll(".project-card.default").forEach((card) => {
      const title = card.querySelectorAll(".card-title")[0].textContent;
      if (title.toLowerCase().includes(value.toLowerCase())) {
        card.style.display = "flex";
      } else {
        card.style.display = "none";
      }
    });
  }

  const searchBar = document.getElementById("research-bar");
  searchBar.addEventListener("change", function (e) {
    const value = e.target.value;
    hideCard(value);
  });
  hideCard("");
});
