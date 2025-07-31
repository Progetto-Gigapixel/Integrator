document.addEventListener("DOMContentLoaded", function () {
  const buttons = document.querySelectorAll(".light-btn");
  //console.log(buttons);
  if (buttons.length == 0) {
    return;
  }

  buttons.forEach((button) => {
    button.addEventListener("click", function (e) {
      e.preventDefault();
      //toggle on off
      const id = button.getAttribute("data-value");
      if (this.classList.contains("on")) {
        this.classList.remove("on");
        //call external capturer api
        fetch(`/api/capturer/lights/${id}/off`, {});
        this.classList.add("off");
      } else {
        this.classList.remove("off");
        fetch(`/api/capturer/lights/${id}/on`, {});
        this.classList.add("on");
      }
    });
  });
  const offButton = document.getElementById("off-button");
  offButton.addEventListener("click", async function (e) {
    e.preventDefault();
    fetch(`/api/capturer/lights/allOff`, {}).then((res) => {
      if (res.status == 200) {
        //console.log(res);
        document.querySelectorAll(".light-btn.on").forEach((button) => {
          button.classList.remove("on");
          button.classList.add("off");
        });
      } else {
        //console.log(res);
      }
    });
  });
  const on14Button = document.getElementById("on14-button");
  on14Button.addEventListener("click", function (e) {
    e.preventDefault();
    res = fetch(`/api/capturer/lights/1234on`, {}).then((res) => {
      if (res.status == 200) {
        //console.log(res);
        document.querySelectorAll(".light-btn.off").forEach((button) => {
          const id = button.getAttribute("data-value");
          if ([1, 2, 3, 4].includes(parseInt(id))) {
            button.classList.remove("off");
            button.classList.add("on");
          }
        });
      } else {
        //console.log(res);
      }
    });
  });
});
