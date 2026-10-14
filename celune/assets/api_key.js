const form = document.querySelector("#key-form");
    const input = document.querySelector("#api-key");
    const button = document.querySelector("#submit");
    const error = document.querySelector("#error");
    const invalidMessage = form.dataset.invalidMessage;
    const errorMessage = form.dataset.errorMessage;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      error.textContent = "";
      try {
        const response = await fetch("/ui/key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: input.value }),
        });
        if (response.ok) {
          window.location.replace("/ui");
          return;
        }
        error.textContent = response.status === 401 ? invalidMessage : errorMessage;
        return;
      } catch {
        error.textContent = errorMessage;
      } finally {
        button.disabled = false;
      }
    });
