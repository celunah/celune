(() => {
  if (window.__celuneLogAutoscrollInstalled) {
    return;
  }
  window.__celuneLogAutoscrollInstalled = true;

  const logScrollThreshold = 24;

  function isNearLogBottom(logElement) {
    return logElement.scrollHeight - logElement.scrollTop - logElement.clientHeight
      <= logScrollThreshold;
  }

  function scrollLogToBottom() {
    const logElement = document.querySelector("#celune-log-panel pre");
    if (!logElement) {
      return;
    }
    logElement.scrollTop = logElement.scrollHeight;
  }

  function updateLogFollowState(event) {
    window.__celuneLogAutoscrollFollow = isNearLogBottom(event.currentTarget);
  }

  function installLogObserver() {
    const logElement = document.querySelector("#celune-log-panel pre");
    if (!logElement) {
      return;
    }

    if (window.__celuneLogAutoscrollTarget === logElement) {
      return;
    }

    const previousLogElement = window.__celuneLogAutoscrollTarget;
    if (previousLogElement) {
      window.__celuneLogAutoscrollFollow = isNearLogBottom(previousLogElement);
    } else if (typeof window.__celuneLogAutoscrollFollow !== "boolean") {
      window.__celuneLogAutoscrollFollow = true;
    }

    window.__celuneLogAutoscrollTarget = logElement;
    logElement.addEventListener("scroll", updateLogFollowState, {
      passive: true,
    });
    if (window.__celuneLogAutoscrollFollow) {
      window.requestAnimationFrame(scrollLogToBottom);
    }

    if (window.__celuneLogAutoscrollObserver) {
      window.__celuneLogAutoscrollObserver.disconnect();
    }

    const observer = new MutationObserver(() => {
      if (window.__celuneLogAutoscrollFollow) {
        scrollLogToBottom();
      }
    });
    observer.observe(logElement, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    window.__celuneLogAutoscrollObserver = observer;
  }

  function updateConnectionToastState() {
    const connectionLostTitle = "Connection Lost";
    const connectionLostMessage = "Connection to the server was lost.";
    let hasConnectionLost = false;
    const errorToasts = document.querySelectorAll(
      '[data-testid="toast-body"].error',
    );

    for (const toast of errorToasts) {
      const title = toast.querySelector(
        ".toast-title-row span",
      )?.textContent?.trim();
      const message = toast.querySelector(".toast-message-text")?.textContent;
      const isConnectionLost = Boolean(
        title === connectionLostTitle
        && message?.trim().startsWith(connectionLostMessage),
      );
      toast.classList.toggle("celune-gradio-connection-lost", isConnectionLost);
      hasConnectionLost ||= isConnectionLost;
    }

    document.body.classList.toggle(
      "celune-gradio-connection-lost",
      hasConnectionLost,
    );
  }

  const pageObserver = new MutationObserver(() => {
    installLogObserver();
    updateConnectionToastState();
  });

  function startLogAutoscroll() {
    installLogObserver();
    pageObserver.observe(document.body, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    updateConnectionToastState();
    window.setInterval(installLogObserver, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startLogAutoscroll, {
      once: true,
    });
  } else {
    startLogAutoscroll();
  }
})();

(() => {
  function handleRecordingShortcut(event) {
    if (
      !event.altKey
      || event.ctrlKey
      || event.metaKey
      || event.key.toLowerCase() !== "r"
    ) {
      return;
    }

    const recordButton = document.querySelector(
      "#celune-record-hotkey button, button#celune-record-hotkey",
    );
    if (!recordButton || recordButton.disabled) {
      return;
    }

    event.preventDefault();
    recordButton.click();
  }

  document.addEventListener("keydown", handleRecordingShortcut);
})();

(() => {
  let overlay = null;
  let activeConsentKey = null;

  function removeOverlay() {
    if (overlay) {
      overlay.remove();
      overlay = null;
      activeConsentKey = null;
    }
  }

  function updateFooterStatus(status) {
    const footer = document.querySelector("#celune-status .footer-block");
    if (!footer) {
      return;
    }
    const textNode = [...footer.childNodes].find(
      (node) => node.nodeType === Node.TEXT_NODE,
    );
    if (textNode) {
      textNode.textContent = status;
    } else {
      footer.insertBefore(document.createTextNode(status), footer.firstChild);
    }
  }

  function makeText(tag, value, className) {
    const element = document.createElement(tag);
    element.textContent = value;
    if (className) {
      element.className = className;
    }
    return element;
  }

  function showConsentNotice(message) {
    const notice = makeText("div", message);
    notice.setAttribute("role", "status");
    notice.className = "celune-consent-notice";
    document.body.append(notice);
    window.setTimeout(() => notice.remove(), 7000);
  }

  async function submitDecision(model, accepted, accept, decline, failure, input) {
    accept.disabled = true;
    decline.disabled = true;
    failure.hidden = true;
    try {
      const response = await fetch("/ui/models/consent", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: model.model_id,
          revision: model.revision,
          accepted,
        }),
      });
      if (response.status === 403 && !accepted) {
        removeOverlay();
        showConsentNotice(model.declined_message);
        return;
      }
      if (!response.ok) {
        throw new Error();
      }
      removeOverlay();
      showConsentNotice(model.accepted_message);
    } catch {
      failure.hidden = false;
      accept.disabled = input.value !== model.consent_phrase;
      decline.disabled = false;
    }
  }

  function showConsent(model) {
    const key = `${model.model_id}@${model.revision}`;
    if (activeConsentKey === key) {
      return;
    }
    removeOverlay();
    activeConsentKey = key;
    overlay = document.createElement("div");
    overlay.id = "celune-model-consent";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    const panel = document.createElement("section");
    panel.setAttribute("aria-labelledby", "celune-model-consent-title");
    panel.append(makeText("h2", model.display_name, "title"));
    panel.querySelector("h2").id = "celune-model-consent-title";
    panel.append(makeText("p", model.prompt));
    const source = document.createElement("a");
    source.href = model.source_url;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    source.textContent = model.source_url;
    panel.append(source);
    if (model.repository_url && model.repository_url !== model.source_url) {
      const repository = document.createElement("a");
      repository.href = model.repository_url;
      repository.target = "_blank";
      repository.rel = "noopener noreferrer";
      repository.textContent = model.repository_url;
      panel.append(repository);
    }
    panel.append(makeText("p", model.instruction));
    for (const detail of model.details) {
      panel.append(makeText("p", detail));
    }
    const input = document.createElement("input");
    input.type = "text";
    input.autocomplete = "off";
    input.placeholder = model.entry_hint;
    input.setAttribute("aria-label", model.entry_hint);
    panel.append(input);
    const failure = makeText("p", model.submit_failed_message);
    failure.hidden = true;
    failure.setAttribute("role", "alert");
    panel.append(failure);
    const actions = document.createElement("div");
    actions.className = "actions";
    const accept = makeText("button", model.accept_label);
    accept.type = "button";
    accept.dataset.role = "accept";
    accept.disabled = true;
    const decline = makeText("button", model.decline_label);
    decline.type = "button";
    decline.dataset.role = "decline";
    input.addEventListener("input", () => {
      accept.disabled = input.value !== model.consent_phrase;
    });
    accept.addEventListener("click", () => submitDecision(model, true, accept, decline, failure, input));
    decline.addEventListener("click", () => submitDecision(model, false, accept, decline, failure, input));
    actions.append(accept, decline);
    panel.append(actions);
    overlay.append(panel);
    document.body.append(overlay);
    input.focus();
  }

  async function pollConsent() {
    try {
      const response = await fetch("/ui/models/pending", {
        credentials: "same-origin",
        cache: "no-store",
      });
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      updateFooterStatus(payload.status);
      const model = payload.pending?.[0];
      if (model) {
        showConsent(model);
      } else {
        removeOverlay();
      }
    } catch {
      return;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", pollConsent, { once: true });
  } else {
    pollConsent();
  }
  window.setInterval(pollConsent, 1500);
})();
