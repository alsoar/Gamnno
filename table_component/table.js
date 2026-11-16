"use strict";

(() => {
  const el = id => document.getElementById(id);
  const slots = new Map();
  let model = null, selected = null, alternatives = [], submitted = false;
  let keyboardMove = false, lastHeight = -1;

  function send(type, fields) {
    window.parent.postMessage({isStreamlitMessage: true, type, ...fields}, "*");
  }

  function reportHeight() {
    const height = Math.ceil(el("component").getBoundingClientRect().height);
    if (height === lastHeight) return;
    lastHeight = height;
    send("streamlit:setFrameHeight", {height});
  }

  function location(slot) {
    if (slot === 0) return "Bar";
    if (slot === 1) return "Brown off";
    if (slot === 26) return "White off";
    return "Point " + (slot - 2);
  }

  function addPoint(slot, row) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "point";
    button.dataset.slot = slot;
    button.dataset.point = slot - 2;
    const label = document.createElement("span");
    label.className = "point-label";
    label.textContent = slot - 2;
    const stack = document.createElement("span");
    stack.className = "stack";
    button.append(label, stack);
    row.append(button);
    slots.set(slot, button);
  }

  function drawPieces(slot, button) {
    const [white, brown] = model.board[slot];
    const stamp = white + "," + brown;
    if (button.dataset.counts === stamp) return;
    button.dataset.counts = stamp;
    if (slot === 1 || slot === 26) {
      button.querySelector(".tray-count").textContent = white + brown;
      button.classList.toggle("occupied", white + brown > 0);
      return;
    }
    const container = button.querySelector(slot === 0 ? ".pieces" : ".stack");
    const limit = slot === 0 ? 1 : 5;
    container.replaceChildren();
    for (const [count, color] of [[white, "white"], [brown, "brown"]]) {
      const visible = Math.min(limit, count);
      for (let n = 1; n <= visible; n++) {
        const chip = document.createElement("span");
        chip.className = "checker " + color;
        if (n === visible && count > limit) chip.textContent = count;
        container.append(chip);
      }
    }
  }

  function clearSelection() {
    selected = null;
    alternatives = [];
  }

  function submit(move, byKeyboard) {
    if (!model.interactive || submitted) return;
    submitted = true;
    keyboardMove = byKeyboard;
    clearSelection();
    paint();
    send("streamlit:setComponentValue", {dataType: "json", value: {position: model.position, move}});
  }

  function choose(slot, byKeyboard) {
    if (!model.interactive || submitted) return;
    const fromHere = model.options.filter(m => m.source === selected && m.target === slot);
    if (fromHere.length === 1) {
      submit(fromHere[0], byKeyboard);
      return;
    }
    if (fromHere.length > 1) {
      alternatives = fromHere.sort((a, b) => a.die - b.die);
      paint();
      if (byKeyboard) el("choices").firstElementChild.focus();
      return;
    }
    const selectable = model.options.some(m => m.source === slot);
    selected = selectable && selected !== slot ? slot : null;
    alternatives = [];
    paint();
    if (byKeyboard && selected !== null) {
      el("component").querySelector(".destination")?.focus();
    }
  }

  function paint() {
    const interactive = model.interactive && !submitted;
    const sources = new Set(model.options.map(m => m.source));
    const targets = new Set(model.options.filter(m => m.source === selected).map(m => m.target));
    for (const [slot, button] of slots) {
      drawPieces(slot, button);
      const source = interactive && sources.has(slot);
      const dest = interactive && targets.has(slot);
      button.disabled = !source && !dest;
      button.classList.toggle("selectable", source);
      button.classList.toggle("destination", dest);
      button.classList.toggle("selected", interactive && selected === slot);
      button.classList.toggle("last-source", model.last_move?.source === slot);
      button.classList.toggle("last-target", model.last_move?.target === slot);
      button.setAttribute("aria-pressed", String(interactive && selected === slot));
      const [white, brown] = model.board[slot];
      const detail = white + " White, " + brown + " Brown";
      const hint = dest ? ", legal destination" : source ? ", selectable" : "";
      button.setAttribute("aria-label", location(slot) + ", " + detail + hint);
    }

    el("interaction").hidden = !model.interactive || (!model.options.length && !submitted);
    let text = "Select a highlighted checker.";
    if (submitted) text = "Applying move…";
    else if (alternatives.length) text = "Choose a die to bear off.";
    else if (selected !== null) text = location(selected) + " selected. Choose a marked destination.";
    else if (sources.size === 1 && sources.has(0)) text = "Select the bar to re-enter a checker.";
    if (el("status").textContent !== text) el("status").textContent = text;
    const choices = el("choices");
    const focusedDie = choices.contains(document.activeElement) ? document.activeElement.dataset.die : null;
    choices.replaceChildren();
    for (const move of alternatives) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.die = move.die;
      button.textContent = "Use " + move.die;
      button.addEventListener("click", event => submit(move, event.detail === 0));
      choices.append(button);
      if (button.dataset.die === focusedDie) button.focus({preventScroll: true});
    }
  }

  for (let i = 13; i >= 2; i--) addPoint(i, el("top-row"));
  for (let i = 14; i <= 25; i++) addPoint(i, el("bottom-row"));
  for (const id of ["bar", "brown-off", "white-off"]) {
    const button = el(id);
    slots.set(Number(button.dataset.slot), button);
  }
  for (const [slot, button] of slots) {
    button.addEventListener("click", event => choose(slot, event.detail === 0));
  }
  document.addEventListener("keydown", event => {
    if (event.key !== "Escape" || selected === null) return;
    const button = slots.get(selected);
    clearSelection();
    paint();
    button.focus();
  });

  window.addEventListener("message", event => {
    if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
    const args = event.data.args;
    if (!args || !Array.isArray(args.board) || args.board.length !== 27) return;
    const changed = !model || model.position !== args.position;
    const fromKeyboard = changed && keyboardMove;
    model = {...args, options: args.options || [], interactive: !!args.interactive && !event.data.disabled};
    if (changed) {
      submitted = false;
      keyboardMove = false;
      clearSelection();
    } else if (!model.interactive || !model.options.some(m => m.source === selected)) {
      clearSelection();
    }
    if (event.data.theme?.textColor) document.documentElement.style.setProperty("--text", event.data.theme.textColor);
    el("component").hidden = false;
    paint();
    if (fromKeyboard && model.interactive) el("component").querySelector(".selectable")?.focus({preventScroll: true});
    reportHeight();
  });

  new ResizeObserver(reportHeight).observe(el("component"));
  send("streamlit:componentReady", {apiVersion: 1});
})();
