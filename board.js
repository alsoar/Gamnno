"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const names = ["White", "Brown"];
  const bar = 0, brownOff = 1, whiteOff = 26;
  const slots = new Map();
  const controls = Object.fromEntries(
    ["mode", "pause", "step", "restart", "undo", "commit", "roll", "pass"].map(id => [id, byId(id)])
  );
  let phase = "loading"; // active turn, ready to roll, or over
  let board = null, player = 0, roll = [], original = null;
  let prefix = [], options = [], done = false, botPath = null;
  let source = null, alternatives = [], notes = [], history = "", lastMove = null;
  let pending = false, botPending = false, paused = false, error = "", timer = null, version = 0;

  function label(index) {
    if (index === bar) return "bar";
    if (index === brownOff || index === whiteOff) return "off";
    return String(index - 2);
  }

  function controller() {
    return controls.mode.value.split("-")[player];
  }

  function isBot() {
    return controller() !== "human";
  }

  function sameMove(a, b) {
    return a.checker === b.checker && a.source === b.source && a.target === b.target && a.die === b.die;
  }

  function clearSelection() {
    source = null;
    alternatives = [];
  }

  function chooseSlot(index) {
    if (pending || phase !== "active" || done || isBot()) return;
    if (source !== null) {
      const candidates = options.filter(m => m.source === source && m.target === index);
      if (candidates.length === 1) {
        playMove(candidates[0]);
        return;
      }
      if (candidates.length > 1) {
        alternatives = candidates;
        paint();
        byId("choices").querySelector("button").focus();
        return;
      }
    }
    const selectable = options.some(m => m.source === index);
    source = selectable && source !== index ? index : null;
    alternatives = [];
    paint();
  }

  function setupBoard() {
    for (const [row, points] of [
      ["top-row", Array.from({length: 12}, (_, i) => 13 - i)],
      ["bottom-row", Array.from({length: 12}, (_, i) => 14 + i)],
    ]) {
      for (const index of points) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "point";
        button.dataset.point = index;
        const number = document.createElement("span");
        number.className = "point-label";
        number.textContent = label(index);
        const pieces = document.createElement("span");
        pieces.className = "stack";
        button.append(number, pieces);
        button.addEventListener("click", () => chooseSlot(index));
        byId(row).append(button);
        slots.set(index, {button, pieces});
      }
    }
    for (const [index, id] of [[bar, "bar"], [brownOff, "brown-off"], [whiteOff, "white-off"]]) {
      const button = byId(id);
      button.addEventListener("click", () => chooseSlot(index));
      slots.set(index, {button, pieces: button.querySelector(".pieces")});
    }
  }

  function putCheckers(container, color, count, limit) {
    const visible = Math.min(count, limit);
    for (let i = 1; i <= visible; i++) {
      const chip = document.createElement("span");
      chip.className = "checker " + names[color].toLowerCase();
      if (i === visible && count > limit) chip.textContent = count;
      container.append(chip);
    }
  }

  function paintBoard(rawPlayer = phase === "over" ? 1 - player : player) {
    const interactive = phase === "active" && !pending && !done && !isBot();
    const sources = new Set(options.map(m => m.source));
    const destinations = new Set(options.filter(m => m.source === source).map(m => m.target));
    for (const [index, {button, pieces}] of slots) {
      const pair = board ? board[index] : [0, 0];
      const off = index === whiteOff || index === brownOff;
      const title = off ? names[index === whiteOff ? 0 : 1] + " bear-off" :
        index === bar ? "Bar" : "Point " + label(index);
      button.setAttribute("aria-label", title + ", " + pair[0] + " White, " + pair[1] + " Brown");
      button.classList.toggle("selectable", interactive && sources.has(index));
      button.classList.toggle("selected", interactive && source === index);
      button.classList.toggle("destination", interactive && destinations.has(index));
      button.classList.toggle("last-move", !!lastMove && lastMove.source === index);
      button.classList.toggle("last-target", !!lastMove && lastMove.target === index);
      button.setAttribute("aria-pressed", String(interactive && source === index));
      button.disabled = !(interactive && (sources.has(index) || destinations.has(index)));
      const stamp = pair.join(",");
      if (pieces.dataset.counts === stamp) continue;
      pieces.dataset.counts = stamp;
      pieces.replaceChildren();
      if (off && !pair[0] && !pair[1]) pieces.textContent = "0";
      putCheckers(pieces, 0, pair[0], off || index === bar ? 1 : 5);
      putCheckers(pieces, 1, pair[1], off || index === bar ? 1 : 5);
    }
    const raw = byId("board-state");
    if (!board) {
      raw.textContent = "";
      return;
    }
    // The existing browser protocol supplies count pairs. The engine stores
    // one signed count per physical point, with the bars alongside them.
    const points = board.slice(2, 26).map(([white, brown]) => white - brown);
    const groups = [];
    for (let i = 0; i < points.length; i += 6) groups.push(points.slice(i, i + 6).join(", "));
    raw.textContent = [
      "board = [", "  " + groups.join(",\n  "), "]",
      "white_bar = " + board[bar][0],
      "brown_bar = " + board[bar][1],
      "player = " + (rawPlayer === 0 ? "+1" : "-1"),
      "white_off = " + board[whiteOff][0] + "  (inferred)",
      "brown_off = " + board[brownOff][1] + "  (inferred)",
    ].join("\n");
  }

  function paintDice() {
    const holder = byId("dice");
    const values = roll.length === 2 && roll[0] === roll[1] ? Array(4).fill(roll[0]) : roll;
    const spent = prefix.map(m => m.die);
    holder.replaceChildren();
    for (const die of values) {
      const item = document.createElement("span");
      item.className = "die";
      item.dataset.value = die;
      item.textContent = die;
      const i = spent.indexOf(die);
      if (i !== -1) {
        item.classList.add("used");
        spent.splice(i, 1);
      }
      item.setAttribute("aria-label", "Die " + die + (i === -1 ? ", available" : ", played"));
      holder.append(item);
    }
  }

  function status() {
    if (error) return error;
    if (phase === "loading") return "Setting up the board…";
    if (phase === "over") return names[player] + " wins by bearing off a checker.";
    if (phase === "ready") return isBot() ? (paused ? "Paused before the roll." : "Rolling next…") : "Roll for the next turn.";
    if (done && prefix.length === 0) return "No legal moves on this roll. Pass the turn.";
    if (done) return "All playable dice used. Finish the turn.";
    if (isBot()) {
      if (paused) return "Paused. Step to see the next move.";
      return controller() === "heuristic" ? "Playing the heuristic’s chosen turn." : "Playing a random legal turn.";
    }
    if (alternatives.length) return "Choose which die to use.";
    if (source !== null) return "Choose a highlighted destination.";
    if (board[bar][player]) return "Select the bar to re-enter a checker.";
    return "Select a checker to move.";
  }

  function paint() {
    paintBoard();
    paintDice();
    byId("turn-label").textContent = phase === "over" ? "Winner" : "Current turn";
    byId("player").textContent = names[player];
    byId("history").textContent = history;
    const message = status();
    if (byId("status").textContent !== message) byId("status").textContent = message;
    const human = !isBot(), active = phase === "active", watching = controls.mode.value !== "human-human";
    controls.pause.hidden = !watching;
    controls.pause.disabled = phase === "loading" || phase === "over";
    controls.pause.textContent = paused ? "Resume" : "Pause";
    controls.pause.setAttribute("aria-pressed", String(paused));
    controls.step.hidden = !watching || !paused;
    controls.step.disabled = pending || human || (!active && phase !== "ready");
    controls.roll.hidden = !human || phase !== "ready";
    controls.roll.disabled = pending;
    controls.undo.hidden = !human || !active;
    controls.undo.disabled = pending || !prefix.length;
    const passing = done && prefix.length === 0;
    controls.commit.hidden = !human || !active || passing;
    controls.commit.disabled = pending || !done;
    controls.pass.hidden = !human || !active || !passing;
    controls.pass.disabled = pending;
    const choices = byId("choices");
    choices.replaceChildren();
    for (const move of alternatives) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Use " + move.die;
      button.disabled = pending;
      button.addEventListener("click", () => playMove(move));
      choices.append(button);
    }
  }

  async function post(path, payload) {
    const response = await fetch("/api/" + path, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.error || "The server could not process this turn.");
    }
    return response.json();
  }

  function scheduleBot() {
    clearTimeout(timer);
    if (pending || paused || !isBot() || (phase !== "active" && phase !== "ready")) return;
    timer = setTimeout(botStep, 900);
  }

  // Only the latest transaction may update the board.
  async function transact(action, byBot = false) {
    const request = ++version;
    clearTimeout(timer);
    pending = true;
    botPending = byBot;
    error = "";
    paint();
    try {
      const update = await action(() => request === version);
      if (request === version) update();
    } catch (err) {
      if (request === version) {
        error = "Could not update the board. " + err.message;
        paused = true;
      }
    } finally {
      if (request === version) {
        pending = false;
        botPending = false;
        paint();
        scheduleBot();
      }
    }
  }

  function inspect(start, moves) {
    return post("choices", {board: start.board, player: start.player, dice: start.dice, prefix: moves});
  }

  function showChoices(result) {
    board = result.board;
    options = result.options;
    done = result.done;
    if (board[whiteOff][0] || board[brownOff][1]) phase = "over";
    clearSelection();
  }

  function begin(opening) {
    const payload = opening ? {} : {board, player};
    if (opening) {
      phase = "loading";
      board = null;
      original = null;
      player = 0;
      roll = [];
      prefix = [];
      options = [];
      history = "";
      lastMove = null;
      clearSelection();
    }
    return transact(async () => {
      const start = await post(opening ? "new" : "roll", payload);
      const choice = await inspect(start, []);
      return () => {
        phase = "active";
        original = start;
        player = start.player;
        roll = start.dice;
        prefix = [];
        notes = [];
        botPath = null;
        if (opening) history = names[player] + " won the opening roll.";
        showChoices(choice);
      };
    }, !opening && isBot());
  }

  async function prepareMove(move, plan) {
    const path = [...prefix, move];
    let note = names[move.checker] + " " + label(move.source) + " → " + label(move.target) + " (die " + move.die + ")";
    if (board[move.target][1 - move.checker]) note += " · " + names[1 - move.checker] + " to the bar";
    const result = await inspect(original, path);
    return () => {
      prefix = path;
      botPath = plan;
      notes.push(note);
      history = note;
      lastMove = move;
      showChoices(result);
    };
  }

  function playMove(move) {
    if (pending || phase !== "active" || done) return;
    return transact(() => prepareMove(move, null));
  }

  function undo() {
    if (pending || phase !== "active" || isBot() || !prefix.length) return;
    const path = prefix.slice(0, -1);
    return transact(async () => {
      const result = await inspect(original, path);
      return () => {
        notes.pop();
        history = notes[notes.length - 1] || "Back at the start of this turn.";
        prefix = path;
        lastMove = path[path.length - 1] || null;
        botPath = null;
        showChoices(result);
      };
    });
  }

  function finishTurn() {
    if (pending || phase !== "active" || !done) return;
    if (!prefix.length) history = names[player] + " passed.";
    phase = "ready";
    player = 1 - player;
    prefix = [];
    roll = [];
    options = [];
    clearSelection();
    paint();
    scheduleBot();
  }

  function randomChoice(items) {
    return items[Math.floor(Math.random() * items.length)];
  }

  function botStep() {
    if (pending || !isBot()) return;
    if (phase === "ready") return begin(false);
    if (phase !== "active") return;
    if (done) return finishTurn();
    return transact(async current => {
      let plan = botPath;
      if (!plan && controller() === "heuristic") {
        const selected = await post("heuristic", {
          board: original.board, player: original.player, dice: original.dice, prefix,
        });
        if (!current()) return;
        plan = selected.moves;
      } else if (!plan) {
        // Random play samples distinct end boards uniformly from a fresh turn.
        const matching = original.turns.filter(t => prefix.every((m, i) => t.moves[i] && sameMove(m, t.moves[i])));
        if (matching.length) plan = randomChoice(matching).moves;
      }
      return prepareMove(plan ? plan[prefix.length] : randomChoice(options), plan);
    }, true);
  }

  function cancelBotRequest() {
    if (botPending) {
      ++version;
      pending = false;
      botPending = false;
    }
  }

  function describeRecordedMove(move, actor, before) {
    const who = names[actor];
    const start = label(move.source), end = label(move.target);
    let text;
    if (move.checker !== actor) {
      text = who + " moved " + names[move.checker] + " backward from " + start + " to " + end;
    } else if (move.source === bar) {
      text = who + " re-entered from the bar on point " + end;
    } else if (move.target === whiteOff || move.target === brownOff) {
      text = who + " bore off from point " + start;
    } else {
      text = who + " moved from " + start + " to " + end;
    }
    text += " with a " + move.die;
    const hit = 1 - move.checker;
    if (before[move.target][hit]) {
      text += hit === actor
        ? ", sending " + who + "’s own blot to the bar"
        : ", capturing " + names[hit] + "’s blot and sending it to the bar";
    }
    return text + ".";
  }

  async function startReplay(seed, embedded) {
    phase = "replay";
    document.body.dataset.mode = "replay";
    byId("live-controls").hidden = true;
    byId("replay-controls").hidden = false;
    const previous = byId("replay-prev"), next = byId("replay-next"), seek = byId("replay-seek");
    previous.disabled = next.disabled = seek.disabled = true;
    byId("status").textContent = "Loading the recording…";
    try {
      let recording;
      if (embedded) {
        recording = JSON.parse(embedded.textContent);
      } else {
        if (!/^\d+$/.test(seed)) throw new Error("Unknown game seed.");
        const response = await fetch("/replay-" + seed + ".json");
        if (!response.ok) throw new Error("No recording found for seed " + seed + ".");
        recording = await response.json();
      }
      let currentBoard = recording.opening;
      const frames = [{turnIndex: 0, played: 0, board: currentBoard}];
      recording.turns.forEach((turn, turnIndex) => {
        if (!turn.moves.length) frames.push({turnIndex, played: 0, board: currentBoard});
        turn.boards.forEach((after, i) => {
          currentBoard = after;
          frames.push({turnIndex, played: i + 1, board: currentBoard});
        });
      });
      const last = frames.length - 1;
      const unit = recording.turns.some(turn => !turn.moves.length) ? "Step" : "Move";
      let cursor = 0;
      function show(index) {
        cursor = Math.max(0, Math.min(last, index));
        const frame = frames[cursor], turn = recording.turns[frame.turnIndex];
        const ended = cursor > 0 && frame.played === turn.moves.length;
        board = frame.board;
        player = turn.player;
        roll = turn.roll;
        prefix = turn.moves.slice(0, frame.played);
        lastMove = frame.played ? prefix[frame.played - 1] : null;
        paintBoard(ended ? 1 - player : player);
        paintDice();
        byId("turn-label").textContent = "Turn " + (frame.turnIndex + 1) + " of " + recording.turns.length;
        byId("player").textContent = names[player];
        if (cursor === 0) {
          byId("status").textContent = "Opening position";
          byId("history").textContent = names[player] + " won the opening roll.";
        } else {
          byId("history").textContent = lastMove
            ? describeRecordedMove(lastMove, player, frames[cursor - 1].board)
            : names[player] + " passed. Neither die could be played.";
          if (cursor === last) {
            byId("status").textContent = names[recording.winner] + " wins by bearing off a checker.";
          } else {
            byId("status").textContent = lastMove ? "Move " + frame.played + " of " + turn.moves.length + " on this roll." : "Forced pass";
          }
        }
        const position = unit + " " + cursor + " of " + last;
        byId("replay-position").textContent = position;
        seek.value = cursor;
        seek.setAttribute("aria-valuetext", position + ", turn " + (frame.turnIndex + 1));
        previous.disabled = cursor === 0;
        next.disabled = cursor === last;
      }
      seek.min = 0;
      seek.max = last;
      seek.step = 1;
      seek.disabled = false;
      previous.addEventListener("click", () => show(cursor - 1));
      next.addEventListener("click", () => show(cursor + 1));
      seek.addEventListener("input", () => show(Number(seek.value)));
      document.addEventListener("keydown", event => {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
        if (event.target instanceof Element && event.target.matches("input, select, textarea, [contenteditable=true]")) return;
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault();
          show(cursor + (event.key === "ArrowRight" ? 1 : -1));
        }
      });
      show(0);
    } catch (err) {
      byId("status").textContent = "Could not open the replay. " + err.message;
    }
  }

  setupBoard();
  const embedded = byId("recording");
  const replaySeed = new URLSearchParams(location.search).get("replay");
  if (embedded || replaySeed !== null) {
    byId("replay-link").setAttribute("aria-current", "page");
    byId("live-link").removeAttribute("aria-current");
    startReplay(replaySeed, embedded);
    return;
  }
  byId("live-link").setAttribute("aria-current", "page");
  byId("replay-link").removeAttribute("aria-current");

  controls.restart.addEventListener("click", () => begin(true));
  controls.roll.addEventListener("click", () => begin(false));
  controls.commit.addEventListener("click", finishTurn);
  controls.pass.addEventListener("click", finishTurn);
  controls.undo.addEventListener("click", undo);
  controls.step.addEventListener("click", botStep);
  controls.pause.addEventListener("click", () => {
    paused = !paused;
    if (paused) cancelBotRequest();
    paint();
    scheduleBot();
  });
  controls.mode.addEventListener("change", () => {
    clearTimeout(timer);
    cancelBotRequest();
    clearSelection();
    botPath = null;
    paused = false;
    paint();
    scheduleBot();
  });
  begin(true);
})();
