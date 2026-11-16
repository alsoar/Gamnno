"""History browser and step-by-step replay for the Streamlit app."""

from typing import NamedTuple


COLORS = ("White", "Brown")
ROLES = {"human": "Human", "baseline": "Baseline bot"}
TYPES = {"human-bot": "Human vs bot", "bot-bot": "Bot vs bot", "human-human": "Human vs human"}


class Frame(NamedTuple):
    turn: int
    used: int
    before: list
    board: list
    move: object


def replay_frames(record):
    """Reference saved boards directly, including an unchanged frame for each forced pass."""
    current = record["opening"]
    frames = [Frame(0, 0, current, current, None)]
    for index, turn in enumerate(record["turns"]):
        if not turn["moves"]:
            frames.append(Frame(index, 0, current, current, None))
        for used, (move, after) in enumerate(zip(turn["moves"], turn["boards"]), start=1):
            frames.append(Frame(index, used, current, after, move))
            current = after
    return frames


def _game_label(summary):
    date = summary["completed_at"]
    when = date[:16].replace("T", " ") + " UTC · " if date else ""
    return (f"{when}{TYPES[summary['game_type']]} · Seed {summary['seed']} · "
            f"{COLORS[summary['winner']]} won · {summary['turn_count']} turns")


def render_history(store):
    import streamlit as st
    from board_view import move_description, render_board

    summaries = store.list_games()
    filters = ["All games", TYPES["human-bot"], TYPES["bot-bot"]]
    if any(s["game_type"] == "human-human" for s in summaries):
        filters.append(TYPES["human-human"])
    left, right = st.columns([1, 3])
    kind = left.selectbox("Game type", filters, key="history_game_type")
    games = {s["id"]: s for s in summaries if kind == "All games" or TYPES[s["game_type"]] == kind}
    if not games:
        st.info("No games match this filter. You can save your game to history after it ends.")
        return
    ids = list(games)
    default = ids[0]
    if all(s["completed_at"] is None for s in games.values()) and "baseline-2095" in games:
        default = "baseline-2095"
    game_id = right.selectbox("Recorded game", ids, index=ids.index(default),
                              format_func=lambda key: _game_label(games[key]), key="history_game")
    summary = games[game_id]
    selected = st.session_state.get("_history_replay")
    if selected is None or selected[0] != game_id:
        record = store.load(game_id)
        selected = (game_id, record, replay_frames(record))
        st.session_state["_history_replay"] = selected
        st.session_state["history_seek"] = 0
    _, record, frames = selected
    last = len(frames) - 1
    position = st.session_state.get("history_seek", 0)
    frame = frames[position]
    turn = record["turns"][frame.turn]
    player = COLORS[turn["player"]]
    label = "Move" if last == summary["move_count"] else "Step"
    roles = summary["players"]

    st.markdown(f"**{COLORS[summary['winner']]} won** in {summary['turn_count']} player turns. "
                f"White played as **{ROLES[roles['white']]}** and Brown as **{ROLES[roles['brown']]}**.")
    dice = " + ".join(str(die) for die in turn["roll"])
    st.caption(f"Turn {frame.turn + 1} of {summary['turn_count']} · {player} · Roll {dice}")
    render_board(frame.board, frame.move)
    if position == 0:
        description = f"Opening position. {player} won the opening roll."
    elif frame.move is None:
        description = f"{player} passed. Neither die could be played."
    else:
        description = move_description(frame.move, turn["player"], frame.before)
    st.write(description)

    def step(delta):
        st.session_state["history_seek"] = max(0, min(last, st.session_state["history_seek"] + delta))

    previous, following = st.columns(2)
    previous.button("Previous move", key="history_previous", disabled=position == 0,
                    on_click=step, args=(-1,), use_container_width=True)
    following.button("Next move", key="history_next", disabled=position == last,
                     on_click=step, args=(1,), use_container_width=True)
    st.slider(f"{label} in recording", min_value=0, max_value=last, step=1, key="history_seek")
    st.caption(f"{label} {position} of {last}")
