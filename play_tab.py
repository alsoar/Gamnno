"""Human play against the deterministic baseline, with one match per browser session."""

import sqlite3
import streamlit as st

import backgammon as game
import board_view
from match import Match
from visualizer import _browser_color


def _name(checker):
    return "White" if checker == game.WHITE else "Brown"


def _point(point):
    if point == game.BAR:
        return "bar"
    if point == game.OFF:
        return "off"
    return str(point)


def _new_game():
    side = game.WHITE if st.session_state.play_new_color == "White" else game.BROWN
    st.session_state.play_match = Match.new(human_color=side)
    for key in ("play_source", "play_saved", "play_save_error"):
        st.session_state.pop(key, None)


def _save_game(match, store):
    try:
        store.save(match.recording())
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.session_state.play_save_error = f"Could not save this game. {exc}"
    else:
        st.session_state.play_saved = match.id
        st.session_state.history_game_type = "Human vs bot"
        st.session_state.history_game = match.id
        st.session_state.pop("play_save_error", None)


def _roll_text(match):
    first, second = match.dice
    available = list(match.dice) if first != second else [first] * 4
    for move in match.prefix:
        available.remove(move.die)
    dice = " / ".join(str(die) for die in available)
    st.markdown(f"**Roll {first} · {second}**")
    if available:
        st.caption(f"Dice remaining · {dice}")
    else:
        st.caption("All dice played.")


def _move_label(move, preview, actor):
    if move.source == game.BAR:
        action = f"Enter on {_point(move.target)}"
    elif move.target == game.OFF:
        action = "Bear off"
    else:
        action = f"Move to {_point(move.target)}"
    label = f"{action} · die {move.die}"
    if move.target != game.OFF and int(preview.board[move.target]) == -move.checker:
        hit = "self-hit" if move.checker != actor else "capture"
        label += f" · {hit}"
    return label


def _human_controls(match):
    preview, options, done = match.preview()
    st.markdown(f"**Your turn · {_name(match.human_color)}**")
    _roll_text(match)
    if options:
        groups = {}
        for move in options:
            groups.setdefault((move.checker, move.source), []).append(move)

        def source_label(source):
            checker, point = source
            location = "on the bar" if point == game.BAR else f"on {point}"
            direction = "forward" if checker == match.human_color else "backward"
            return f"{_name(checker)} {location} · {direction}"

        sources = sorted(groups, key=lambda s: (s[0] != match.human_color, s[1]))
        if st.session_state.get("play_source") not in sources:
            st.session_state.pop("play_source", None)
        selected = st.selectbox(
            "Checker to move", sources, format_func=source_label, key="play_source",
            help="Use the board or choose a checker here. White advances to higher points and Brown to lower points.",
        )
        for move in sorted(groups[selected], key=lambda m: m.die):
            st.button(
                _move_label(move, preview, match.human_color),
                key=f"play_move_{move.checker}_{move.source}_{move.target}_{move.die}",
                on_click=match.choose, args=(move,), width="stretch",
            )
    elif not match.prefix:
        st.info("Neither die can be played. Pass to the baseline.")

    if match.prefix:
        steps = [f"{_name(m.checker)} {_point(m.source)} → {_point(m.target)} ({m.die})"
                 for m in match.prefix]
        st.caption("Played · " + "  /  ".join(steps))

    st.button(
        "Pass turn" if done and not match.prefix else "End turn", key="play_end",
        on_click=match.finish_human_turn, disabled=not done, type="primary", width="stretch",
    )
    if match.prefix:
        st.button("Undo last move", key="play_undo", on_click=match.undo, width="stretch")


def _last_bot_turn(match):
    bot = _browser_color(-match.human_color)
    # After a human wins, their winning turn follows the baseline's latest turn.
    index = next((i for i in range(len(match.turns) - 1, -1, -1)
                  if match.turns[i]["player"] == bot), None)
    if index is None:
        return
    turn = match.turns[index]
    dice = " · ".join(map(str, turn["roll"]))
    with st.expander(f"Baseline’s last turn · {_name(-match.human_color)} rolled {dice}", expanded=True):
        if not turn["moves"]:
            st.write("The baseline passed.")
            return
        before = next((match.turns[i]["boards"][-1] for i in range(index - 1, -1, -1)
                       if match.turns[i]["boards"]), None)
        if before is None:
            before = game.to_legacy(game.starting_state())
        for move, after in zip(turn["moves"], turn["boards"]):
            st.write(board_view.move_description(move, bot, before))
            before = after


def render_play(store):
    """Render a session's active game and accept explicit History saves after a win."""
    if "play_match" not in st.session_state:
        st.session_state.play_match = Match.new()
    match = st.session_state.play_match
    if match.winner is None and match.state.player != match.human_color:
        match.play_bot_turn()

    with st.expander("New game"):
        st.radio("Your color in the new game", ["White", "Brown"], horizontal=True, key="play_new_color")
        st.button("Start new game", key="play_new", on_click=_new_game)

    st.caption(f"You play {_name(match.human_color)} · Baseline plays {_name(-match.human_color)}")
    table, controls = st.columns([2, 1], gap="medium")
    with table:
        if match.winner is None:
            board_view.playable_board(match)
        else:
            moves = match.turns[-1]["moves"]
            board_view.render_board(game.to_legacy(match.state), moves[-1] if moves else None,
                                    key="play_readonly_board")
    with controls:
        if match.winner is None:
            _human_controls(match)
        else:
            who = "You won" if match.winner == match.human_color else "The baseline won"
            st.success(f"{who}. {_name(match.winner)} bore off a checker on turn {len(match.turns)}.")
            if st.session_state.get("play_saved") == match.id:
                st.success("Game saved to History.")
            else:
                st.button("Save to History", type="primary", key="play_save", width="stretch",
                          on_click=_save_game, args=(match, store))
                if st.session_state.get("play_save_error"):
                    st.error(st.session_state.play_save_error)
    _last_bot_turn(match)
