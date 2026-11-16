"""The same point, bar and tray view for the live table and recorded games."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import backgammon as game
from visualizer import _browser_move


_board = components.declare_component("magnno_table", path=Path(__file__).with_name("table_component"))
COLORS = ("White", "Brown")


def move_description(move, player, before):
    """Describe a recorded move; colors and point indices use the browser protocol."""
    actor, checker = COLORS[player], COLORS[move["checker"]]
    source, target, die = move["source"], move["target"], move["die"]
    if source == 0:
        text = f"{actor} entered from the bar on point {target - 2}"
    elif target in (1, 26):
        text = f"{actor} bore off from point {source - 2}"
    elif move["checker"] != player:
        text = f"{actor} moved a {checker} checker backward from {source - 2} to {target - 2}"
    else:
        text = f"{actor} moved from {source - 2} to {target - 2}"
    text += f" with a {die}"
    hit = 1 - move["checker"]
    if before[target][hit]:
        if hit == player:
            text += f", sending {actor}’s own blot to the bar"
        else:
            text += f", capturing {COLORS[hit]}’s blot"
    return text + "."


def render_board(board, last_move=None, *, key="history_board"):
    _board(board=board, last_move=last_move, interactive=False, options=[],
           position="", key=key, default=None)


def playable_board(match):
    """Send legal next moves and accept clicks only for the position that issued them."""
    preview, options, _ = match.preview()
    revision = st.session_state.get("board_click_revision", 0)
    position = f"{match.id}/{len(match.turns)}/{match.prefix}/{revision}"
    legal = [_browser_move(move) for move in options]
    last_move = None
    if match.prefix:
        last_move = _browser_move(match.prefix[-1])
    elif match.turns and match.turns[-1]["moves"]:
        last_move = match.turns[-1]["moves"][-1]
    event = _board(board=game.to_legacy(preview), last_move=last_move,
                   interactive=bool(legal), options=legal, position=position,
                   key="live_clickable_board", default=None)
    if isinstance(event, dict) and event.get("position") == position and event.get("move") in legal:
        match.choose(options[legal.index(event["move"])])
        st.session_state.board_click_revision = revision + 1
        st.rerun()
