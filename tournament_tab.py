"""Browse the bundled measured tournament without starting simulations in the app."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from selfplay import DEFAULT_MAX_TURNS


DEFAULT_REPORT = Path(__file__).with_name("results") / "round-robin.json"


@st.cache_resource
def _read_report(path, modification_time):
    with path.open() as stream:
        return json.load(stream)


def _percent(value):
    return "—" if value is None else f"{value:.1%}"


@st.fragment
def render_tournament():
    path = Path(os.environ.get("MAGNNO_TOURNAMENT_REPORT", DEFAULT_REPORT)).expanduser()
    if not path.is_file():
        st.info("Generate the tournament report using the command in the README.")
        return
    report = _read_report(path, path.stat().st_mtime_ns)
    st.subheader("Round robin")
    st.write(f"{report['games']:,} games across {report['pairings']} pairings. "
             f"{report['completed']} finished and {report['unfinished']} reached the "
             f"{report['max_turns']:,}-turn limit.")
    if report["max_turns"] != DEFAULT_MAX_TURNS:
        st.caption(f"This saved run used a different limit. New simulations default to {DEFAULT_MAX_TURNS:,} player turns.")
    view = st.radio("Head-to-head matrix", ["Wins / all starts", "Wins / finished games", "Unfinished / starts"],
                    horizontal=True)
    rows = {}
    for agent, opponents in report["matrix"].items():
        cells = {}
        for name, cell in opponents.items():
            if view == "Wins / all starts":
                fraction = cell["win_fraction_all"]
            elif view == "Wins / finished games":
                fraction = cell["win_fraction_completed"]
            else:
                fraction = cell["unfinished"] / cell["appearances"]
            cells[name.title()] = _percent(fraction)
        rows[agent.title()] = cells
    st.dataframe(pd.DataFrame.from_dict(rows, orient="index"), width="stretch")
    st.caption("Read across a row for one agent’s results. Self-play pools both seats. "
               "A dash means no game finished in that pairing.")

    st.subheader("Playing styles")
    metrics = {
        "Capture turns": "capture_turn_fraction", "Backward steps": "backward_step_fraction",
        "Blots": "mean_blots", "Bar": "mean_bar", "Concentration": "mean_concentration",
        "Occupied points": "mean_occupied", "Blockade": "mean_blockade",
        "Victory progress": "mean_victory_gain", "Eligible": "eligible_fraction",
        "Observed game length": "average_game_length",
    }
    percentages = {"capture_turn_fraction", "backward_step_fraction", "eligible_fraction"}
    styles = {}
    for name, profile in report["profiles"].items():
        styles[name.title()] = {label: _percent(profile[key]) if key in percentages else
                                (round(profile[key], 2) if profile[key] is not None else "—")
                                for label, key in metrics.items()}
    st.dataframe(pd.DataFrame.from_dict(styles, orient="index"), width="stretch")
    st.caption("Board measures are averages after the agent’s turns. Eligibility is measured after both "
               "players’ turns. Observed game lengths include games stopped at the limit.")
    near = report["diversity"]["flagged_near_pairs"]
    if near:
        st.write("Similar measured profiles: " + ", ".join(f"{a.title()} and {b.title()}" for a, b in near) + ".")
    else:
        st.write("No pair met the configured similarity threshold in this run.")
