"""Run the magnno table with: streamlit run streamlit_app.py"""

import os
from pathlib import Path

import streamlit as st

from history_store import HistoryStore
from history_tab import render_history
from play_tab import render_play


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="magnno", page_icon="🎲", layout="wide")


@st.cache_resource
def open_history(path):
    return HistoryStore(Path(path).expanduser())


st.html("""
<style>
  .stMainBlockContainer {max-width: 1160px; padding-top: 4.1rem; padding-bottom: 2rem}
  .magnno-head {display: flex; align-items: center; gap: .72rem; border-bottom: 2px solid #233f34;
    padding: .1rem 0 1rem; margin: 0 0 .15rem; color:#233f34}
  .magnno-head h1 {font: italic 700 clamp(2.8rem, 5.4vw, 3.8rem)/.96 Georgia, serif;
    letter-spacing: -.085em; padding: 0 .12em .07em 0; margin: 0}
  .magnno-head small {border-left: 1px solid #c7c4b4; padding-left: 1rem; font-size: .84rem; color: #536559}
  .magnno-mark {position:relative; display:block; flex:none; width:2.8rem; height:2.8rem;
    border-radius:30%; background:#233f34}
  .magnno-mark::before,.magnno-mark::after {content:""; position:absolute; width:44%;
    aspect-ratio:1; border-radius:50%}
  .magnno-mark::before {background:#d68a5d; right:17%; top:15%}
  .magnno-mark::after {background:#f6e5bc; left:18%; bottom:16%; border:3px solid #f6e5bc;
    box-shadow:0 0 0 2px #233f34,inset 0 0 0 1px #b39b70}
  .stTabs [role="tab"] p {font-weight:650}
  @media (max-width:640px) {.stMainBlockContainer {padding-top: 3.9rem}
    .magnno-head {gap:.55rem;padding-bottom:.65rem}
    .magnno-head small {font-size:.72rem;padding-left:.65rem}
    .magnno-mark {width:2.15rem;height:2.15rem}}
</style>
<header class="magnno-head"><span class="magnno-mark" aria-hidden="true"></span>
<h1>magnno</h1><small>Backgammon variant</small></header>
""")

store = open_history(os.environ.get("MAGNNO_HISTORY_DB", str(ROOT / "data" / "history.sqlite3")))
play, history = st.tabs(["Play", "Game history"])
with play:
    render_play(store)
with history:
    render_history(store)
