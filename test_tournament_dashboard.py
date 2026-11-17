"""Regression checks for the bundled tournament and the dashboard's win denominators."""

import json
import os
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agents import AGENT_NAMES
from selfplay import DEFAULT_MAX_TURNS
from tournament_tab import DEFAULT_REPORT


class TournamentDashboardTests(TestCase):
    def test_bundled_matrix_and_censored_lengths_match_the_report(self):
        report = json.loads(DEFAULT_REPORT.read_text())
        self.assertEqual(report["max_turns"], DEFAULT_MAX_TURNS)
        self.assertEqual(set(report["matrix"]), set(AGENT_NAMES))
        self.assertEqual(len(report["matchups"]), len(AGENT_NAMES) * (len(AGENT_NAMES) + 1) // 2)
        self.assertEqual(report["games"], report["completed"] + report["unfinished"])
        capped = report["length"]["unfinished_observed"]["histogram"]
        self.assertEqual(capped, {str(DEFAULT_MAX_TURNS): report["unfinished"]})

    def test_dashboard_displays_all_three_matrix_views(self):
        with patch.dict(os.environ, {"MAGNNO_TOURNAMENT_REPORT": str(DEFAULT_REPORT)}):
            app = AppTest.from_string("from tournament_tab import render_tournament\nrender_tournament()").run()
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.dataframe), 2)
            self.assertEqual(app.dataframe[0].value.shape, (9, 9))
            all_starts = app.dataframe[0].value.copy()
            app.radio[0].set_value("Wins / finished games").run()
            self.assertFalse(all_starts.equals(app.dataframe[0].value))
            app.radio[0].set_value("Unfinished / starts").run()
            self.assertFalse(list(app.exception))
            self.assertEqual(app.dataframe[0].value.shape, (9, 9))
