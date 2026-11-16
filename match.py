"""Human versus baseline match controller using the native rules engine."""

from copy import deepcopy
from datetime import datetime, timezone
from random import Random
from secrets import randbits
from uuid import uuid4

import backgammon as game
import heuristic
from visualizer import _browser_color, _browser_move


def _board_frame(state):
    return [list(cell) for cell in game.to_legacy(state)]


class Match:
    """One game, including the editable human turn and the completed turn history."""

    @classmethod
    def new(cls, human_color=game.WHITE, seed=None):
        return cls(human_color, seed)

    def __init__(self, human_color=game.WHITE, seed=None):
        self.human_color = game._integer(human_color, "human_color")
        if self.human_color not in (game.WHITE, game.BROWN):
            raise ValueError("human_color must be WHITE or BROWN")
        self.seed = randbits(32) if seed is None else game._integer(seed, "seed")
        self._rng = Random(self.seed)
        player, self.dice = game.opening_roll(self._rng)
        self.state = game.starting_state(player)
        self._opening = _board_frame(self.state)
        self.id = str(uuid4())
        self.turns = []
        self.prefix = ()
        self._draft_frames = []
        self._preview = None
        self._completed_at = None

    @property
    def winner(self):
        return game.winner(self.state)

    def preview(self):
        """Return the displayed state, legal next moves, and whether the turn can end."""
        if self._preview is None:
            if self.winner is not None:
                self._preview = (self.state, (), True)
            else:
                self._preview = game.turn_options(self.state, self.dice, self.prefix)
        return self._preview

    def _require_actor(self, human):
        if self.winner is not None:
            raise ValueError("the game has already finished")
        if (self.state.player == self.human_color) != human:
            raise ValueError("it is the other player's turn")

    def choose(self, move):
        """Play one legal human step. Bearing off commits the win immediately."""
        self._require_actor(True)
        prefix = self.prefix + (move,)
        preview = game.turn_options(self.state, self.dice, prefix)
        self.prefix = prefix
        self._preview = preview
        self._draft_frames.append(_board_frame(preview[0]))
        if game.winner(preview[0]) is not None:
            self.finish_human_turn()

    def undo(self):
        self._require_actor(True)
        if not self.prefix:
            raise ValueError("there is no checker move to undo")
        self.prefix = self.prefix[:-1]
        self._draft_frames.pop()
        self._preview = None

    def finish_human_turn(self):
        """Commit a fully played roll, or the empty move sequence on a forced pass."""
        self._require_actor(True)
        end = game.apply_turn(self.state, self.dice, self.prefix)
        self._commit(self.prefix, end, self._draft_frames)

    def play_bot_turn(self):
        self._require_actor(False)
        turn = heuristic.choose_turn(self.state, self.dice)
        frames = []
        for i in range(1, len(turn.moves) + 1):
            preview, _, _ = game.turn_options(self.state, self.dice, turn.moves[:i])
            frames.append(_board_frame(preview))
        self._commit(turn.moves, turn.state, frames)

    def _commit(self, moves, end, frames):
        record = {
            "player": _browser_color(self.state.player),
            "roll": list(self.dice),
            "moves": [_browser_move(move) for move in moves],
            "boards": frames,
        }
        self.turns.append(record)
        self.state = end
        self.prefix = ()
        self._draft_frames = []
        self._preview = None
        if self.winner is None:
            self.dice = game.roll_dice(self._rng)
        else:
            self._completed_at = datetime.now(timezone.utc).isoformat()

    def recording(self):
        """Return a detached, JSON-ready replay after the first checker bears off."""
        if self.winner is None:
            raise ValueError("only completed games have a recording")
        players = {
            "white": "human" if self.human_color == game.WHITE else "baseline",
            "brown": "human" if self.human_color == game.BROWN else "baseline",
        }
        return {
            "id": self.id, "completed_at": self._completed_at,
            "players": players, "seed": self.seed,
            "opening": deepcopy(self._opening), "turns": deepcopy(self.turns),
            "winner": _browser_color(self.winner),
        }
