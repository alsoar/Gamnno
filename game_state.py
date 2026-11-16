"""Immutable magnno positions. Wire order: 24 signed points, two bars, signed player."""

from hashlib import blake2b
import operator
from struct import Struct
from typing import Iterable, Optional

import numpy as np


WHITE, BROWN = 1, -1
BAR, OFF = 24, 25  # Move sentinels, outside the 24 physical points.
STATE_SIZE = 27
_WIRE = Struct("24bBBb")
_WHITE_COUNTS = bytes(i if i < 128 else 0 for i in range(256))
_BROWN_COUNTS = bytes(256 - i if i >= 128 else 0 for i in range(256))


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_counts(points, white_bar: int, brown_bar: int, player: int) -> None:
    if player not in (WHITE, BROWN):
        raise ValueError("player must be +1 (White) or -1 (Brown)")
    if not (0 <= white_bar <= 15 and 0 <= brown_bar <= 15):
        raise ValueError("each bar must contain 0 to 15 checkers")
    white = white_bar
    brown = brown_bar
    for point in points:
        if point > 0:
            white += point
        else:
            brown -= point
    # Each color starts with 15. A legal game stops at its first bear-off.
    if white > 15 or brown > 15 or white + brown < 29:
        raise ValueError("expected 15 checkers per color, including at most one inferred bear-off")


class State:
    """A signed NumPy board backed by immutable bytes, with separate bars and player."""

    __slots__ = ("_points",)

    def __init__(self, board, white_bar: int, brown_bar: int, player: int):
        try:
            points = tuple(_integer(v, "point count") for v in board)
        except TypeError as exc:
            raise ValueError("expected a one-dimensional board") from exc
        if len(points) != 24:
            raise ValueError("expected 24 point counts")
        white_bar = _integer(white_bar, "white_bar")
        brown_bar = _integer(brown_bar, "brown_bar")
        player = _integer(player, "player")
        _validate_counts(points, white_bar, brown_bar, player)
        self._bind(_WIRE.pack(*points, white_bar, brown_bar, player))

    def _bind(self, data: bytes) -> None:
        if hasattr(self, "_points"):
            raise AttributeError("State is immutable")
        object.__setattr__(self, "_points", np.frombuffer(data, dtype=np.int8, count=24))

    @classmethod
    def _from_valid_bytes(cls, data: bytes):
        state = object.__new__(cls)
        state._bind(data)
        return state

    def __setattr__(self, name, value):
        raise AttributeError("State is immutable")

    def __delattr__(self, name):
        raise AttributeError("State is immutable")

    @property
    def board(self) -> np.ndarray:
        # Reuse the same 24 bytes. A caller changing their view's shape or dtype
        # cannot change the metadata on the engine's private array.
        return np.frombuffer(self._points.base, dtype=np.int8, count=24)

    @property
    def white_bar(self) -> np.uint8:
        return np.uint8(self._points.base[24])

    @property
    def brown_bar(self) -> np.uint8:
        return np.uint8(self._points.base[25])

    @property
    def player(self) -> np.int8:
        return np.int8(WHITE if self._points.base[26] == 1 else BROWN)

    @property
    def white_off(self) -> int:
        data = self._points.base
        return 15 - data[24] - sum(data[:24].translate(_WHITE_COUNTS))

    @property
    def brown_off(self) -> int:
        data = self._points.base
        return 15 - data[25] - sum(data[:24].translate(_BROWN_COUNTS))

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self._points.base == other._points.base

    def __hash__(self) -> int:
        digest = blake2b(self._points.base, digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=True)

    def __reduce__(self):
        return deserialize_state, (self._points.base,)


def serialize_state(state: State) -> bytes:
    """Exact 27-byte representation, already immutable and suitable for storage."""
    return state._points.base


def state_key(state: State) -> bytes:
    """Collision-free position key. For post-roll nodes also key by sorted dice."""
    return serialize_state(state)


def deserialize_state(data) -> State:
    """Validate one packed position. Mutable input buffers are copied."""
    try:
        packed = data if type(data) is bytes else memoryview(data).tobytes()
    except TypeError as exc:
        raise ValueError("state must be a 27-byte buffer") from exc
    if len(packed) != STATE_SIZE:
        raise ValueError("state must be exactly 27 bytes")
    fields = _WIRE.unpack(packed)
    _validate_counts(fields[:24], *fields[24:])
    return State._from_valid_bytes(packed)


def winner(state: State) -> Optional[int]:
    """Return the winning color or None. Off counts are inferred from conservation."""
    if state.white_off:
        return WHITE
    if state.brown_off:
        return BROWN
    return None


def starting_state(player: int = WHITE) -> State:
    points = [0] * 24
    for index, count in ((0, 2), (11, 5), (16, 3), (18, 5)):
        points[index] = count
        points[23 - index] = -count
    return State(points, 0, 0, player)


def to_legacy(state: State) -> tuple:
    """Export the original 27 (white, brown) cells, including bar and trays."""
    points = _WIRE.unpack(serialize_state(state))[:24]
    return (((int(state.white_bar), int(state.brown_bar)), (0, state.brown_off))
            + tuple((max(v, 0), max(-v, 0)) for v in points)
            + ((state.white_off, 0),))


def from_legacy(cells: Iterable, player: int) -> State:
    """Import a full 15-per-color legacy position. Supply the NEW signed player."""
    try:
        pairs = tuple(tuple(_integer(v, "checker count") for v in pair) for pair in cells)
    except TypeError as exc:
        raise ValueError("expected 27 pairs of checker counts") from exc
    if len(pairs) != 27 or any(len(pair) != 2 or min(pair) < 0 for pair in pairs):
        raise ValueError("expected 27 pairs of nonnegative checker counts")
    if pairs[1][0] or pairs[26][1] or any(w and b for w, b in pairs[2:26]):
        raise ValueError("invalid tray or mixed physical point")
    state = State((w - b for w, b in pairs[2:26]), *pairs[0], player)
    if (state.white_off, state.brown_off) != (pairs[26][0], pairs[1][1]):
        raise ValueError("legacy tray counts must match 15 checkers per color")
    return state
