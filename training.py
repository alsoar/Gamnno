"""Player-relative neural inputs and compact, sequential self-play records.

encode_state returns float32 checker counts (2, 24), bar/off counts (4,),
and an optional histogram of the two rolled faces (6,). Channel 0 always
moves toward increasing indices. The auxiliary values are active bar,
opponent bar, active off, opponent off. A pre-roll position has dice=None.

For training data, open a buffered binary file (or gzip file) and call
write_records(file, iterable) once. read_records(file) yields one record at
a time. The five-byte stream header is b"FOBG" plus version 1. Each record
is 29 to 37 bytes:

  0..26   serialize_state(state), in the engine's absolute coordinates
  27      two dice in high/low nibbles, in supplied order (0 = no roll)
  28      bits 0..2 count moves (0 = pass, 7 = no selected action);
          bits 3..4 encode final winner (0 = unknown, 1 = white, 2 = brown)
  29..    one little-endian uint16 per move: source in bits 0..4, target
          in 5..9, die minus one in 10..12, brown checker flag in 13

All reserved bits must be zero. A selected action is a complete turn from
legal_turns, or an equivalent legal ordering. The codec checks the layout
and dice budget; apply_turn can check legality if importing external data.
"""

import struct
from typing import BinaryIO, Iterable, Iterator, NamedTuple, Optional

import numpy as np

from backgammon import BAR, OFF, BROWN, WHITE, Move, State, deserialize_state, serialize_state
from game_state import _integer


class EncodedState(NamedTuple):
    board: np.ndarray
    aux: np.ndarray
    dice: Optional[np.ndarray]


class TrainingRecord(NamedTuple):
    state: State
    dice: Optional[tuple[int, int]] = None
    action: Optional[tuple[Move, ...]] = None
    winner: Optional[int] = None

    @property
    def value_target(self) -> Optional[int]:
        """+1 for the player to act winning, -1 for losing, None for unknown."""
        return None if self.winner is None else int(self.winner) * int(self.state.player)


def _roll(dice):
    if dice is None:
        return None
    try:
        first, second = dice
    except (ValueError, TypeError) as error:
        raise ValueError("expected two dice") from error
    first, second = _integer(first, "die"), _integer(second, "die")
    if not (1 <= first <= 6 and 1 <= second <= 6):
        raise ValueError("dice must be between 1 and 6")
    return first, second


def encode_state(state: State, dice=None) -> EncodedState:
    """Expand a position at training time, optionally including the current roll."""
    if state.player == WHITE:
        relative = state.board
        extras = (state.white_bar, state.brown_bar, state.white_off, state.brown_off)
    else:
        relative = -state.board[::-1]
        extras = (state.brown_bar, state.white_bar, state.brown_off, state.white_off)
    board = np.empty((2, 24), dtype=np.float32)
    np.maximum(relative, 0, out=board[0])
    np.maximum(-relative, 0, out=board[1])
    roll = _roll(dice)
    faces = None
    if roll is not None:
        faces = np.zeros(6, dtype=np.float32)
        for die in roll:
            faces[die - 1] += 1
    return EncodedState(board, np.asarray(extras, dtype=np.float32), faces)


_HEADER = b"FOBG\x01"
_BASE_SIZE = 29
_NO_ACTION = 7
_WORD = struct.Struct("<H")
_WINNERS = (None, WHITE, BROWN)


def _checked_moves(action, dice):
    if dice is None:
        raise ValueError("a selected action requires a roll")
    if not isinstance(action, (tuple, list)):
        raise ValueError("expected a sequence of moves")
    budget = list(dice) if dice[0] != dice[1] else [dice[0]] * 4
    for move in action:
        if not isinstance(move, Move):
            raise ValueError("expected a Move in the selected action")
        checker, source, target, die = (_integer(value, "move field") for value in move)
        if checker not in (WHITE, BROWN):
            raise ValueError("invalid checker color")
        if not (0 <= source <= BAR and (0 <= target < BAR or target == OFF)):
            raise ValueError("invalid source or target")
        if die not in budget:
            raise ValueError("selected action exceeds its rolled dice")
        budget.remove(die)
        yield Move(checker, source, target, die)


def serialize_record(record: TrainingRecord) -> bytes:
    """Pack one record, without a file header. Actions use absolute coordinates."""
    dice = _roll(record.dice)
    winner = None if record.winner is None else _integer(record.winner, "winner")
    if winner not in _WINNERS:
        raise ValueError("expected a white, brown, or unknown winner")
    length = _NO_ACTION
    payload = bytearray()
    if record.action is not None:
        for checker, source, target, die in _checked_moves(record.action, dice):
            word = source | (target << 5) | ((die - 1) << 10) | (int(checker == BROWN) << 13)
            payload.extend(_WORD.pack(word))
        length = len(payload) // _WORD.size
    packed_dice = 0 if dice is None else (dice[0] << 4) | dice[1]
    meta = length | (_WINNERS.index(winner) << 3)
    return serialize_state(record.state) + bytes((packed_dice, meta)) + payload


def _action_length(meta):
    length = meta & 7
    if meta >> 3 >= len(_WINNERS) or length in (5, 6):
        raise ValueError("invalid record flags")
    return None if length == _NO_ACTION else length


def deserialize_record(data: bytes) -> TrainingRecord:
    """Restore one complete record, rejecting truncated or malformed data."""
    if len(data) < _BASE_SIZE:
        raise ValueError("truncated training record")
    packed_dice, meta = data[27:29]
    length = _action_length(meta)
    if len(data) != _BASE_SIZE + _WORD.size * (length or 0):
        raise ValueError("incorrect training record length")
    dice = _roll((packed_dice >> 4, packed_dice & 15)) if packed_dice else None
    action = None
    if length is not None:
        moves = []
        for (word,) in _WORD.iter_unpack(data[_BASE_SIZE:]):
            if word >> 14:
                raise ValueError("reserved move bits must be zero")
            checker = BROWN if word & (1 << 13) else WHITE
            moves.append(Move(checker, word & 31, (word >> 5) & 31, ((word >> 10) & 7) + 1))
        action = tuple(_checked_moves(moves, dice))
    state = deserialize_state(data[:27])
    return TrainingRecord(state, dice, action, _WINNERS[meta >> 3])


def write_records(file: BinaryIO, records: Iterable[TrainingRecord]) -> int:
    """Write a stream header and records sequentially. Return the number written."""
    file.write(_HEADER)
    count = 0
    for record in records:
        file.write(serialize_record(record))
        count += 1
    return count


def read_records(file: BinaryIO) -> Iterator[TrainingRecord]:
    """Read from a buffered binary stream, yielding records as they arrive."""
    if file.read(len(_HEADER)) != _HEADER:
        raise ValueError("unknown or truncated training stream header")
    while True:
        base = file.read(_BASE_SIZE)
        if not base:
            return
        if len(base) < _BASE_SIZE:
            raise ValueError("truncated training record")
        length = _action_length(base[28])
        yield deserialize_record(base + file.read(_WORD.size * (length or 0)))
