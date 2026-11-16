"""magnno rules. Public positions use signed physical points 0..23."""

from functools import lru_cache
from random import Random, choice, randint
from typing import Iterable, Iterator, NamedTuple, Optional, Sequence

from game_state import (
    BAR, BROWN, OFF, STATE_SIZE, WHITE, State, _integer,
    deserialize_state, from_legacy, serialize_state, starting_state, state_key,
    to_legacy, winner,
)


Roll = tuple[int, int]


class Move(NamedTuple):
    """A physical move, including the color of the moved checker and die used."""

    checker: int
    source: int
    target: int
    die: int


class Turn(NamedTuple):
    """A complete action and the next state, with the player already switched."""

    moves: tuple[Move, ...]
    state: State


def roll_dice(rng: Optional[Random] = None) -> Roll:
    draw = randint if rng is None else rng.randint
    return draw(1, 6), draw(1, 6)


def opening_roll(rng: Optional[Random] = None) -> tuple[int, Roll]:
    """One die per color. Reroll ties. The higher roller plays both dice."""
    while True:
        white, brown = roll_dice(rng)
        if white != brown:
            return (WHITE if white > brown else BROWN), (white, brown)


def _pips(dice: Sequence[int]) -> tuple[int, ...]:
    try:
        a, b = dice
    except (ValueError, TypeError) as exc:
        raise ValueError("expected a roll of two dice") from exc
    a, b = _integer(a, "die"), _integer(b, "die")
    if not (1 <= a <= 6 and 1 <= b <= 6):
        raise ValueError("dice must be in 1..6")
    if a == b:
        return (a,) * 4
    return (max(a, b), min(a, b))


def _bar_slot(color: int) -> int:
    return 24 if color == WHITE else 25


def _moves(board, player: int, die: int) -> Iterator[Move]:
    if board[_bar_slot(player)]:
        entry = die - 1 if player == WHITE else 24 - die
        if board[entry] * player >= -1:
            yield Move(player, BAR, entry, die)
        return

    points = range(24) if player == WHITE else range(23, -1, -1)
    exit_point = 24 if player == WHITE else -1
    distance = player * die
    farthest = None
    for source in points:
        count = board[source]
        if not count:
            continue
        checker = WHITE if count > 0 else BROWN
        if checker == player and farthest is None:
            farthest = source
        target = source + distance
        if 0 <= target < 24:
            if board[target] * checker >= -1:
                yield Move(checker, source, target, die)
        elif checker == player and abs(exit_point - farthest) <= 12:
            if target == exit_point or source == farthest:
                yield Move(checker, source, OFF, die)


def _steps(board, player, remaining):
    for i, die in enumerate(remaining):
        if i and die == remaining[i - 1]:
            continue
        rest = remaining[:i] + remaining[i + 1:]
        for move in _moves(board, player, die):
            yield move, rest


def _push(board, move):
    """Perform one legal move on the scratch buffer; return its undo information."""
    checker, source, target, _ = move
    slot = _bar_slot(checker) if source == BAR else source
    old_source = board[slot]
    board[slot] -= 1 if source == BAR else checker
    if target == OFF:
        return slot, old_source, None
    old_target = board[target]
    if old_target == -checker:
        board[_bar_slot(-checker)] += 1
        board[target] = checker
    else:
        board[target] += checker
    return slot, old_source, old_target


def _pop(board, move, undo):
    slot, old_source, old_target = undo
    board[slot] = old_source
    if old_target is not None:
        board[move.target] = old_target
        if old_target == -move.checker:
            board[_bar_slot(-move.checker)] -= 1


def _scratch(state: State):
    # Same contiguous byte layout as State. A signed memoryview returns Python
    # integers on scalar reads and permits in-place moves without NumPy temporaries.
    return memoryview(bytearray(state_key(state))).cast("b")


def _rank(moves):
    return len(moves), moves[0].die if len(moves) == 1 else 0


def _endings(state: State, pips):
    """Map each legal next-state key to a representative full-turn move sequence."""
    if winner(state) is not None:
        return {}
    player = int(state.player)
    board = _scratch(state)
    board[26] = -player  # Every completed result, including a pass, gives up the turn.
    seen = {}
    wins, ordinary = {}, {}
    best = (-1, 0)

    def visit(remaining, moves):
        nonlocal best
        key = bytes(board)
        # At most two unused die values, or four identical dice. A repeated
        # prefix can be skipped only if the unspent dice are also identical.
        bucket = seen.get(remaining)
        if bucket is None:
            seen[remaining] = {key}
        elif key in bucket:
            return
        else:
            bucket.add(key)

        won = bool(moves and moves[-1].target == OFF)
        if remaining and not won:
            played = False
            for move, rest in _steps(board, player, remaining):
                played = True
                undo = _push(board, move)
                visit(rest, moves + (move,))
                _pop(board, move, undo)
            if played:
                return

        priority = _rank(moves)
        if priority > best:
            best = priority
            ordinary.clear()
        if won:
            wins.setdefault(key, moves)
        elif priority == best:
            ordinary.setdefault(key, moves)

    visit(pips, ())
    del visit  # Release the recursive closure's visited sets after this search.
    ordinary.update(wins)
    return ordinary


def legal_turns(state: State, dice: Sequence[int]) -> tuple[Turn, ...]:
    """One full turn per distinct next state. A blocked roll yields one pass."""
    endings = _endings(state, _pips(dice))
    return tuple(Turn(moves, State._from_valid_bytes(key)) for key, moves in endings.items())


def sample_turn(state: State, dice: Sequence[int], rng: Optional[Random] = None) -> Turn:
    """Choose a uniform distinct result, building a NumPy state only for that result."""
    endings = _endings(state, _pips(dice))
    if not endings:
        raise ValueError("the game has already finished")
    pick = choice if rng is None else rng.choice
    key = pick(tuple(endings))
    return Turn(endings[key], State._from_valid_bytes(key))


def _replay(state, pips, moves):
    if winner(state) is not None:
        raise ValueError("the game has already finished")
    board = _scratch(state)
    player = int(state.player)
    remaining = list(pips)
    won = False
    for move in moves:
        if won:
            raise ValueError("no move is allowed after a win")
        if not isinstance(move, Move):
            raise ValueError("expected a Move")
        for field in move:
            _integer(field, "move field")
        if move.die not in remaining or move not in _moves(board, player, int(move.die)):
            raise ValueError(f"illegal move: {move}")
        # Accept integer NumPy scalars without propagating their dtype into arithmetic.
        _push(board, Move(*map(int, move)))
        remaining.remove(move.die)
        won = move.target == OFF
    return board, tuple(remaining), won


def apply_turn(state: State, dice: Sequence[int], moves: Iterable[Move]) -> State:
    """Validate even a noncanonical move order. Return a state for the next player."""
    pips, moves = _pips(dice), tuple(moves)
    board, remaining, won = _replay(state, pips, moves)
    board[26] = -board[26]
    key = bytes(board)
    if remaining and not won:
        legal = _endings(state, pips).get(key)
        if legal is None or _rank(legal) != _rank(moves):
            raise ValueError("turn must use the maximum dice and the higher die when required")
    return State._from_valid_bytes(key)


@lru_cache(maxsize=1)
def _interactive_endings(state, pips):
    return {key: _rank(path) for key, path in _endings(state, pips).items()}


def turn_options(state: State, dice: Sequence[int], prefix: Iterable[Move] = ()):
    """Return (preview, legal next steps, done). A preview retains the moving player."""
    pips, prefix = _pips(dice), tuple(prefix)
    board, remaining, won = _replay(state, pips, prefix)
    preview = State._from_valid_bytes(bytes(board))
    player = int(state.player)
    board[26] = -player
    endings = _interactive_endings(state, pips)

    def complete(rest, first_die):
        used = len(pips) - len(rest)
        return endings.get(bytes(board)) == (used, first_die if used == 1 else 0)

    def can_finish(rest, first_die):
        if complete(rest, first_die):
            return True
        for step, left in _steps(board, player, rest):
            undo = _push(board, step)
            allowed = step.target == OFF or can_finish(left, first_die)
            _pop(board, step, undo)
            if allowed:
                return True
        return False

    first_die = prefix[0].die if prefix else 0
    if won or complete(remaining, first_die):
        return preview, (), True
    options = []
    for step, rest in _steps(board, player, remaining):
        undo = _push(board, step)
        if step.target == OFF or can_finish(rest, first_die or step.die):
            options.append(step)
        _pop(board, step, undo)
    del can_finish
    if not options:
        raise ValueError("prefix cannot complete a legal turn")
    return preview, tuple(options), False
