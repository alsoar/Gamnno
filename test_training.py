"""Neural encoding symmetries, binary compatibility, and streaming checks."""

from io import BytesIO
import unittest

import numpy as np

import backgammon as game
import training as t


def position(stacks, white_bar=0, brown_bar=0, player=game.WHITE):
    board = np.zeros(24, dtype=np.int8)
    for point, count in stacks.items():
        board[point] = count
    return game.State(board, white_bar, brown_bar, player)


class EncodingTests(unittest.TestCase):
    def test_board_coordinates_bar_counts_and_color_swap_symmetry(self):
        state = position({0: 4, 6: 6, 22: 3, 1: -5, 14: -9}, white_bar=2, brown_bar=1)
        swapped = game.State(-state.board[::-1], state.brown_bar, state.white_bar, game.BROWN)
        white = t.encode_state(state, (2, 6))
        brown = t.encode_state(swapped, (6, 2))
        self.assertEqual(white.board.shape, (2, 24))
        for own, mirrored in zip(white, brown):
            self.assertEqual(own.dtype, np.float32)
            np.testing.assert_array_equal(own, mirrored)
        np.testing.assert_array_equal(white.board[0], np.maximum(state.board, 0))
        np.testing.assert_array_equal(white.board[1], np.maximum(-state.board, 0))
        np.testing.assert_array_equal(white.aux, [2, 1, 0, 0])
        np.testing.assert_array_equal(white.dice, [0, 1, 0, 0, 0, 1])
        self.assertFalse(np.shares_memory(white.board, state.board))

    def test_pre_roll_and_double_face_counts(self):
        state = game.starting_state()
        self.assertIsNone(t.encode_state(state).dice)
        np.testing.assert_array_equal(t.encode_state(state, (3, 3)).dice, [0, 0, 2, 0, 0, 0])
        for dice in ((0, 1), (True, 5), (1, 2, 3), (np.nan, 5)):
            with self.subTest(dice=dice), self.assertRaises(ValueError):
                t.encode_state(state, dice)

    def test_derived_off_counts_follow_player_to_act(self):
        state = position({3: -14, 20: 14, 23: 1}, brown_bar=1)
        after = game.apply_turn(state, (1, 3), (game.Move(game.WHITE, 23, game.OFF, 1),))
        self.assertEqual(after.player, game.BROWN)
        np.testing.assert_array_equal(t.encode_state(after).aux, [1, 0, 0, 1])


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.state = game.starting_state()

    def test_replay_a_completed_two_turn_game_from_training_records(self):
        start = position({0: -7, 1: -8, 22: 14}, white_bar=1)
        passed = game.apply_turn(start, (1, 2), ())
        winning_move = (game.Move(game.BROWN, 0, game.OFF, 1),)
        final = game.apply_turn(passed, (1, 6), winning_move)
        won = game.winner(final)
        self.assertEqual(won, game.BROWN)
        stream = BytesIO()
        records = [t.TrainingRecord(start, (1, 2), (), won),
                   t.TrainingRecord(passed, (1, 6), winning_move, won)]
        t.write_records(stream, records)
        stream.seek(0)
        current = start
        targets = []
        for row in t.read_records(stream):
            self.assertEqual(row.state, current)
            targets.append(row.value_target)
            current = game.apply_turn(current, row.dice, row.action)
        self.assertEqual(current, final)
        self.assertEqual(targets, [-1, 1])

    def test_round_trip_and_exact_move_bytes(self):
        state = position({20: 14, 2: -15}, white_bar=1)
        action = (game.Move(game.WHITE, game.BAR, 4, 5), game.Move(game.BROWN, 2, 3, 1))
        before = t.TrainingRecord(state, (5, 1), action, game.BROWN)
        packed = t.serialize_record(before)
        self.assertEqual(packed[:27], game.serialize_state(state))
        self.assertEqual(packed[27:], bytes((0x51, 0x12, 0x98, 0x10, 0x62, 0x20)))
        after = t.deserialize_record(packed)
        self.assertEqual(game.serialize_state(after.state), packed[:27])
        self.assertEqual(after.dice, (5, 1))
        self.assertEqual(after.action, action)
        self.assertEqual(after.winner, game.BROWN)
        self.assertEqual(after.value_target, -1)
        self.assertEqual(t.serialize_record(after), packed)
        played = game.apply_turn(after.state, after.dice, after.action)
        self.assertEqual((played.board[4], played.board[3]), (1, -1))
        self.assertEqual(t.TrainingRecord(played, winner=game.BROWN).value_target, 1)

    def test_doubles_fit_in_37_bytes(self):
        turn = game.legal_turns(self.state, (3, 3))[0]
        record = t.TrainingRecord(self.state, (3, 3), turn.moves, game.WHITE)
        packed = t.serialize_record(record)
        self.assertEqual(len(packed), 37)
        self.assertEqual(t.deserialize_record(packed).action, turn.moves)

    def test_pre_roll_no_action_and_rolled_pass_are_distinct(self):
        blocked = position({0: -7, 1: -8, 22: 14}, white_bar=1)
        records = (t.TrainingRecord(blocked),
                   t.TrainingRecord(blocked, (1, 2)),
                   t.TrainingRecord(blocked, (1, 2), (), game.WHITE))
        encoded = [t.serialize_record(record) for record in records]
        self.assertEqual([len(data) for data in encoded], [29] * 3)
        self.assertEqual(len(set(encoded)), 3)
        before_roll, no_selection, passed = map(t.deserialize_record, encoded)
        self.assertIsNone(before_roll.dice)
        self.assertIsNone(no_selection.action)
        self.assertIsNone(no_selection.value_target)
        self.assertEqual(passed.action, ())
        self.assertEqual(passed.value_target, 1)
        self.assertEqual(game.legal_turns(blocked, (1, 2))[0].moves, passed.action)

    def test_stream_consumes_an_iterable_and_reads_one_record_at_a_time(self):
        file = BytesIO()
        records = (t.TrainingRecord(self.state, winner=side) for side in (game.WHITE, game.BROWN, None))
        self.assertEqual(t.write_records(file, records), 3)
        self.assertEqual(len(file.getvalue()), 5 + 3 * 29)
        file.seek(0)
        it = t.read_records(file)
        self.assertEqual(next(it).winner, game.WHITE)
        self.assertEqual(file.tell(), 5 + 29)
        self.assertEqual([row.winner for row in it], [game.BROWN, None])

    def test_truncated_and_malformed_records_are_rejected(self):
        move = game.legal_turns(self.state, (1, 5))[0].moves
        packed = t.serialize_record(t.TrainingRecord(self.state, (1, 5), move))
        pref = packed[:27]
        corrupt = (
            packed[:-1], packed + b"!", pref + b"\x05\x07", pref + b"\x17\x07",
            pref + b"\x00\x00", pref + b"\x00\x05", pref + b"\x00\xff",
            pref + b"\x15\x01\x00\xc0", pref + b"\x15\x01\x1f\x00",
            pref + b"\x15\x01\x00\x0c",
        )
        for data in corrupt:
            with self.subTest(tail=data[27:]), self.assertRaises(ValueError):
                t.deserialize_record(data)
        for file_data in (b"", b"FOBG\x02", b"FOBG\x01" + packed[:-1]):
            with self.subTest(file=file_data), self.assertRaises(ValueError):
                list(t.read_records(BytesIO(file_data)))

    def test_writer_rejects_inconsistent_action_or_winner(self):
        first = game.Move(game.WHITE, 0, 1, 1)
        cases = (
            (None, (), None),
            ((1, 2), (first, first), None),
            ((1, 1), (first,) * 5, None),
            ((1, 2), (game.Move(0, 0, 1, 1),), None),
            ((1, 2), (game.Move(game.WHITE, 0, game.BAR, 1),), None),
            ((1, 2), (game.Move(game.WHITE, game.OFF, 1, 1),), None),
            ((1, 2), None, 0),
            ((1, 2), None, True),
        )
        for dice, action, winner in cases:
            with self.subTest(case=(dice, action, winner)), self.assertRaises(ValueError):
                t.serialize_record(t.TrainingRecord(self.state, dice, action, winner))


if __name__ == "__main__":
    unittest.main()
