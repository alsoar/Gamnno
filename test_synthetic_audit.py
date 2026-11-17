"""Independent checks for value labels, pre-roll schema, archive integrity and split leakage."""

from collections import Counter
from contextlib import closing
from pathlib import Path
from random import Random
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import backgammon as game
import synthetic_audit as audit
from synthetic_store import VALUE_SCHEMA


class AuditTests(unittest.TestCase):
    def test_wire_integrity_and_percentiles(self):
        state = game.starting_state(game.BROWN)
        self.assertEqual(audit.validate_wire(game.state_key(state)),state)
        broken = bytearray(game.state_key(state))
        for ix,value in ((1,16),(24,16),(26,0)):
            modified = bytearray(broken)
            modified[ix] = value
            with self.subTest(ix=ix), self.assertRaises(audit.AuditError):
                audit.validate_wire(bytes(modified))
        terminal = game.state_key(game.State([-15]+[0]*22+[14],0,0,game.BROWN))
        self.assertEqual(game.winner(audit.validate_wire(terminal,terminal_allowed=True)),game.WHITE)
        with self.assertRaises(audit.AuditError):
            audit.validate_wire(terminal)
        self.assertEqual(audit.distribution(Counter({2:1,8:1}))["median"],5)
        self.assertEqual(audit.distribution(Counter())[ "p90"],None)

    def test_replay_checks_recorded_dice_hits_policy_chain_and_sample(self):
        from synthetic_game import Ply
        seed = 3
        opener,dice = game.opening_roll(Random(seed))
        state = game.starting_state(opener)
        move = game.legal_turns(state,dice)[0]
        c,self_c = audit._capture_counts(state,move.moves)
        ply = Ply(1,state,dice,move.moves,move.state,2,c,self_c)
        entry = {"opener":opener,"policy_trace":bytes([2]),"turns":1,"final":game.state_key(move.state)}
        side = "white" if opener == game.WHITE else "brown"
        stats = {"white":{},"brown":{}}
        stats[side].update(captures=c,self_captures=self_c)
        spec = SimpleNamespace(dice_seed=seed)
        def check(p, positions):
            with patch("synthetic_game.iter_trajectory",return_value=iter((p,))):
                return audit._check_trajectory(b"",entry,positions,spec,stats)
        self.assertEqual(check(ply,{1:(game.state_key(state),0,2)}),1)
        for corrupt in (ply._replace(policy_code=1),ply._replace(dice=(6,6)),
                        ply._replace(after=state),ply._replace(captures=c+1)):
            with self.subTest(ply=corrupt),self.assertRaises(audit.AuditError):
                check(corrupt,{})
        with self.assertRaises(audit.AuditError):
            check(ply,{1:(game.state_key(move.state),0,2)})

    def test_parquet_detects_wrong_label_future_dice_duplicate_ownership_and_omissions(self):
        metadata = {"agents":{"0":"sprinter","1":"assassin"},"categories":{"0":"cross"},
                    "mixture_types":{"0":"fixed"},"strategies":{"0":"baseline"}}
        reached = game.legal_turns(game.starting_state(game.BROWN),(3,4))[0].state
        state = game.state_key(reached)
        mirrored = game.State(-reached.board[::-1],reached.brown_bar,reached.white_bar,game.BROWN)
        other = game.state_key(mirrored)
        self.assertNotEqual(state,other)
        self.assertEqual(audit._actor_key(state),audit._actor_key(other))
        from training import encode_state
        for a,b in zip(encode_state(reached)[:2],encode_state(mirrored)[:2]):
            self.assertTrue((a==b).all())
        base = dict(game_id="a",turn_number=3,position_id=3,board=state[:24],white_bar=0,brown_bar=0,
                    player_to_move=1,winner=1,win_target=1,white_agent=0,brown_agent=1,
                    matchup_category=0,mixture_type=0,game_phase=0,game_length=8,generation_seed=8,
                    source_policy_metadata=0)
        for case in ("valid","valid_mirror_filter","wrong_label","future_die","same_turn_twice",
                     "wrong_side","late_omission","leak","mirrored_leak"):
            database, index = closing(sqlite3.connect(":memory:")), closing(sqlite3.connect(":memory:"))
            with self.subTest(case=case), TemporaryDirectory() as temp, database as db, index as scratch:
                root = Path(temp)
                db.row_factory = sqlite3.Row
                db.execute("""CREATE TABLE games(game_id TEXT,split TEXT,winner INTEGER,turns INTEGER,generation_seed INTEGER,
                           white_agent TEXT,brown_agent TEXT,category TEXT,mixture_type TEXT)""")
                db.execute("""CREATE TABLE samples(game_id TEXT,turn_number INTEGER,state BLOB,phase INTEGER,policy INTEGER)""")
                for gid,split in (("a","train"),("b","test")):
                    db.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?,?)",
                               (gid,split,1,8,8,"sprinter","assassin","cross","fixed"))
                    wire = other if gid=="b" and case in ("valid_mirror_filter","mirrored_leak") else state
                    db.execute("INSERT INTO samples VALUES(?,?,?,?,?)",(gid,3,wire,0,0))
                row = dict(base)
                if case == "wrong_label":
                    row["win_target"] = 0
                if case == "future_die":
                    row["die1"] = 4
                if case == "wrong_side":
                    row["player_to_move"] = -1
                if case == "late_omission":
                    db.execute("INSERT INTO samples VALUES(?,?,?,?,?)",("a",4,game.state_key(game.starting_state(-1)),0,0))
                (root/"train").mkdir()
                schema = VALUE_SCHEMA.append(pa.field("die1",pa.uint8())) if case=="future_die" else VALUE_SCHEMA
                pq.write_table(pa.Table.from_pylist([row]*(2 if case=="same_turn_twice" else 1),schema=schema),
                               root/"train/data.parquet")
                if case in ("leak","mirrored_leak"):
                    (root/"test").mkdir()
                    b = dict(row,game_id="b")
                    if case=="mirrored_leak":
                        b.update(board=other[:24],white_bar=other[24],brown_bar=other[25],
                                 player_to_move=-1,win_target=0)
                    pq.write_table(pa.Table.from_pylist([b],schema=VALUE_SCHEMA),root/"test/data.parquet")
                if case in ("valid","valid_mirror_filter"):
                    diversity,positions,dropped = audit._audit_positions(root,db,scratch,metadata)
                    self.assertEqual((diversity["unique"],dropped),(1,1))
                    self.assertEqual(positions["train"]["labels"],{1:1})
                else:
                    with self.assertRaises(audit.AuditError):
                        audit._audit_positions(root,db,scratch,metadata)


if __name__ == "__main__":
    unittest.main()
