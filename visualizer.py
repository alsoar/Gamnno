"""Serve the local magnno board. Run with python3 visualizer.py."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import webbrowser

import backgammon as game
import heuristic


ASSETS = {
    "/": (Path(__file__).with_name("board.html"), "text/html; charset=utf-8"),
    "/board.js": (Path(__file__).with_name("board.js"), "text/javascript; charset=utf-8"),
    "/replay-2095.json": (Path(__file__).with_name("replay-2095.json"), "application/json; charset=utf-8"),
}
API_PATHS = ("/api/new", "/api/turn", "/api/roll", "/api/choices", "/api/heuristic")


def _native_color(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError("expected browser color 0 (white) or 1 (brown)")
    return game.WHITE if value == 0 else game.BROWN


def _browser_color(color):
    return int(color == game.BROWN)


def _browser_move(move):
    """The web protocol predates the engine's signed 24-point coordinates."""
    target = move.target + 2
    if move.target == game.OFF:
        target = 26 if move.checker == game.WHITE else 1
    return {
        "checker": _browser_color(move.checker),
        "source": 0 if move.source == game.BAR else move.source + 2,
        "target": target,
        "die": move.die,
    }


def _native_move(data):
    if not isinstance(data, dict) or data.keys() != {"checker", "source", "target", "die"}:
        raise ValueError("each move needs checker, source, target and die")
    checker = _native_color(data["checker"])
    source, target = data["source"], data["target"]
    if type(source) is not int or type(target) is not int:
        raise ValueError("expected integer source and target")
    if source != 0 and not 2 <= source <= 25:
        raise ValueError("expected the bar or a physical point as source")
    home_tray = 26 if checker == game.WHITE else 1
    if target != home_tray and not 2 <= target <= 25:
        raise ValueError("expected a physical point or the moved color's tray as target")
    return game.Move(checker, game.BAR if source == 0 else source - 2,
                     game.OFF if target == home_tray else target - 2, data["die"])


def game_response(path, payload):
    """Translate a JSON request to and from the rules engine."""
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    if path == "/api/new":
        player, dice = game.opening_roll()
        state = game.starting_state(player)
    else:
        if "board" not in payload or "player" not in payload:
            raise ValueError("expected board and player")
        state = game.from_legacy(payload["board"], _native_color(payload["player"]))
        if path == "/api/roll":
            dice = game.roll_dice()
        elif path in ("/api/turn", "/api/choices", "/api/heuristic"):
            dice = payload.get("dice")
            if not isinstance(dice, list):
                raise ValueError("expected a dice array")
            if path in ("/api/choices", "/api/heuristic"):
                prefix = payload.get("prefix", [])
                if not isinstance(prefix, list):
                    raise ValueError("expected a list of moves for prefix")
                moves = tuple(_native_move(m) for m in prefix)
                if path == "/api/heuristic":
                    turn = heuristic.choose_turn(state, dice, prefix=moves)
                    return {"moves": [_browser_move(m) for m in turn.moves],
                            "board": game.to_legacy(turn.state)}
                current, options, done = game.turn_options(state, dice, moves)
                return {"board": game.to_legacy(current), "options": [_browser_move(m) for m in options], "done": done}
        else:
            raise ValueError("unknown game request")

    turns = game.legal_turns(state, dice)
    winning_color = game.winner(state)
    return {
        "board": game.to_legacy(state),
        "player": _browser_color(state.player),
        "dice": dice,
        "turns": [
            {"moves": [_browser_move(m) for m in turn.moves], "board": game.to_legacy(turn.state)}
            for turn in turns
        ],
        "winner": None if winning_color is None else _browser_color(winning_color),
    }


class BoardHandler(BaseHTTPRequestHandler):
    def _reply(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, data):
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self._reply(status, "application/json; charset=utf-8", body)

    def do_GET(self):
        asset = ASSETS.get(self.path.partition("?")[0])
        if asset is None:
            self.send_error(404)
            return
        file, content_type = asset
        try:
            body = file.read_bytes()
        except OSError:
            self.send_error(500, f"Could not read {file.name} next to visualizer.py")
            return
        self._reply(200, content_type, body)

    def do_POST(self):
        path = self.path.partition("?")[0]
        if path not in API_PATHS:
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("expected a JSON body of up to 64 KiB")
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            state = game_response(path, payload)
        except (ValueError, RecursionError) as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="local port (default: auto)")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening it")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")

    with ThreadingHTTPServer(("127.0.0.1", args.port), BoardHandler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(url, flush=True)
        try:
            if not args.no_browser:
                webbrowser.open(url)
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
