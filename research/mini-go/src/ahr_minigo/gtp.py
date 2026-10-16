"""Bounded GNU Go GTP opponent, independent of the mini-Go rules implementation.

Actions are row-major integers with row zero at the top; ``size * size`` is
pass. Replay the full chronological move history when resetting: reconstructing
only the board discards ko history. The caller remains the authority for legality
and scoring; any disagreement must be reported, never silently replaced.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
from typing import Iterable


class GTPError(RuntimeError):
    """Engine failure, rejected command, unexpected response, or timeout."""


class GnuGo:
    """GNU Go 3.8 level-one opponent with Chinese area rules by default.

    GNU Go's final-score estimates may adjudicate dead stones differently from a
    rules harness that scores the actual board after two passes. Do not use the
    engine's score to adjudicate harness games.
    """

    def __init__(
        self,
        binary: str | Path,
        *,
        size: int = 3,
        komi: float = 0.5,
        level: int = 1,
        seed: int = 0,
        timeout_s: float = 10.0,
        ko_rule: str = "positional",
        rules: str = "chinese",
        capture_all_dead: bool = True,
    ):
        if not 1 <= size <= 19:
            raise ValueError("GNU Go board size must be in 1..19")
        if not math.isfinite(komi) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Finite komi and positive finite timeout required")
        if ko_rule not in {"simple", "positional", "situational"}:
            raise ValueError("ko_rule must be simple, positional, or situational")
        if rules not in {"chinese", "japanese"}:
            raise ValueError("rules must be chinese or japanese")
        if not 0 <= level <= 10:
            raise ValueError("GNU Go level must be in 0..10")
        self.size, self.komi = size, float(komi)
        self.level, self.seed = level, int(seed)
        self.timeout_s = float(timeout_s)
        self.rules, self.ko_rule = rules, ko_rule
        self.capture_all_dead = bool(capture_all_dead)
        self.binary = str(Path(binary).resolve())
        ko_flag = "--simple-ko" if ko_rule == "simple" else f"--{ko_rule}-superko"
        self.argv = [
            self.binary, "--mode", "gtp", "--quiet", "--level", str(level),
            "--seed", str(seed), "--never-resign", f"--{rules}-rules", ko_flag,
            "--forbid-suicide", "--boardsize", str(size), "--komi", str(komi),
        ]
        if self.capture_all_dead:
            self.argv.append("--capture-all-dead")
        self._stderr = tempfile.TemporaryFile()
        self._selector = selectors.DefaultSelector()
        self._buffer = b""
        self._id = 0
        self._closed = False
        try:
            self._process = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, bufsize=0,
            )
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            self.name = self.command("name")
            self.version = self.command("version")
            self.protocol_version = self.command("protocol_version")
            self.reset()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _color(color: int | str) -> str:
        if color in (1, "b", "B", "black", "Black"):
            return "black"
        if color in (-1, "w", "W", "white", "White"):
            return "white"
        raise ValueError(f"Unknown color {color!r}; expected +1/-1 or black/white")

    def to_vertex(self, action: int | None) -> str:
        if action is None or action == self.size * self.size:
            return "pass"
        if not isinstance(action, int) or not 0 <= action < self.size * self.size:
            raise ValueError(f"Invalid action {action!r}")
        row, col = divmod(action, self.size)
        return f"{'ABCDEFGHJKLMNOPQRST'[col]}{self.size - row}"

    def from_vertex(self, vertex: str) -> int:
        vertex = vertex.strip().upper()
        if vertex == "PASS":
            return self.size * self.size
        if vertex == "RESIGN":
            raise GTPError("Engine resigned despite --never-resign")
        try:
            col = "ABCDEFGHJKLMNOPQRST".index(vertex[0])
            row = self.size - int(vertex[1:])
        except (ValueError, IndexError) as exc:
            raise GTPError(f"Malformed GTP vertex: {vertex!r}") from exc
        if not 0 <= row < self.size or not 0 <= col < self.size:
            raise GTPError(f"Out-of-board GTP vertex: {vertex!r}")
        return row * self.size + col

    def command(self, command: str, *, timeout_s: float | None = None) -> str:
        if self._closed:
            raise GTPError("Engine is closed")
        if "\n" in command or "\r" in command:
            raise ValueError("One GTP command per request")
        timeout = self.timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite timeout required")
        self._id += 1
        request_id = self._id
        deadline = time.monotonic() + timeout
        try:
            self._process.stdin.write(f"{request_id} {command}\n".encode("ascii"))
            self._process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise GTPError(f"Cannot send GTP command {command!r}") from exc
        while b"\n\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                self.close()
                raise GTPError(f"GNU Go timed out after {timeout:g}s: {command}")
            chunk = os.read(self._process.stdout.fileno(), 65536)
            if not chunk:
                self._stderr.seek(0)
                diagnostic = self._stderr.read(4096).decode("utf-8", "replace")
                raise GTPError(f"GNU Go exited during {command!r}: {diagnostic}")
            self._buffer += chunk.replace(b"\r", b"")
        raw, self._buffer = self._buffer.split(b"\n\n", 1)
        response = raw.decode("utf-8", "replace").strip()
        header, _, body = response.partition("\n")
        marker, _, first_line = header.partition(" ")
        expected_ok, expected_error = f"={request_id}", f"?{request_id}"
        if marker not in {expected_ok, expected_error}:
            raise GTPError(f"Unexpected GTP response to {command!r}: {response!r}")
        payload = (first_line + ("\n" + body if body else "")).strip()
        if marker == expected_error:
            raise GTPError(f"GNU Go rejected {command!r}: {payload}")
        return payload

    def reset(self, moves: Iterable[tuple[int | str, int | None]] = ()) -> None:
        self.command(f"boardsize {self.size}")
        self.command("clear_board")
        self.command(f"komi {self.komi}")
        for color, action in moves:
            self.play(color, action)

    def play(self, color: int | str, action: int | None) -> None:
        self.command(f"play {self._color(color)} {self.to_vertex(action)}")

    def genmove(self, color: int | str) -> int:
        """Generate AND play a move in the engine's internal position."""
        return self.from_vertex(self.command(f"genmove {self._color(color)}"))

    @property
    def metadata(self) -> dict:
        return {
            "name": self.name, "version": self.version,
            "protocol_version": self.protocol_version,
            "binary": self.binary, "argv": list(self.argv),
            "size": self.size, "komi": self.komi, "level": self.level,
            "seed": self.seed, "rules": self.rules, "ko_rule": self.ko_rule,
            "capture_all_dead": self.capture_all_dead,
            "timeout_s": self.timeout_s,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        self._selector.close()
        self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
