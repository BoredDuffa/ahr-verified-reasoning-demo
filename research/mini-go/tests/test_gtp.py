"""Protocol and replay checks against the locally built GNU Go engine."""

from pathlib import Path
import unittest

from ahr_minigo.gtp import GnuGo, GTPError


BINARY = Path(__file__).resolve().parents[1] / "vendor/gnugo-3.8/interface/gnugo"


@unittest.skipUnless(BINARY.is_file(), "Local GNU Go binary has not been built")
class GnuGoTests(unittest.TestCase):
    def test_coordinate_conversion_including_skipped_i_column(self):
        with GnuGo(BINARY, size=19) as engine:
            self.assertEqual(engine.to_vertex(0), "A19")
            self.assertEqual(engine.to_vertex(8), "J19")
            self.assertEqual(engine.to_vertex(360), "T1")
            for action in range(362):
                self.assertEqual(engine.from_vertex(engine.to_vertex(action)), action)
            with self.assertRaises(GTPError):
                engine.from_vertex("I1")

    def test_3x3_move_generation_and_full_replay(self):
        with GnuGo(BINARY, size=3, seed=19) as engine:
            history = [(1, engine.genmove(1)), (-1, engine.genmove(-1))]
            board = engine.command("showboard")
            engine.reset(history)
            self.assertEqual(engine.command("showboard"), board)
            self.assertEqual(engine.version, "3.8")
            self.assertEqual(engine.command("query_boardsize"), "3")
            self.assertEqual(engine.command("get_komi"), "0.5")

    def test_5x5_capture_and_illegal_occupied_point(self):
        with GnuGo(BINARY, size=5) as engine:
            # Capture white A5 with black A4 and B5; chronological play preserves
            # all captures and any subsequent ko-history dependence.
            history = [(1, 5), (-1, 0), (1, 1)]
            engine.reset(history)
            self.assertEqual(engine.command("color A5"), "empty")
            with self.assertRaises(GTPError):
                engine.play(-1, 1)
            self.assertEqual(engine.command("color B5"), "black")
            self.assertIn(engine.genmove(-1), range(26))

    def test_two_passes_and_reset(self):
        with GnuGo(BINARY, size=3) as engine:
            engine.reset([(1, 4), (-1, 9), (1, 9)])
            self.assertEqual(engine.command("color B2"), "black")
            engine.reset()
            self.assertEqual(engine.command("color B2"), "empty")


if __name__ == "__main__":
    unittest.main()
