import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ahr_mvp"))

from answer import format_verified_answer  # noqa: E402
from ahr_mvp_cli import EXAMPLE_PROMPTS, cmd_demo  # noqa: E402
from router import route_prompt, route_prompt_file  # noqa: E402


EXAMPLES = ROOT / "src" / "ahr_mvp" / "examples"


class AhrMvpAnswerTests(unittest.TestCase):
    def test_formats_graph_unsat_answer(self):
        answer = format_verified_answer(route_prompt_file(EXAMPLES / "graph_two_triangles_unsat.txt"))
        self.assertIn("No such graph exists", answer)
        self.assertIn("checked 32768 graphs", answer)
        self.assertIn("exactly 2 triangles", answer)

    def test_formats_hanoi_unsat_answer(self):
        answer = format_verified_answer(route_prompt_file(EXAMPLES / "hanoi_5disk_after8_bcap2.txt"))
        self.assertIn("No legal completion exists", answer)
        self.assertIn("checked 200 states", answer)
        self.assertIn("after move 8", answer)

    def test_formats_operation_unsat_answer(self):
        answer = format_verified_answer(route_prompt_file(EXAMPLES / "operation_len5_unsat.txt"))
        self.assertIn("No composition exists", answer)
        self.assertIn("checked all 1024 programs", answer)
        self.assertIn("Closest candidates", answer)

    def test_formats_unsupported_answer(self):
        answer = format_verified_answer(route_prompt("unsupported"))
        self.assertIn("Unsupported by the current MVP router", answer)

    def test_all_demo_examples_are_supported(self):
        for path in EXAMPLE_PROMPTS.values():
            with self.subTest(path=path.name):
                self.assertNotEqual(route_prompt_file(path)["module"], "unsupported")

    def test_demo_command_exits_successfully(self):
        class Args:
            json = False

        with redirect_stdout(StringIO()):
            self.assertEqual(cmd_demo(Args()), 0)


if __name__ == "__main__":
    unittest.main()
