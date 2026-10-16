import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ahr_mvp"))

from operation_synthesis import execute, synthesize, verify  # noqa: E402


class OperationSynthesisTests(unittest.TestCase):
    def test_rev_execution(self):
        result = execute([1, 2, 3], ["REV"])
        self.assertEqual(result.output, (3, 2, 1))
        self.assertFalse(result.truncated)

    def test_verify_valid_program(self):
        result = verify([1, 2, 3], [3, 2, 1], ["REV"])
        self.assertIs(result["valid"], True)
        self.assertEqual(result["output"], [3, 2, 1])

    def test_synthesize_finds_rev(self):
        result = synthesize([1, 2, 3], [3, 2, 1], 1)
        self.assertTrue(result.passed)
        self.assertIn(("REV",), result.solutions)

    def test_synthesize_no_solution_for_fixed_length(self):
        result = synthesize([1, 2, 3], [999], 1)
        self.assertFalse(result.passed)
        self.assertEqual(result.checked, 4)


if __name__ == "__main__":
    unittest.main()
