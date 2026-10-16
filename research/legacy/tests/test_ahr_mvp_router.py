import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ahr_mvp"))

from router import route_prompt, route_prompt_file  # noqa: E402


EXAMPLES = ROOT / "src" / "ahr_mvp" / "examples"


class AhrMvpRouterTests(unittest.TestCase):
    def test_routes_graph_prompt(self):
        result = route_prompt_file(EXAMPLES / "graph_two_triangles_unsat.txt")
        self.assertEqual(result["module"], "graph")
        self.assertEqual(result["parsed"]["vertices"], 6)
        self.assertEqual(result["parsed"]["diameter"], 3)
        self.assertTrue(result["parsed"]["edge_critical_connected"])
        self.assertEqual(result["parsed"]["triangles"], 2)
        self.assertEqual(result["result"]["checked"], 32768)
        self.assertEqual(result["result"]["conclusion"], "unsat_under_exhaustive_simple_graph_search")

    def test_routes_hanoi_prompt(self):
        result = route_prompt_file(EXAMPLES / "hanoi_5disk_after8_bcap2.txt")
        self.assertEqual(result["module"], "hanoi")
        self.assertEqual(result["parsed"]["disks"], 5)
        self.assertEqual(result["parsed"]["start"], "A")
        self.assertEqual(result["parsed"]["target"], "C")
        self.assertEqual(result["parsed"]["switch_after"], 8)
        self.assertEqual(result["parsed"]["capped_peg"], "B")
        self.assertEqual(result["parsed"]["cap"], 2)
        self.assertEqual(result["result"]["conclusion"], "unsat_under_bounded_hanoi_rule_revision_search")

    def test_routes_operation_prompt(self):
        result = route_prompt_file(EXAMPLES / "operation_len5_unsat.txt")
        self.assertEqual(result["module"], "operation_synthesis")
        self.assertEqual(result["parsed"]["start"], [1, 4, 9, 16, 25, 36])
        self.assertEqual(result["parsed"]["target"], [1, 6, 5, 14, 9])
        self.assertEqual(result["parsed"]["length"], 5)
        self.assertEqual(result["result"]["checked"], 1024)
        self.assertEqual(result["result"]["conclusion"], "unsat_under_declared_operation_semantics")

    def test_unsupported_prompt_is_explicit(self):
        result = route_prompt("What is beta of alpha after twenty unrelated problems?")
        self.assertEqual(result["module"], "unsupported")
        self.assertEqual(result["result"]["conclusion"], "unsupported_prompt_for_current_mvp")


if __name__ == "__main__":
    unittest.main()
