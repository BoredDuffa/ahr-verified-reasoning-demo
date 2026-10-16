import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ahr_mvp"))

from graph_csp import search_graph, verify_graph  # noqa: E402
from hanoi import solve_rule_revision, verify_hanoi_solution  # noqa: E402


class GraphCspTests(unittest.TestCase):
    def test_verify_path_graph(self):
        result = verify_graph(
            vertices=4,
            edges=[[0, 1], [1, 2], [2, 3]],
            diameter=3,
            edge_critical_connected=True,
            triangles=0,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["checks"]["diameter"], 3)

    def test_user_graph_example_is_unsat(self):
        result = search_graph(vertices=6, diameter=3, edge_critical_connected=True, triangles=2)
        self.assertFalse(result.passed)
        self.assertEqual(result.checked, 32768)
        self.assertEqual(result.conclusion, "unsat_under_exhaustive_simple_graph_search")


class HanoiRevisionTests(unittest.TestCase):
    def test_hanoi_three_disks_solves(self):
        result = solve_rule_revision(disks=3, switch_after=2, capped_peg="B", cap=2)
        self.assertTrue(result.passed)
        verification = verify_hanoi_solution(
            disks=3,
            moves=result.moves or (),
            switch_after=2,
            capped_peg="B",
            cap=2,
        )
        self.assertTrue(verification["valid"])

    def test_hanoi_five_disk_user_shape_solves_or_proves(self):
        result = solve_rule_revision(disks=5, switch_after=8, capped_peg="B", cap=2)
        self.assertIn(result.conclusion, {"sat", "unsat_under_bounded_hanoi_rule_revision_search"})
        if result.moves is not None:
            self.assertGreaterEqual(len(result.moves), 8)
            verification = verify_hanoi_solution(
                disks=5,
                moves=result.moves,
                switch_after=8,
                capped_peg="B",
                cap=2,
            )
            self.assertTrue(verification["valid"])


if __name__ == "__main__":
    unittest.main()
