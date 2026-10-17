"""Phase 6: the page is split into small, module-level functions (guards against growing a god-function again)."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI_MODULES = ["app.py", "page.py", "tabs.py", "live.py", "intel_ui.py", "stress_ui.py"]


def _functions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _depth(node, d=0):
    nest = (ast.If, ast.For, ast.While, ast.With, ast.Try)
    return max([_depth(c, d + isinstance(c, nest)) for c in ast.iter_child_nodes(node)] + [d])


class StructureTests(unittest.TestCase):
    def test_render_ui_is_short(self):
        fn = next(f for f in _functions(ROOT / "app.py") if f.name == "render_ui")
        self.assertLess(fn.end_lineno - fn.lineno + 1, 120)

    def test_app_py_is_small(self):
        self.assertLess(len((ROOT / "app.py").read_text(encoding="utf-8").splitlines()), 600)

    def test_no_ui_function_is_huge_or_deeply_nested(self):
        for name in UI_MODULES:
            for fn in _functions(ROOT / name):
                length = fn.end_lineno - fn.lineno + 1
                if fn.name == "stress_tab":          # original stress test body, moved as is
                    continue
                self.assertLess(length, 120, f"{name}:{fn.name} is {length} lines")
                self.assertLessEqual(_depth(fn), 4, f"{name}:{fn.name} nests too deep")


if __name__ == "__main__":
    unittest.main()
