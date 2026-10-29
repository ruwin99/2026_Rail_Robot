"""Python 3.10(Jetson)과 3.11(개발 PC)의 Future 타임아웃 예외 차이를 막는 구문 검사.

3.11부터 concurrent.futures.TimeoutError가 내장 TimeoutError의 별칭이지만 3.10에서는 별개 클래스다.
`except TimeoutError`만 있으면 개발 PC 시험은 통과하고 Jetson에서만 Future.result(timeout=...)의 예외가 새어 나간다.
그래서 timeout 인자를 쓰는 .result() 호출은 FutureTimeoutError(별칭 포함)를 잡거나 Exception으로 감싸야 한다."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
SCANNED = [p for p in JETSON_CODE.rglob("*.py") if "__pycache__" not in p.parts]


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    names: set[str] = set()
    nodes = []
    if isinstance(handler.type, ast.Tuple):
        nodes = list(handler.type.elts)
    elif handler.type is not None:
        nodes = [handler.type]
    else:
        names.add("BaseException")
    for node in nodes:
        names.add(node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", ""))
    return names


def _timed_result_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "result"
                and any(keyword.arg == "timeout" for keyword in node.keywords)):
            yield node


def _is_guarded(call: ast.Call, tries: list[ast.Try]) -> bool:
    for statement in tries:
        inside = any(call is node for body in statement.body for node in ast.walk(body))
        if not inside:
            continue
        for handler in statement.handlers:
            names = _handler_names(handler)
            if names & {"FutureTimeoutError", "Exception", "BaseException"}:
                return True
    return False


class FutureTimeoutHandlingTests(unittest.TestCase):
    def test_timed_future_results_handle_the_python_310_timeout_class(self):
        unguarded = []
        calls = 0
        for path in SCANNED:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
            for call in _timed_result_calls(tree):
                calls += 1
                if not _is_guarded(call, tries):
                    unguarded.append(f"{path.relative_to(JETSON_CODE)}:{call.lineno}")
        self.assertGreater(calls, 0, "검사 대상 호출을 찾지 못했다")
        self.assertEqual(
            unguarded, [],
            "Future.result(timeout=...)는 Python 3.10에서 내장 TimeoutError가 아닌 예외를 던진다. "
            "`from concurrent.futures import TimeoutError as FutureTimeoutError`로 함께 잡아라.",
        )

    def test_guard_detects_the_old_pattern(self):
        source = "try:\n    future.result(timeout=1)\nexcept TimeoutError:\n    pass\n"
        tree = ast.parse(source)
        tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        call = next(_timed_result_calls(tree))
        self.assertFalse(_is_guarded(call, tries))
        fixed = ast.parse(
            "try:\n    future.result(timeout=1)\nexcept (TimeoutError, FutureTimeoutError):\n    pass\n"
        )
        self.assertTrue(_is_guarded(next(_timed_result_calls(fixed)),
                                    [n for n in ast.walk(fixed) if isinstance(n, ast.Try)]))

    @unittest.skipUnless(sys.version_info < (3, 11), "3.11부터는 두 예외가 같은 클래스")
    def test_documented_difference_exists_on_this_interpreter(self):
        import concurrent.futures
        self.assertIsNot(concurrent.futures.TimeoutError, TimeoutError)


if __name__ == "__main__":
    unittest.main()
