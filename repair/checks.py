from copy import deepcopy
from select_chunks import select_chunks

DATA = [
    {"id": "a", "chapter": 1, "score": 0.9, "text": "one two three four"},
    {"id": "b", "chapter": 2, "score": 1.0, "text": "five six"},
    {"id": "b", "chapter": 2, "score": 0.8, "text": "five six"},
    {"id": "c", "chapter": 2, "score": 0.9, "text": "seven eight nine"},
    {"id": "d", "chapter": 3, "score": 2.0, "text": "future"},
]

CASES = [
    ("skip_large", DATA, 2, 5, ["b", "c"]),
    ("exact_budget", DATA, 2, 6, ["b", "a"]),
    ("chapter_boundary", DATA, 1, 4, ["a"]),
    ("dedup_and_ties", DATA, 2, 50, ["b", "a", "c"]),
    ("empty", [], 2, 5, []),
    ("zero_budget", DATA, 2, 0, []),
]

failures = []
for name, rows, chapter, budget, expected in CASES:
    supplied = deepcopy(rows)
    before = deepcopy(supplied)
    try:
        result = select_chunks(supplied, chapter, budget)
        assert result == [next(r for r in rows if r["id"] == cid) for cid in expected], (result, expected)
        assert supplied == before, "input mutated"
    except Exception as exc:
        failures.append((name, repr(exc)))

for name, detail in failures:
    print("FAIL", name, detail)
if failures:
    raise SystemExit(1)
print("CHECKS_PASSED")
