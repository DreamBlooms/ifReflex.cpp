"""Emit expected readout values from the reflex reference implementation.

Run with a Python that has numpy and the reflex checkout on sys.path, e.g.:

    python3 scripts/parity/ref_readout.py > /tmp/expected.txt

The numbers are compared against scripts/parity/test_readout.cpp. The readout
math (softmax, normalised-entropy confidence, score expectation, permutation
merge, and the 8-weight calibration head) is ported from
kshetrajna12/reflex src/reflex/readout.py and src/reflex/calibration_head.py.
"""
import math
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required to generate reference values")

REFLEX_SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/port/reflex/src"
sys.path.insert(0, REFLEX_SRC)

from reflex.calibration_head import features, temperature as head_temperature  # noqa: E402
from reflex.readout import softmax, confidence  # noqa: E402
from reflex.schema import ScoreQuestion  # noqa: E402
from reflex.readout import to_answer  # noqa: E402

KINDS = ["noul", "choice", "score"]
LOGITS = [
    [0.5, -1.25, 2.0, 0.0],
    [1.75, 0.25, -0.5, 3.25, -2.0],
    [-0.1, 0.2, -0.3],
]
# float32 exact values (what the C++ test uses as float inputs)
LOGITS32 = [[float(np.float32(v)) for v in row] for row in LOGITS]


def emit(tag, values):
    print(tag + " " + " ".join("%.12f" % v for v in values))


print("SOFTMAX")
for row in LOGITS:
    emit("", softmax(np.asarray(row, dtype=np.float64), 1.0))

print("SOFTMAX_T")
for row, t in zip(LOGITS, [0.7, 1.5, 2.25]):
    emit("", softmax(np.asarray(row, dtype=np.float64), t))

print("CONFIDENCE")
for row in LOGITS:
    print("%.12f" % confidence(softmax(np.asarray(row, dtype=np.float64), 1.0)))

W = [0.03, -0.11, 0.07, 0.02, 0.4, -0.25, 1.1, -0.6]
print("FEATURES")
for kind, row, st in zip(KINDS, LOGITS32, [17, 200, 1]):
    f = features(kind, np.asarray(row), st)
    emit("", f)
    print("temp %.12f" % head_temperature(np.array(W), f))

print("MERGE")
# two permutation branches of a 3-option choice, temperature 1


def merge(keys_per_branch, logits_per_branch, temp=1.0):
    acc = {}
    for keys in keys_per_branch:
        for k in keys:
            acc[k] = 0.0
    for keys, logits in zip(keys_per_branch, logits_per_branch):
        p = softmax(np.asarray(logits, dtype=np.float64), temp)
        for k, pr in zip(keys, p):
            acc[k] += pr
    n = len(keys_per_branch)
    total = sum(acc.values())
    return {k: (v / n) / (total / n) for k, v in acc.items()}


m = merge(
    [["payments", "account", "other"], ["other", "payments", "account"]],
    [[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
)
emit("", [m["payments"], m["account"], m["other"]])

print("ANSWER")
a = to_answer("choice", {"payments": 0.5, "account": 0.3, "other": 0.2}, None)
print("choice %s %.12f" % (a.choice, a.confidence))
sq = ScoreQuestion(type="score", instructions="How urgent?", criteria=["low", "mid", "high"])
s = to_answer("score", {0: 0.1, 1: 0.7, 2: 0.2}, sq)
print("score %.12f %.12f" % (s.score, s.confidence))
n = to_answer("noul", {True: 0.72, False: 0.28}, None)
print("noul %.12f" % n.noul)
