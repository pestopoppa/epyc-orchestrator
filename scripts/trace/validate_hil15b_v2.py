#!/usr/bin/env python3
"""PROPOSED independent model-free validator; no import of candidate solver.
Run only in canonical grant after GNX review. Reads JSON emitted by generator.
"""
import json
import re
import sys
from pathlib import Path


def oracle(family, d):
    if family == "lcg":
        # Exponentiate affine maps rather than iterating the recurrence.
        a, b, ra, rb, n, m = d["a"], d["c"], 1, 0, d["n"], d["m"]
        while n:
            if n & 1:
                ra, rb = a*ra % m, (a*rb+b) % m
            a, b = a*a % m, (a*b+b) % m
            n //= 2
        return str((ra*d["x0"]+rb) % m)
    if family == "ledger":
        initial = dict(d["balances"])
        accepted = []
        rejected = 0
        def balance(k):
            return initial[k] + sum(n for s,t,n in accepted if t == k) - sum(n for s,t,n in accepted if s == k)
        for s,t,n in d["transfers"]:
            if balance(s) < n:
                rejected += 1
            else:
                accepted.append((s,t,n))
        return ",".join(str(balance(k)) for k in sorted(initial)) + ";" + str(rejected)
    if family == "dag":
        # Enumerate every route; independent of topological shortest-path DP.
        weights = []
        def walk(v, weight):
            if v == d["vertices"]-1:
                weights.append(weight)
            for s,t,w in d["edges"]:
                if s == v:
                    assert s < t
                    walk(t, weight+w)
        walk(0, 0)
        best = min(weights)
        return str(best)+","+str(weights.count(best))
    if family == "rewrite":
        chars = list(d["text"])
        for _ in range(d["rounds"]):
            for old,new in d["rules"]:
                out, i = [], 0
                while i < len(chars):
                    if chars[i:i+len(old)] == list(old):
                        out.extend(new)
                        i += len(old)
                    else:
                        out.append(chars[i])
                        i += 1
                chars = out
            for _ in range(d["rotate"]):
                if chars:
                    chars.append(chars.pop(0))
        return "".join(chars)
    if family == "records":
        seen, selected = set(), []
        for row in reversed(d["rows"]):
            key, group, value, multiplier = row
            if key in seen:
                continue
            seen.add(key)
            if group == d["group"] and value >= d["threshold"]:
                selected.append(value*multiplier)
        return str(len(selected))+","+str(sum(selected))
    raise AssertionError(family)


def main(out):
    tasks = json.loads((out/"hil15b_v2_tasks.json").read_text())
    warrants = json.loads((out/"hil15b_v2_warrants.json").read_text())
    by_key = {w["task_key"]: w for w in warrants}
    assert len(tasks) == len(by_key) == 20
    assert sum(t["candidate_split"] == "calibration" for t in tasks) == 16
    assert sum(t["candidate_split"] == "heldout" for t in tasks) == 4
    for t in tasks:
        # Verify exactly the presented input, rather than trusting solver metadata.
        raw = t["prompt"].split("\nJSON: ",1)[1].split("\nTask: ",1)
        d, instruction = json.loads(raw[0]), raw[1]
        w = by_key[t["task_key"]]
        assert d == w["inputs"] and instruction == w["instruction"]
        expected = oracle(w["family"], d)
        assert expected == w["answer"] == t["mock_answer"]
        assert t["tool_required"] is True
        assert re.search(t["expect"], expected)
        assert re.search(t["expect"], " \n"+expected+"\n ")
        # Exact final-only answer: no embedded correct-answer false positives.
        for bad in ("", "wrong", "answer: "+expected, expected+"x", "x"+expected):
            assert not re.search(t["expect"], bad), (t["task_key"], bad)
        if expected.isdigit():
            assert not re.search(t["expect"], str(int(expected)+1))
        elif expected:
            assert not re.search(t["expect"], expected[:-1]+("z" if expected[-1] != "z" else "y"))
    # Tiny semantic fixtures independent of candidate data catch oracle mistakes.
    assert oracle("ledger", {"balances":[["a",2],["b",0]],"transfers":[["a","b",3],["a","b",2]]}) == "0,2;1"
    assert oracle("dag", {"vertices":2,"edges":[[0,1,3],[0,1,3],[0,1,4]]}) == "3,2"
    assert oracle("rewrite", {"text":"aaa","rules":[["aa","b"]],"rounds":1,"rotate":1}) == "ab"
    assert oracle("records", {"rows":[["a","x",9,2],["a","y",10,2]],"group":"x","threshold":0}) == "0,0"
    assert oracle("lcg", {"x0":5,"a":7,"c":3,"m":11,"n":0}) == "5"
    print("20 independent answer/input/checker comparisons and 5 semantic fixtures passed; no model/difficulty claim")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
