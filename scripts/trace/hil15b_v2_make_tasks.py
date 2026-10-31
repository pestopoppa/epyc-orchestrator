#!/usr/bin/env python3
"""PROPOSED ONLY: unexecuted HIL-15b v2 generator. No model/network imports.
No shared output path. Usage after granted review: script OUT_DIR
20 candidates: 16 calibration + 4 frozen heldout (never pilot/tune on heldout).
Native driver schema preserved; extra metadata ignored by current load_tasks.
"""
import json
import re
import sys
from pathlib import Path


def solve(family, data):
    if family == "lcg":
        x = data["x0"]
        for _ in range(data["n"]):
            x = (data["a"] * x + data["c"]) % data["m"]
        return str(x)
    if family == "ledger":
        balances = dict(data["balances"])
        rejected = 0
        for src, dst, amount in data["transfers"]:
            if balances[src] >= amount:
                balances[src] -= amount
                balances[dst] += amount
            else:
                rejected += 1
        return ",".join(str(balances[k]) for k in sorted(balances)) + ";" + str(rejected)
    if family == "dag":
        # All edges go from lower vertex index to higher: topological order given.
        distance = {0: 0}
        count = {0: 1}
        for node in range(data["vertices"]):
            if node not in distance:
                continue
            for src, dst, weight in data["edges"]:
                if src != node:
                    continue
                candidate = distance[src] + weight
                if dst not in distance or candidate < distance[dst]:
                    distance[dst], count[dst] = candidate, count[src]
                elif candidate == distance[dst]:
                    count[dst] += count[src]
        goal = data["vertices"] - 1
        return str(distance[goal]) + "," + str(count[goal])
    if family == "rewrite":
        s = data["text"]
        for _ in range(data["rounds"]):
            for old, new in data["rules"]:
                s = s.replace(old, new)
            s = s[data["rotate"]:] + s[:data["rotate"]]
        return s
    if family == "records":
        # Encounter order is authoritative; duplicate IDs overwrite earlier rows.
        latest = {}
        for row in data["rows"]:
            latest[row[0]] = row
        selected = [r for r in latest.values() if r[1] == data["group"] and r[2] >= data["threshold"]]
        return str(len(selected)) + "," + str(sum(r[2] * r[3] for r in selected))
    raise ValueError(family)


INSTRUCTIONS = {
    "lcg": "Compute x_n by x_(i+1)=(a*x_i+c) mod m, starting at x0. Answer the exact base-10 integer.",
    "ledger": "Process transfers [source,destination,amount] in listed order. Execute only if source has at least amount; otherwise reject without changing balances. Answer balances in ascending account-name order, comma separated, followed by semicolon and number rejected. No spaces.",
    "dag": "Edges are [source,destination,weight], directed; parallel edges are distinct. Find minimum total path weight from vertex 0 to vertex vertices-1, and number of distinct paths achieving it. Answer minimum,count with no spaces. Negative weights are allowed; this graph is acyclic.",
    "rewrite": "For each round apply each [old,new] rule in listed order to the entire current string, replacing left-to-right non-overlapping occurrences simultaneously per rule (new text is not rescanned by that same rule). After all rules, rotate the whole string left by rotate characters. Repeat rounds times. Answer the exact final ASCII string, case sensitive.",
    "records": "Rows are [id,group,value,multiplier]. Retain only the LAST row for each id in encounter order, then filter retained rows to group equal to group and value >= threshold. Answer count,sum(value*multiplier) with no spaces. Do not filter before deduplicating.",
}


def inputs():
    # First LCG is the existing lcg5-hard anchor with its exact input identity.
    yield "lcg5-hard", "lcg", dict(x0=12345, a=1103515245, c=12345, m=2**31, n=5), "calibration"
    for i in range(1, 4):
        yield f"h15bv2-lcg-{i}-hard", "lcg", dict(x0=37+i, a=48271, c=i*19, m=2147483647, n=7+i), "calibration"
    for i in range(4):
        split = "heldout" if i == 3 else "calibration"
        yield f"h15bv2-ledger-{i}-hard", "ledger", {
            "balances": [["A", 23+i], ["B", 11], ["C", 7]],
            "transfers": [["A","B",17], ["B","C",19], ["C","A",13], ["B","A",10],
                          ["A","C",25+i], ["C","B",7], ["B","A",12], ["A","B",16],
                          ["C","A",9], ["A","C",5], ["B","C",3+i], ["C","B",17]],
        }, split
        yield f"h15bv2-dag-{i}-hard", "dag", {
            "vertices": 9,
            "edges": [[0,1,3+i], [0,2,4], [0,3,9], [1,3,-2], [2,3,-1], [1,4,5],
                      [2,4,4], [3,4,3], [3,5,6], [4,5,1], [4,6,2], [5,6,1],
                      [5,7,4], [6,7,2-i], [6,8,6], [7,8,1], [2,6,9]],
        }, split
        yield f"h15bv2-rewrite-{i}-hard", "rewrite", {
            "text": ["ababaabbacaba", "cabaabbabaca", "bacaababbaab", "aabbacababba"][i],
            "rules": [["aba","bc"], ["bb","a"], ["ca","ab"], ["aa","cb"]],
            "rounds": 3+i, "rotate": 1,
        }, split
        yield f"h15bv2-records-{i}-hard", "records", {
            "group": "blue", "threshold": 7+i,
            "rows": [["u1","blue",9,3], ["u2","red",12,2], ["u3","blue",8,4],
                     ["u1","red",15,2], ["u4","blue",7,5], ["u2","blue",11,3],
                     ["u5","blue",16,2], ["u3","blue",6,9], ["u6","red",22,1],
                     ["u5","blue",10+i,4], ["u7","blue",8,3], ["u4","red",19,2],
                     ["u6","blue",7+i,6], ["u7","blue",9-i,5]],
        }, split


def build(scratch):
    tasks, warrants = [], []
    for key, family, data, split in inputs():
        answer = solve(family, data)
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        path = str(Path(scratch) / (key + ".json"))
        # Input is fully in prompt. No hidden files, environment/time or answer keys.
        prompt = (f"Use file_write_safe(path={path!r}, content=the following JSON text, backup=False) to save exactly this JSON; "
                  f"then use peek(n=10000, file_path={path!r}) to read that same file back. Solve from its contents. "
                  "Give only the final answer.\nJSON: " + payload + "\nTask: " + INSTRUCTIONS[family])
        tasks.append(dict(task_key=key, prompt=prompt, expect=r"\A\s*"+re.escape(answer)+r"\s*\Z",
                          mock_answer=answer, tool_required=True, candidate_split=split))
        warrants.append(dict(task_key=key, family=family, inputs=data, instruction=INSTRUCTIONS[family],
                             answer=answer, split=split, status="UNVALIDATED_NOT_EMPIRICAL"))
    assert len(tasks) == 20 and len({t["task_key"] for t in tasks}) == 20
    return tasks, warrants


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    tasks, warrants = build(str(out / "task-files"))
    for name, obj in (("hil15b_v2_tasks.json", tasks), ("hil15b_v2_warrants.json", warrants)):
        (out/name).write_text(json.dumps(obj, sort_keys=True, indent=2)+"\n")
    for split in ("calibration", "heldout"):
        (out/(split+"_tasks.json")).write_text(json.dumps([t for t in tasks if t["candidate_split"] == split], indent=2)+"\n")
