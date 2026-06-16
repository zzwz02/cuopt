#!/usr/bin/env python3
"""Aggregate /tmp/perf_{c500,a100,cudss}.csv into docs/dev/performance.md.
Columns per build: orig = cuOpt's printed solve time; metric = PDLP solve-phase (m1)
or barrier-iterations-total (m3). Degrades to whatever CSVs are present."""
import csv, os
BUILDS = [("c500", "C500 (in-house LDLᵀ / MACA)"),
          ("a100", "A100 (in-house LDLᵀ)"),
          ("cudss", "A100 (original cuDSS)")]
LP = "graph40-40 ex10 datt256_lp woodlands09 savsched1 nug08-3rd qap15 scpm1 neos3 a2864 ns1687037 square41".split()

data, present = {}, []
for b, _ in BUILDS:
    p = f"/tmp/perf_{b}.csv"
    if not os.path.exists(p):
        continue
    present.append(b)
    with open(p) as fh:
        for r in csv.DictReader(fh):
            data[(b, r["set"], r["instance"], r["method"])] = r

def cell(b, sset, inst, method):
    r = data.get((b, sset, inst, method))
    if not r:
        return " | "
    o = r["orig_time_s"] or "–"
    m = r["metric_s"] or "–"
    st = r["status"]
    tag = "" if st in ("Optimal", "") else f" ({st})"
    return f"{o}{tag} | {m}"

def table(sset, method, instances, metric_name):
    hdr = "| instance |" + "".join(f" {lbl} time | {metric_name} |" for b, lbl in BUILDS if b in present)
    sep = "|---|" + "".join("---|---|" for b in present)
    lines = [hdr, sep]
    for inst in instances:
        row = f"| {inst} |"
        for b, _ in BUILDS:
            if b in present:
                row += " " + cell(b, sset, inst, method) + " |"
        lines.append(row)
    return "\n".join(lines)

qp = sorted({k[2] for k in data if k[1] == "maros_meszaros"})
out = ["# cuOpt LP/QP performance: C500 vs A100 vs original cuDSS",
       "",
       f"> Builds present: {', '.join(present)}. All runs `CUDA_MODULE_LOADING=EAGER`; "
       "Mittelmann `--time-limit 600`, Maros-Mészáros `--time-limit 180`. "
       "`time` = cuOpt's printed solve time; for PDLP (method 1) the phase metric is the "
       "**PDLP solve-phase** time (excl. presolve/build), for barrier (method 3) the "
       "**barrier-iterations-total** time (excl. setup/factorization). Non-Optimal status annotated.",
       "",
       "## 1. Mittelmann LP — method 1 (PDLP)",
       "", "phase metric = PDLP solve-phase time", "",
       table("mittelmann", "1", LP, "PDLP-phase"),
       "",
       "## 2. Mittelmann LP — method 3 (barrier)",
       "", "phase metric = barrier-iterations-total time", "",
       table("mittelmann", "3", LP, "barrier-iters"),
       "",
       f"## 3. Maros-Mészáros QP — method 3 (barrier), full {len(qp)} set",
       "", "phase metric = barrier-iterations-total time", "",
       table("maros_meszaros", "3", qp, "barrier-iters"),
       ""]
with open("/home/cuopt-26.06-maca/docs/dev/performance.md", "w") as fh:
    fh.write("\n".join(out))
print(f"wrote performance.md; builds={present}; QP={len(qp)}; LP={len(LP)}")
