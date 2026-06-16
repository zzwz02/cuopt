#!/usr/bin/env python3
"""Benchmark cuopt_cli timing: original printed time + new phase metric.
Usage: perf_bench.py <cli_path> <tag> <out_csv>
  Mittelmann LP (12) x method 1 (PDLP) & 3 (barrier); Maros-Meszaros (138) x method 3.
Captures: status, iters, original printed time, and the new phase metric
  method 1 -> "PDLP solve phase time"   method 3 -> "Barrier iterations total time".
CUDA_MODULE_LOADING=EAGER is forced. Writes one CSV row per run (incremental).
"""
import subprocess, re, sys, os, csv

CLI, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LP_DIR = "/home/cuopt-26.06-maca/datasets/linear_programming"
QP_DIR = "/home/cuopt-26.06-maca/datasets/benchmarks/maros_meszaros"
LP = "graph40-40 ex10 datt256_lp woodlands09 savsched1 nug08-3rd qap15 scpm1 neos3 a2864 ns1687037 square41".split()
ENV = dict(os.environ, CUDA_MODULE_LOADING="EAGER")  # caller sets CUDA_VISIBLE_DEVICES (A100) or leaves it (C500/MACA)

def run(path, method, tl):
    try:
        p = subprocess.run([CLI, path, "--method", str(method), "--time-limit", str(tl)],
                           capture_output=True, text=True, env=ENV, timeout=tl + 180)
        return p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        return (e.stdout or "") + "\n[HARNESS_TIMEOUT]"

def parse(out, method):
    status = iters = orig = metric = ""
    if method == 1:
        m = re.search(r'Status:\s*(\w+)', out);                         status = m.group(1) if m else ""
        m = re.search(r'Iterations:\s*(\d+).*?Time:\s*([\d.]+)s', out)
        if m: iters, orig = m.group(1), m.group(2)
        m = re.search(r'PDLP solve phase time[^:]*:\s*([\d.]+)s', out);  metric = m.group(1) if m else ""
    else:
        m = re.search(r'(Optimal|Suboptimal) solution found in (\d+) iterations and ([\d.]+)\s*(?:s|seconds)', out)
        if m: status, iters, orig = m.group(1), m.group(2), m.group(3)
        m = re.search(r'Barrier iterations total time:\s*([\d.]+)s', out); metric = m.group(1) if m else ""
        if not status:
            if re.search(r'[Nn]umerical', out):            status = "NumError"
            elif re.search(r'time limit', out, re.I):       status = "TimeLimit"
            elif "[HARNESS_TIMEOUT]" in out:                status = "HangTimeout"
    return status, orig, metric, iters

def main():
    fh = open(OUT, "w", newline="")
    w = csv.writer(fh); w.writerow(["build", "set", "instance", "method", "status", "orig_time_s", "metric_s", "iters"]); fh.flush()
    jobs = [("mittelmann", f"{LP_DIR}/{n}/{n}.mps", n, m, 600) for n in LP for m in (1, 3)]
    jobs += [("maros_meszaros", f"{QP_DIR}/{fn}", fn[:-4], 3, 180)
             for fn in sorted(os.listdir(QP_DIR)) if fn.upper().endswith(".QPS")]
    for i, (sset, path, name, method, tl) in enumerate(jobs, 1):
        st, orig, metric, iters = parse(run(path, method, tl), method)
        w.writerow([TAG, sset, name, method, st, orig, metric, iters]); fh.flush()
        print(f"[{TAG} {i}/{len(jobs)}] {sset}/{name} m{method}: {st} orig={orig} metric={metric} it={iters}", file=sys.stderr)
    fh.close()
    print(f"DONE {TAG}: {len(jobs)} runs -> {OUT}")

main()
