import sys, math, time
import numpy as np, cudf
from cuopt import routing


def euclid_matrix(xy, rounded):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    return (np.round(d) if rounded else d).astype(np.float32)


def parse_cvrp(path):
    """TSPLIB CVRP (.vrp). Returns (D, demand, capacity, depot_idx)."""
    coords, demand, cap, depot = {}, {}, None, 1
    section = None
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("CAPACITY"):
            cap = int(s.split(":")[1])
        elif u.startswith("NODE_COORD_SECTION"):
            section = "coord"
        elif u.startswith("DEMAND_SECTION"):
            section = "demand"
        elif u.startswith("DEPOT_SECTION"):
            section = "depot"
        elif u.startswith("EOF") or u.startswith("EDGE_WEIGHT_SECTION"):
            section = None
        elif section == "coord":
            p = s.split()
            coords[int(p[0])] = (float(p[1]), float(p[2]))
        elif section == "demand":
            p = s.split()
            demand[int(p[0])] = int(p[1])
        elif section == "depot":
            v = int(s.split()[0])
            if v > 0:
                depot = v
    ids = sorted(coords)
    xy = np.array([coords[i] for i in ids])
    dem = np.array([demand[i] for i in ids], dtype=np.int32)
    depot_idx = ids.index(depot)
    # reorder depot to index 0
    order = [depot_idx] + [i for i in range(len(ids)) if i != depot_idx]
    xy, dem = xy[order], dem[order]
    return euclid_matrix(xy, rounded=True), dem, cap, 0


def parse_solomon(path):
    """Solomon / Homberger CVRPTW. Returns (D, demand, capacity, n_veh, earliest, latest, service)."""
    lines = [l.rstrip("\n") for l in open(path)]
    n_veh = cap = None
    rows = []
    i = 0
    while i < len(lines):
        u = lines[i].strip().upper()
        if u.startswith("NUMBER"):
            p = lines[i + 1].split()
            n_veh, cap = int(p[0]), int(p[1])
            i += 2
            continue
        if u.startswith("CUST NO"):
            i += 1
            while i < len(lines):
                p = lines[i].split()
                if len(p) >= 7:
                    rows.append([float(x) for x in p[:7]])
                i += 1
            break
        i += 1
    arr = np.array(rows)  # cols: id x y demand ready due service
    xy = arr[:, 1:3]
    dem = arr[:, 3].astype(np.int32)
    earliest = arr[:, 4].astype(np.int32)
    latest = arr[:, 5].astype(np.int32)
    service = arr[:, 6].astype(np.int32)
    return euclid_matrix(xy, rounded=False), dem, cap, n_veh, earliest, latest, service


def solve(name, kind, path, n_veh, budget):
    if kind == "cvrp":
        D, dem, cap, _ = parse_cvrp(path)
        n = D.shape[0]
        dm = routing.DataModel(n, n_veh)
        dm.add_cost_matrix(cudf.DataFrame(D))
        dm.add_capacity_dimension("demand", cudf.Series(dem), cudf.Series([cap] * n_veh))
    else:
        D, dem, cap, file_veh, e, l, sv = parse_solomon(path)
        n = D.shape[0]
        n_veh = n_veh or file_veh
        dm = routing.DataModel(n, n_veh)
        dm.add_cost_matrix(cudf.DataFrame(D))
        dm.add_capacity_dimension("demand", cudf.Series(dem), cudf.Series([cap] * n_veh))
        dm.set_order_time_windows(cudf.Series(e), cudf.Series(l))
        dm.set_order_service_times(cudf.Series(sv))
    s = routing.SolverSettings()
    s.set_time_limit(budget)
    t0 = time.time()
    sol = routing.Solve(dm, s)
    dt = time.time() - t0
    print("BENCH name=%s n=%d veh=%d budget=%ds -> status=%s vehicles=%s cost=%.2f wall=%.1fs"
          % (name, n, n_veh, budget, sol.get_status(), sol.get_vehicle_count(),
             sol.get_total_objective(), dt), flush=True)


if __name__ == "__main__":
    # args: name kind path n_veh budget
    solve(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
