"""
Path Planning Assignment
Student : Kalpana Bhaskar | Roll No: 24011101049
Environment : Map 2
Algorithms  : E9 = Cuckoo Search (evolutionary)
              H9 = Bidirectional A* (heuristic)
Requirements: numpy (required), matplotlib (optional, for the plot)
Run         : python path_planning_kalpana.py
"""

import heapq
import math
import time
import numpy as np

# ============================ ENVIRONMENT ============================
GRID = 20
START = (1, 1)
GOAL = (18, 18)
SQRT2 = math.sqrt(2)
INF = float("inf")
RUNS = 10            # independent runs for the stochastic algorithm


def build_map2():
    """Obstacle cells for Map 2 (approximated from the handout figure)."""
    obs = set()

    def hline(y, x1, x2):
        for x in range(x1, x2 + 1):
            obs.add((x, y))

    def vline(x, y1, y2):
        for y in range(y1, y2 + 1):
            obs.add((x, y))

    hline(17, 2, 13)     # top bar
    vline(2, 6, 16)      # left bar
    vline(17, 8, 16)     # right bar
    hline(12, 8, 12)     # spiral: top
    vline(8, 7, 12)      # spiral: left
    hline(7, 8, 11)      # spiral: bottom
    vline(12, 9, 12)     # spiral: right
    hline(10, 10, 11)    # spiral: inner hook
    vline(10, 9, 10)     # spiral: inner hook
    hline(4, 5, 10)      # middle bar
    vline(15, 5, 7)      # small vertical bar
    hline(2, 7, 17)      # bottom bar
    return obs


def is_free(cell, obs):
    x, y = cell
    return 0 <= x < GRID and 0 <= y < GRID and cell not in obs


def segment_free(p, q, obs, step=0.1):
    """Check a straight segment by sampling points along it."""
    n = max(2, int(math.dist(p, q) / step))
    for i in range(n + 1):
        t = i / n
        x = p[0] + (q[0] - p[0]) * t
        y = p[1] + (q[1] - p[1]) * t
        if not is_free((int(round(x)), int(round(y))), obs):
            return False
    return True


def path_length(points):
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def count_turns(points, angle_thresh_deg=5.0):
    """Count interior vertices where the heading changes by more than the threshold."""
    turns = 0
    for i in range(1, len(points) - 1):
        a = math.atan2(points[i][1] - points[i - 1][1], points[i][0] - points[i - 1][0])
        b = math.atan2(points[i + 1][1] - points[i][1], points[i + 1][0] - points[i][0])
        d = abs(math.degrees(b - a)) % 360
        d = min(d, 360 - d)
        if d > angle_thresh_deg:
            turns += 1
    return turns


# ============================ E9: CUCKOO SEARCH ============================
N_WP = 6                 # intermediate waypoints per candidate path
DIM = 2 * N_WP
PENALTY = 50.0           # cost added per colliding segment


def decode(x):
    """Turn a flat vector into a path: START -> waypoints -> GOAL."""
    wps = [(float(np.clip(x[2 * i], 0, GRID - 1)),
            float(np.clip(x[2 * i + 1], 0, GRID - 1))) for i in range(N_WP)]
    return [START] + wps + [GOAL]


def evaluate(x, obs):
    """Fitness = path length + penalty * (number of colliding segments)."""
    pts = decode(x)
    bad = sum(1 for i in range(len(pts) - 1)
              if not segment_free(pts[i], pts[i + 1], obs))
    return path_length(pts) + PENALTY * bad, bad


def levy(size, beta=1.5):
    """Lévy-distributed steps (Mantegna's algorithm)."""
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    return u / np.abs(v) ** (1 / beta)


def cuckoo_search(obs, n_nests=25, max_iter=300, pa=0.25, alpha=0.1):
    lo, hi = 0.0, GRID - 1.0
    nests = np.random.uniform(lo, hi, (n_nests, DIM))
    fit = np.array([evaluate(n, obs)[0] for n in nests])
    b = int(np.argmin(fit))
    best, best_f = nests[b].copy(), fit[b]

    for _ in range(max_iter):
        # (a) Lévy flight: each nest generates a new candidate
        for i in range(n_nests):
            new = np.clip(nests[i] + alpha * levy(DIM) * (nests[i] - best), lo, hi)
            f_new = evaluate(new, obs)[0]
            j = np.random.randint(n_nests)          # compare with a random nest
            if f_new < fit[j]:
                nests[j], fit[j] = new, f_new

        # (b) Abandon a fraction pa of nests and rebuild them randomly
        abandon = np.random.rand(n_nests) < pa
        for i in np.where(abandon)[0]:
            nests[i] = np.random.uniform(lo, hi, DIM)
            fit[i] = evaluate(nests[i], obs)[0]

        # (c) Update the global best
        b = int(np.argmin(fit))
        if fit[b] < best_f:
            best, best_f = nests[b].copy(), fit[b]

    return best


# ============================ H9: BIDIRECTIONAL A* ============================
NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def bidirectional_astar(obs):
    gF, gB = {START: 0.0}, {GOAL: 0.0}
    pF, pB = {START: None}, {GOAL: None}
    openF = [(octile(START, GOAL), 0.0, START)]
    openB = [(octile(GOAL, START), 0.0, GOAL)]
    closedF, closedB = set(), set()
    mu, meet, expansions = INF, None, 0

    def expand(heap, g, par, closed, g_other, target):
        nonlocal mu, meet, expansions
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            return
        closed.add(cur)
        expansions += 1
        for dx, dy in NEIGH:
            nb = (cur[0] + dx, cur[1] + dy)
            if not is_free(nb, obs):
                continue
            # no corner cutting on diagonal moves
            if dx and dy and not (is_free((cur[0] + dx, cur[1]), obs) and
                                  is_free((cur[0], cur[1] + dy), obs)):
                continue
            ng = g[cur] + (SQRT2 if dx and dy else 1.0)
            if ng < g.get(nb, INF):
                g[nb] = ng
                par[nb] = cur
                heapq.heappush(heap, (ng + octile(nb, target), ng, nb))
            if nb in g_other and g[nb] + g_other[nb] < mu:
                mu, meet = g[nb] + g_other[nb], nb

    # Alternate: always expand the side with the smaller open list
    while openF and openB:
        if min(openF[0][0], openB[0][0]) >= mu:   # stopping rule
            break
        if len(openF) <= len(openB):
            expand(openF, gF, pF, closedF, gB, GOAL)
        else:
            expand(openB, gB, pB, closedB, gF, START)

    if meet is None:
        return None, expansions

    fwd, n = [], meet
    while n is not None:
        fwd.append(n)
        n = pF[n]
    fwd.reverse()                      # START ... meet

    bwd, n = [], meet
    while n is not None:
        bwd.append(n)
        n = pB[n]                      # meet ... GOAL

    return fwd + bwd[1:], expansions


# ============================ MAIN ============================
def main():
    obs = build_map2()
    print("=" * 62)
    print("Kalpana Bhaskar | 24011101049 | Map 2 | E9 (CS) & H9 (Bi-A*)")
    print("=" * 62)

    # ---- H9: Bidirectional A* (deterministic, one run) ----
    t0 = time.perf_counter()
    path_h, exp_h = bidirectional_astar(obs)
    t_h = time.perf_counter() - t0
    ok_h = (path_h is not None and path_h[0] == START and path_h[-1] == GOAL
            and all(is_free(p, obs) for p in path_h))
    print("\n[H9] Bidirectional A*")
    if ok_h:
        print(f"  Success          : Yes")
        print(f"  Path length      : {path_length(path_h):.3f}")
        print(f"  Computation time : {t_h * 1000:.2f} ms")
        print(f"  Nodes expanded   : {exp_h}")
        print(f"  Turns            : {count_turns(path_h)}")
        print(f"  Collision-free   : Yes")
    else:
        print("  No path found.")

    # ---- E9: Cuckoo Search (stochastic, RUNS runs) ----
    successes, lengths, times, turns = 0, [], [], []
    best_pts, best_len = None, INF
    for r in range(RUNS):
        np.random.seed(r)
        t0 = time.perf_counter()
        x = cuckoo_search(obs)
        times.append(time.perf_counter() - t0)

        pts = decode(x)
        _, bad = evaluate(x, obs)
        if bad == 0:
            successes += 1
            L = path_length(pts)
            lengths.append(L)
            turns.append(count_turns(pts))
            if L < best_len:
                best_len, best_pts = L, pts

    print(f"\n[E9] Cuckoo Search  ({RUNS} runs)")
    print(f"  Success rate     : {100 * successes / RUNS:.1f} %")
    if lengths:
        print(f"  Best path length : {min(lengths):.3f}")
        print(f"  Mean path length : {np.mean(lengths):.3f}  (successful runs)")
        print(f"  Mean turns       : {np.mean(turns):.1f}  (successful runs)")
    else:
        print("  No collision-free path found in any run. Try more iterations or nests.")
    print(f"  Mean time        : {np.mean(times):.3f} s per run")
    print(f"  Collision-free   : {'Yes' if successes else 'No'} (in successful runs)")

    # ---- Optional plot ----
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(6, 6))
        for (ox, oy) in obs:
            ax.add_patch(Rectangle((ox - 0.5, oy - 0.5), 1, 1, color="black"))
        if path_h:
            ax.plot([p[0] for p in path_h], [p[1] for p in path_h],
                    "b-o", ms=3, label="H9 Bi-A*")
        if best_pts:
            ax.plot([p[0] for p in best_pts], [p[1] for p in best_pts],
                    "m-o", ms=3, label="E9 Cuckoo (best)")
        ax.scatter(*START, c="green", s=120, zorder=5)
        ax.scatter(*GOAL, c="red", marker="*", s=200, zorder=5)
        ax.set_xlim(-0.5, GRID - 0.5)
        ax.set_ylim(-0.5, GRID - 0.5)
        ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
        ax.grid(True, alpha=0.3)
        ax.set_title("Map 2: paths")
        ax.legend()
        plt.savefig("map2_paths.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to map2_paths.png")
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()