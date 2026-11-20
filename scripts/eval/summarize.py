import json, glob, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(f)); A, B, C = r["A"], r["B"], r["C"]
    rows.append((r["spec"].split("|")[0], A["r5"], A["r10"], A["r50"], A["mrr"], B["r50"], B["mrr"], r["cpu_query_ms"], C["r5"], C["r50"], C["mrr"]))
rows.sort(key=lambda x: -x[10])
print("| model | A R@5 | A R@10 | A R@50 | A MRR | B R@50 | B MRR | C R@5 | C R@50 | C MRR | CPU ms/q |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.3f} | {r[5]:.2f} | {r[6]:.3f} | {r[8]:.2f} | {r[9]:.2f} | {r[10]:.3f} | {r[7]:.1f} |")
