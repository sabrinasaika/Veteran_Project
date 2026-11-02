"""LaTeX tables and figure of the RSA-911 privacy attack results for the paper.

Reads the aggregate attack metrics written by rsa911/privacy_attacks.py (one file
per per-query epsilon setting) and writes one .tex file with a re-identification
table, an attack results table, and a pgfplots figure of attack accuracy against
total privacy loss spent on the target. The paper preamble needs
\\usepackage{pgfplots} (and \\pgfplotsset{compat=1.17} or later).

Usage:
  python statistics/privacy_attack_tables.py      # every privacy_attack_results.json under RSA911_WORK_DIR
  python statistics/privacy_attack_tables.py --results a.json b.json --out statistics/paper/privacy_tables.tex
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsa911.config import work_dir  # noqa: E402
from rsa911_tables import num, pct, table, tex  # noqa: E402

# Categorical slots 1-3 (blue, orange, aqua), one per per-query epsilon setting.
# Slots 1 and 2 are checked for colour-vision deficiency separation.
SERIES = [("2A78D6", "*"), ("EB6834", "square*"), ("1BAF7A", "triangle*")]
NO_NOISE_X = 100  # where the suppression-only point sits on the log axis (epsilon = infinity)


def eps_label(summary: dict) -> str:
    return f"$\\varepsilon_q{{=}}{summary['dp_settings']['epsilon_per_query']:g}$"


def reid_table(summary: dict) -> str:
    rows = [[tex(name[:1].upper() + name[1:]), pct(u["share_unique"]), pct(u["share_in_class_below_5"]),
             pct(u["share_in_class_below_min_cell"])]
            for name, u in summary["reidentification_risk"].items()]
    return table("tab:reid", "Share of veteran exiters identifiable from quasi-identifiers an attacker might know. "
                 "Attributes are cumulative down the rows.",
                 "lrrr", ["Known attributes", "Unique", "Group $<5$", "Group $<11$"], rows)


def _dp_rows(summary: dict) -> list[tuple[dict, dict]]:
    pairs = zip(summary["membership_inference"], summary["attribute_inference"]["results"])
    return [(m, a) for m, a in pairs if m["epsilon_spent"] > 0]


def attack_table(summaries: list[dict]) -> str:
    first = summaries[0]
    exact_m = first["membership_inference"][0]
    exact_a = first["attribute_inference"]["results"][0]
    rows = [["No noise (suppression only)", "--", "--", num(exact_m["accuracy"], 3), num(exact_a["accuracy"], 3), "--"]]
    dp = sorted(((s, m, a) for s in summaries for m, a in _dp_rows(s)), key=lambda t: t[1]["epsilon_spent"])
    for s, m, a in dp:
        rows.append([f"DP, {eps_label(s)}", str(m["repeats"]), num(m["epsilon_spent"], 0),
                     num(m["accuracy"], 3), num(a["accuracy"], 3), num(m["dp_accuracy_bound"], 3)])
    caption = (f"Attack accuracy on {first['targets']} veterans unique on all quasi-identifiers, sorted by privacy loss. "
               "$\\varepsilon_q$ is the per-query budget, $k$ the number of repeated queries averaged, and "
               "$\\varepsilon$ spent the total privacy loss of the attack. Bound is the highest accuracy any attacker can "
               "achieve at that loss. Majority-class baseline for attribute inference: "
               f"{num(first['attribute_inference']['majority_class_baseline_accuracy'], 3)}. "
               "95\\% intervals are shown in Figure~\\ref{fig:attacks}.")
    return table("tab:attacks", caption, "lrrrrr",
                 ["Interface", "$k$", "$\\varepsilon$ spent", "Membership", "Attribute", "Bound"], rows)


def _points(rows: list[dict]) -> str:
    out = []
    for r in rows:
        lo, hi = r["accuracy_95ci"]
        acc = r["accuracy"]
        out.append(f"({r['epsilon_spent']:g},{acc}) += (0,{hi - acc:.4f}) -= (0,{acc - lo:.4f})")
    return " ".join(out)


def _panel(title: str, summaries: list[dict], pick, ref_y: float, ref_label: str, legend: bool) -> str:
    spent = sorted({m["epsilon_spent"] for s in summaries for m, _ in _dp_rows(s)})
    ticks = ",".join(f"{x:g}" for x in spent) + f",{NO_NOISE_X}"
    labels = ",".join(f"{x:g}" for x in spent) + ",$\\infty$"
    exact = pick(summaries[0], 0)["accuracy"]
    lines = [
        "\\begin{axis}[\n"
        f"  title={{{title}}}, title style={{font=\\small}},\n"
        "  width=0.48\\textwidth, height=0.34\\textwidth,\n"
        f"  xmode=log, xmin=0.7, xmax=150, xtick={{{ticks}}}, xticklabels={{{labels}}},\n"
        "  ymin=0.4, ymax=1.04, ytick={0.4,0.5,0.6,0.7,0.8,0.9,1.0},\n"
        "  xlabel={Total privacy loss spent on the target ($\\varepsilon$)}, ylabel={Attack accuracy},\n"
        "  ymajorgrids, grid style={gray!20}, axis lines*=left,\n"
        "  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
        + ("  legend to name=attacklegend, legend columns=-1,\n"
           "  legend style={font=\\footnotesize, draw=none, /tikz/every even column/.append style={column sep=8pt}},\n"
           if legend else "")
        + "]"
    ]
    for i, s in enumerate(summaries):
        color, mark = SERIES[i % len(SERIES)]
        rows = [r for r in (pick(s, j) for j in range(len(s["membership_inference"]))) if r["epsilon_spent"] > 0]
        lines.append(f"\\addplot[color={{rgb,255:red,{int(color[:2], 16)};green,{int(color[2:4], 16)};blue,{int(color[4:], 16)}}}, "
                     f"thick, mark={mark}, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {{{_points(rows)}}};")
        lines.append(f"\\addlegendentry{{DP, {eps_label(s)}}}")
    lines.append(f"\\addplot[black, only marks, mark=diamond*, mark size=3pt] coordinates {{({NO_NOISE_X},{exact})}};")
    lines.append("\\addlegendentry{No noise (suppression only)}")
    lines.append("\\addplot[gray, densely dashed, thick, domain=0.7:150, samples=100] {min(1, exp(x/2)/(1+exp(x/2)))};")
    lines.append("\\addlegendentry{Theoretical bound}")
    lines.append(f"\\addplot[gray!70, densely dotted, domain=0.7:150, forget plot] {{{ref_y}}};")
    lines.append(f"\\node[font=\\scriptsize, text=gray, anchor=south west] at (axis cs:0.75,{ref_y}) {{{ref_label}}};")
    lines.append("\\end{axis}")
    if not legend:
        lines = [line for line in lines if not line.startswith("\\addlegendentry")]
    return "\n".join(lines)


def accuracy_figure(summaries: list[dict]) -> str:
    baseline = summaries[0]["attribute_inference"]["majority_class_baseline_accuracy"]
    membership = _panel("Membership inference", summaries, lambda s, j: s["membership_inference"][j], 0.5, "chance", True)
    attribute = _panel("Attribute inference (employed at exit)", summaries,
                       lambda s, j: s["attribute_inference"]["results"][j], baseline, "majority-class baseline", False)
    return (
        "\\begin{figure*}[!htbp]\n\\centering\n"
        "\\ref{attacklegend}\\\\[4pt]\n"
        "\\begin{tikzpicture}\n" + membership + "\n\\end{tikzpicture}\\hfill\n"
        "\\begin{tikzpicture}\n" + attribute + "\n\\end{tikzpicture}\n"
        "\\caption{Attack accuracy against the total privacy loss spent on the target, with 95\\% Wilson intervals. "
        "$\\infty$ marks the interface without noise. Under differential privacy, results from both per-query budgets "
        "follow the same rising curve and stay below the theoretical bound: leakage is governed by the total loss spent, "
        "not by the noise added to each answer.}\n"
        "\\label{fig:attacks}\n\\end{figure*}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", nargs="+", help="attack result files (default: all under RSA911_WORK_DIR)")
    parser.add_argument("--out", default="statistics/paper/privacy_tables.tex")
    args = parser.parse_args()

    paths = [Path(p).expanduser() for p in args.results] if args.results else sorted(work_dir().glob("**/privacy_attack_results.json"))
    if not paths:
        sys.exit("No privacy_attack_results.json found; run python -m rsa911.privacy_attacks first.")
    summaries = sorted((json.loads(p.read_text()) for p in paths),
                       key=lambda s: s["dp_settings"]["epsilon_per_query"], reverse=True)
    parts = [reid_table(summaries[0]), attack_table(summaries), accuracy_figure(summaries)]
    out = Path(args.out)
    out.write_text("% Generated by statistics/privacy_attack_tables.py -- do not edit by hand.\n"
                   "% Requires \\usepackage{pgfplots} in the preamble.\n\n" + "\n".join(parts), encoding="utf-8")
    print(f"Wrote 2 tables and 1 figure from {len(summaries)} result file(s) to {out}")


if __name__ == "__main__":
    main()
