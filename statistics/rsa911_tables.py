"""LaTeX tables of RSA-911 statistical findings for the paper.

Reads the privacy-safe summaries written by rsa911/private_aggregate.py,
rsa911/analysis.py and rsa911/statistical_analysis.py and writes one .tex
file of tables. Cells suppressed upstream (None) are shown as "--".

Usage:
  python statistics/rsa911_tables.py                       # reads RSA911_WORK_DIR
  python statistics/rsa911_tables.py --summary-dir data/rsa911_private --out statistics/paper/rsa911_tables.tex
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsa911.config import work_dir  # noqa: E402

TOP_N = 10


def tex(text) -> str:
    text = str(text)
    for char, repl in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                       ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        text = text.replace(char, repl)
    return text


def num(value, digits: int = 2) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def pct(value) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def table(label: str, caption: str, columns: str, header: list[str], rows: list[list[str]]) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows) or r"\multicolumn{%d}{c}{(no reportable rows)} \\" % len(header)
    return (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{columns}}}\n\\hline\n"
        + " & ".join(header) + " \\\\\n\\hline\n"
        + body + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    )


def descriptive_table(stat: dict, agg: dict) -> str:
    desc = stat["descriptive_statistics"]
    rows = [
        ["Veteran exiters", num(desc["veteran_records"])],
        ["Employed at exit (codes 1--5)", num(desc["employed_count"])],
        ["Employment rate", pct(desc["employment_rate_blank_as_not_employed"])],
    ]
    for key, label in [("exit_hourly_wage_positive_stats", "Hourly wage at exit (\\$, $>0$)"),
                       ("exit_weekly_hours_positive_stats", "Weekly hours at exit ($>0$)")]:
        s = agg.get(key)
        if s and s.get("p10") is not None and s.get("p90") is not None:
            rows.append([f"{label}, mean [10th, 90th percentile]", f"{num(s['mean'])} [{num(s['p10'])}, {num(s['p90'])}] ($n={num(s['n'])}$)"])
        elif s:
            rows.append([f"{label}, mean", f"{num(s['mean'])} ($n={num(s['n'])}$)"])
    return table("tab:rsa911-descriptive", "Descriptive statistics for veteran exiters, RSA-911 PY2021--PY2023.",
                 "ll", ["Measure", "Value"], rows)


def outcome_table(agg: dict) -> str:
    rows = [[tex(r["label"]), num(r["n"]), pct(r["percent"])] for r in agg.get("employment_outcome_exit_counts", [])]
    return table("tab:rsa911-outcomes", "Employment outcome at exit among veteran exiters (blank outcomes excluded from the denominator).",
                 "lrr", ["Exit outcome", "$n$", "\\%"], rows)


def disability_table(agg: dict) -> str:
    rows = []
    groups = sorted(agg.get("key_disability_group_summaries", {}).values(), key=lambda g: g["n"], reverse=True)
    for g in groups:
        wage = (g.get("exit_hourly_wage_positive_stats") or {}).get("mean")
        rows.append([tex(g["label"]), num(g["n"]), pct(g["employment_rate_blank_as_not_employed"]), num(wage)])
    return table("tab:rsa911-disability", "Employment rate at exit by disability group (primary or secondary). Groups overlap.",
                 "lrrr", ["Disability group", "$n$", "Employed", "Mean wage (\\$)"], rows)


def categorical_table(stat: dict) -> str:
    scores = stat["feature_discriminant_analysis"]["categorical_feature_scores"][:TOP_N]
    rows = []
    for s in scores:
        groups = s["reportable_groups"]
        rates = [g["employment_rate"] for g in groups if g["employment_rate"] is not None]
        spread = f"{pct(min(rates))}--{pct(max(rates))}" if rates else "--"
        rows.append([tex(s["feature"]), num(s["score"], 3), spread])
    return table("tab:rsa911-cramer", f"Top {TOP_N} categorical variables by association with employment at exit (Cram\\'er's $V$). "
                 "The last column gives the range of employment rates across reportable categories.",
                 "lrr", ["Variable", "Cram\\'er's $V$", "Employment rate range"], rows)


def numeric_table(stat: dict) -> str:
    rows = [[tex(s["feature"]), num(s["n"]), num(s["correlation_with_employed"], 3),
             num(s["employed_mean"]), num(s["not_employed_mean"]), num(s["standardized_mean_difference"], 3)]
            for s in stat["feature_discriminant_analysis"]["numeric_feature_scores"][:TOP_N]]
    return table("tab:rsa911-numeric", "Numeric variables: point-biserial correlation with employment at exit and standardized mean difference (SMD).",
                 "lrrrrr", ["Variable", "$n$", "$r$", "Employed mean", "Not employed mean", "SMD"], rows)


def logistic_table(stat: dict) -> str:
    fi = stat["feature_importance"]
    if fi.get("status") != "completed":
        return f"% Logistic regression suppressed: {fi.get('reason')}\n"
    rows = [[tex(f["feature"]), num(f["coefficient"], 3)] for f in fi["top_absolute_coefficients"][:TOP_N]]
    caption = (f"Largest standardized coefficients from an L2 logistic regression predicting employment at exit "
               f"($n={num(fi['n'])}$, hold-out AUC $={num(fi['auc_holdout'], 3)}$). Positive values indicate higher "
               "predicted employment. Associations are not causal.")
    return table("tab:rsa911-logit", caption, "lr", ["Feature", "Coefficient"], rows)


def disparity_table(analysis: dict) -> str:
    rows = []
    for key, r in analysis["black_disparity"]["results"].items():
        b, nb = r["black_veterans"], r["nonblack_veterans"]
        diff = r["rate_difference_black_minus_nonblack"]
        rows.append([tex(key.replace("_", " ")), num(b["n"]), pct(b["rate"]), num(nb["n"]), pct(nb["rate"]),
                     "--" if diff is None else f"{100 * diff:+.1f}"])
    return table("tab:rsa911-disparity", "Unadjusted employment rates for Black and non-Black veteran exiters under three outcome "
                 "definitions. Difference in percentage points (Black minus non-Black).",
                 "lrrrrr", ["Outcome definition", "$n_B$", "Rate$_B$", "$n_{NB}$", "Rate$_{NB}$", "Diff."], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary-dir", default=str(work_dir()))
    parser.add_argument("--out", default="statistics/paper/rsa911_tables.tex")
    args = parser.parse_args()

    src = Path(args.summary_dir).expanduser()
    agg = json.loads((src / "aggregate_summary.json").read_text())
    analysis = json.loads((src / "analysis_summary.json").read_text())
    stat = json.loads((src / "statistical_analysis_summary.json").read_text())

    parts = [descriptive_table(stat, agg), outcome_table(agg), disability_table(agg), categorical_table(stat),
             numeric_table(stat), logistic_table(stat), disparity_table(analysis)]
    out = Path(args.out)
    out.write_text("% Generated by statistics/rsa911_tables.py -- do not edit by hand.\n\n" + "\n".join(parts), encoding="utf-8")
    print(f"Wrote {len(parts)} tables to {out}")


if __name__ == "__main__":
    main()
