"""
Merges the JSON files written by `benchmark_vs_astropy_real.py --json` (one per
machine and per workload) into one comparison table, in Markdown.

    python scripts/make_comparison_table.py benchmarks/results/*.json > benchmarks/COMPARISON.md

Columns: astropy on each machine, then euclidrust on the CPU of each machine,
then euclidrust on the GPU of each machine that has one. Every number is the
median of the repetitions recorded in the file, in milliseconds.
"""
import json
import sys
from pathlib import Path


def short_cpu(name):
    """'Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz' -> 'i9-10850K'."""
    if not name:
        return "?"
    for token in name.replace("(R)", "").replace("(TM)", "").split():
        if token[:1] == "i" and token[1:2].isdigit() and "-" in token:
            return token
    return name


def fmt(ms):
    if ms is None:
        return "n/a"
    return f"{ms:,.0f}" if ms >= 100 else (f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}")


def main(paths):
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not reports:
        sys.exit(__doc__)
    machines = []                                    # in the order they first appear
    for r in reports:
        label = r["machine"].get("label") or short_cpu(r["machine"].get("cpu"))
        if label not in [m["label"] for m in machines]:
            machines.append({"label": label, **r["machine"]})
    workloads = []
    for r in reports:
        if r.get("workload") not in workloads:
            workloads.append(r.get("workload"))

    def find(workload, label):
        for r in reports:
            if r.get("workload") == workload and (r["machine"].get("label") or short_cpu(r["machine"].get("cpu"))) == label:
                return r
        return None

    gpu_names = {}
    for r in reports:
        g = (r.get("stats") or {}).get("gpu")
        if g and g.get("available"):
            gpu_names[r["machine"].get("label")] = g.get("name", "GPU").replace("NVIDIA GeForce ", "")

    def family(label):                               # "i7 (NUC)" -> "i7"
        return label.split(" ")[0]

    columns = [(f"astropy CPU {family(m['label'])}", m["label"], "astropy") for m in machines]
    columns += [(f"EuclidRust CPU {family(m['label'])}" + (" NUC" if "NUC" in m["label"] else ""), m["label"], "cpu")
                for m in machines]
    columns += [(f"Euclid GPU {gpu_names[m['label']]}", m["label"], "gpu") for m in machines if m["label"] in gpu_names]

    out = []
    out.append("## Machines\n")
    out.append("| Label | Processor | Threads | RAM | System | GPU |")
    out.append("|---|---|---|---|---|---|")
    for m in machines:
        out.append(f"| {m['label']} | {m.get('cpu') or '?'} | {m.get('cpu_threads')} | {m.get('ram_gib')} GiB | "
                   f"{m.get('os')} (Python {m.get('python')}) | {gpu_names.get(m['label'], 'none used')} |")
    out.append("")
    out.append("## Time to compute (milliseconds, median, lower is better)\n")
    header = "| Operation | " + " | ".join(c[0] for c in columns) + " |"
    out.append(header)
    out.append("|---|" + "---|" * len(columns))

    for w in workloads:
        some = next(r for r in reports if r.get("workload") == w)
        size_gb = some["file"]["size_bytes"] / 1e9
        title = f"{w} ({size_gb:.2f} GB{', tile-compressed' if some['file'].get('compressed') else ''})"
        out.append("| **" + title + "** |" + " |" * len(columns))
        # --- global statistics
        cells = []
        for _, label, kind in columns:
            r = find(w, label)
            s = (r or {}).get("stats") or {}
            if kind == "astropy":
                cells.append(fmt(s.get("astropy_ms")))
            elif kind == "cpu":
                cells.append(fmt(s.get("euclidrust_cpu_ms")))
            else:
                g = s.get("gpu") or {}
                cells.append(fmt(g.get("end_to_end_ms")) if "end_to_end_ms" in g else "not supported yet" if g.get("unsupported") else "n/a")
        out.append("| min / max / mean of the whole image | " + " | ".join(cells) + " |")
        # --- the GPU with the image already in its memory
        if any(k == "gpu" for _, _, k in columns):
            cells = []
            for _, label, kind in columns:
                if kind != "gpu":
                    cells.append("")
                    continue
                g = (((find(w, label) or {}).get("stats") or {}).get("gpu")) or {}
                cells.append(fmt(g.get("resident_ms")) if "resident_ms" in g else "n/a")
            if any(c not in ("", "n/a") for c in cells):
                out.append("| same, image already on the GPU (copied once, " + ", ".join(
                    f"{(((find(w, label) or {}).get('stats') or {}).get('gpu') or {}).get('upload_ms', 0):.0f} ms" for _, label, k in columns if k == "gpu"
                ) + ") | " + " | ".join(cells) + " |")
        # --- cutout
        cells = []
        for _, label, kind in columns:
            c = (find(w, label) or {}).get("cutout")
            if kind == "astropy":
                cells.append(fmt(c["astropy_ms"]) if c else "n/a")
            elif kind == "cpu":
                cells.append(fmt(c["euclidrust_cpu_ms"]) if c else "n/a")
            else:
                cells.append("(runs on the CPU)")
        out.append("| 50×50 cutout by RA/Dec | " + " | ".join(cells) + " |")

    out.append("")
    out.append("## How many times faster than astropy on the same machine\n")
    out.append("| Operation | " + " | ".join(c[0] for c in columns if c[2] != "astropy") + " |")
    out.append("|---|" + "---|" * len([c for c in columns if c[2] != "astropy"]))
    for w in workloads:
        for label_op, getter in (("min / max / mean of the whole image", lambda r, kind: (
                (r["stats"]["euclidrust_cpu_ms"] if kind == "cpu" else (r["stats"].get("gpu") or {}).get("end_to_end_ms")),
                r["stats"]["astropy_ms"])),
                ("50×50 cutout by RA/Dec", lambda r, kind: ((r["cutout"] or {}).get("euclidrust_cpu_ms") if kind == "cpu" else None,
                                                            (r["cutout"] or {}).get("astropy_ms")))):
            cells = []
            for _, label, kind in columns:
                if kind == "astropy":
                    continue
                r = find(w, label)
                try:
                    num, den = getter(r, kind)
                    cells.append(f"×{den / num:,.1f}" if num and den else "n/a")
                except (TypeError, KeyError):
                    cells.append("n/a")
            out.append(f"| {w}: {label_op} | " + " | ".join(cells) + " |")
    out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
