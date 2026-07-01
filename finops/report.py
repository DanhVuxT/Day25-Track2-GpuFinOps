"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly") -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    lines += [
        "",
        "## Technical analysis",
        "",
        "- GPU-Util is a time-active signal, not an efficiency signal. A GPU can show near-100% utilization while MFU is low because kernels are stalled on memory bandwidth, launch overhead, or small batches; NimbusAI should prioritize MFU/MBU and $/1M-token over raw $/GPU-hour.",
        "- Priority order: first apply inference cascade/cache/batch because it directly cuts unit cost per token, then move steady jobs to reserved and interruptible jobs to spot, then right-size util-lies, then shut down idle GPUs.",
        "- Do not pick the cheapest GPU by hourly rate alone: memory-bound inference needs enough HBM capacity and bandwidth. The right target is the lowest $/GB-VRAM option that still clears the workload's bandwidth requirement.",
    ]
    if sustainability and sustainability.get("cache_economics"):
        c = sustainability["cache_economics"]
        lines += [
            "",
            "## Extension: Cache economics",
            "",
            f"- Observed cached reads/write-token: {c.get('avg_reads', 0):.2f}",
            f"- Break-even cached reads/write-token: {c.get('break_even_reads', 0):.2f}",
            f"- Decision: {'apply prompt caching' if c.get('applied') else 'do not count cache savings yet'}",
        ]
    if sustainability and sustainability.get("reasoning_budget"):
        r = sustainability["reasoning_budget"]
        lines += [
            "",
            "## Extension: Reasoning budget",
            "",
            f"- Reasoning traffic: {r.get('traffic_pct', 0):.1f}% of requests",
            f"- Reasoning cost share: {r.get('cost_pct', 0):.1f}% of optimized inference cost",
            f"- Reasoning energy share: {r.get('energy_pct', 0):.1f}% of inference Wh",
            f"- Routing rule: reserve reasoning for complex tasks, alert at 10% traffic, and hard-cap it at {r.get('cap_traffic_pct', 5):.0f}% of traffic.",
            f"- Estimated savings from the cap: ${r.get('cap_savings_monthly', 0):,.0f}/month and {r.get('cap_kwh_savings_monthly', 0):,.1f} kWh/month.",
        ]
    if sustainability and sustainability.get("rightsize"):
        lines += [
            "",
            "## Extension: MBU-aware right-sizing",
            "",
            "| GPU | Current | Recommended | Required BW (TB/s) | Current $/GB | Target $/GB | Savings/mo |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for r in sustainability["rightsize"]:
            lines.append(
                f"| {r['gpu_id']} | {r['current']} | {r['recommended']} | "
                f"{r['required_bw_tbs']:.2f} | ${r['current_usd_per_gb']:.4f} | "
                f"${r['recommended_usd_per_gb']:.4f} | ${r['monthly_savings']:,.0f} |"
            )
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
            "- Carbon-aware scheduling matters because lower-carbon regions can also reduce electricity cost; latency-sensitive serving should stay close to users, while interruptible training and eval can move to the cleanest acceptable region.",
        ]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
