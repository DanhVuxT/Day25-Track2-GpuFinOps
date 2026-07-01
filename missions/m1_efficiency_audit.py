"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        gtype = a["type"]
        summary.append({
            "gpu_id": gid, "gpu_type": gtype,
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "achieved_bw_tbs": round((sum(a["mbu"]) / len(a["mbu"])) * num(cat[gtype]["peak_bw_tbs"]), 3),
        })

    lies = metrics.flag_util_lies(summary)
    rightsize = []
    for lie in lies:
        cur = lie["gpu_type"]
        cur_price = num(cat[cur]["on_demand_hr"])
        cur_hbm = num(cat[cur]["hbm_gb"])
        required_bw = lie["achieved_bw_tbs"] / 0.60 if lie["achieved_bw_tbs"] > 0 else 0.0
        candidates = []
        for gtype, c in cat.items():
            hourly = num(c["on_demand_hr"])
            if (
                num(c["hbm_gb"]) >= cur_hbm
                and num(c["peak_bw_tbs"]) >= required_bw
                and hourly < cur_price
            ):
                candidates.append((hourly, num(c["peak_bw_tbs"]), gtype, c))
        if candidates:
            _, _, target, target_cat = sorted(candidates)[0]
            delta = cur_price - num(target_cat["on_demand_hr"])
            rightsize.append({
                "gpu_id": lie["gpu_id"],
                "current": cur,
                "recommended": target,
                "required_bw_tbs": round(required_bw, 2),
                "current_usd_per_gb": round(cur_price / cur_hbm, 4),
                "recommended_usd_per_gb": round(num(target_cat["on_demand_hr"]) / num(target_cat["hbm_gb"]), 4),
                "monthly_savings": round(delta * 24 * 30, 2),
                "savings_pct": round(delta / cur_price * 100, 1),
            })
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")
        if rightsize:
            print("\nRight-sizing candidates (MBU-aware, keep HBM and required bandwidth):")
            print(f"{'gpu':14}{'current':9}{'target':9}{'req_bw':>8}{'$/GB now':>10}{'$/GB tgt':>10}{'save/mo':>10}")
            for r in rightsize:
                print(
                    f"{r['gpu_id']:14}{r['current']:9}{r['recommended']:9}"
                    f"{r['required_bw_tbs']:>8.2f}{r['current_usd_per_gb']:>10.4f}"
                    f"{r['recommended_usd_per_gb']:>10.4f}${r['monthly_savings']:>9,.0f}"
                )

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "rightsize_recommendations": rightsize,
        "rightsize_monthly_savings": round(sum(r["monthly_savings"] for r in rightsize), 2),
    }


if __name__ == "__main__":
    run()
