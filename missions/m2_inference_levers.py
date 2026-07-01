"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    cache_input = sum(int(num(r["cached_input_tokens"])) for r in rows)
    written_input = sum(max(0, int(num(r["input_tokens"])) - int(num(r["cached_input_tokens"]))) for r in rows)
    avg_cache_reads = cache_input / written_input if written_input else 0.0
    cache_break_even = pricing.cache_break_even_reads()
    cache_worth_it = pricing.cache_is_worth_it(avg_cache_reads)
    reasoning = {"requests": 0, "tokens": 0, "optimized_cost": 0.0, "wh": 0.0}
    non_reasoning = {"requests": 0, "tokens": 0, "optimized_cost": 0.0, "wh": 0.0}
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"])) if cache_worth_it else 0
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        row_opt_cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += row_opt_cost

        bucket = reasoning if is_reasoning else non_reasoning
        bucket["requests"] += 1
        bucket["tokens"] += inp + out
        bucket["optimized_cost"] += row_opt_cost
        bucket["wh"] += sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    reasoning_cost_pct = reasoning["optimized_cost"] / opt_cost * 100 if opt_cost else 0.0
    reasoning_traffic_pct = reasoning["requests"] / len(rows) * 100 if rows else 0.0
    total_wh = reasoning["wh"] + non_reasoning["wh"]
    reasoning_energy_pct = reasoning["wh"] / total_wh * 100 if total_wh else 0.0
    target_reasoning_pct = 5.0
    excess_reasoning_frac = max(0.0, (reasoning_traffic_pct - target_reasoning_pct) / reasoning_traffic_pct) if reasoning_traffic_pct else 0.0
    capped_cost_savings = reasoning["optimized_cost"] * excess_reasoning_frac
    capped_wh_savings = (
        reasoning["wh"] - sustainability.wh_per_query(reasoning["tokens"])
    ) * excess_reasoning_frac

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"cache     : avg_reads={avg_cache_reads:.2f}, break_even={cache_break_even:.2f}, applied={cache_worth_it}")
        print(f"reasoning : {reasoning_traffic_pct:.1f}% of requests, {reasoning_cost_pct:.1f}% of optimized cost, {reasoning_energy_pct:.1f}% of Wh")
        print(f"cap rule  : alert at 10%, hard-cap reasoning at {target_reasoning_pct:.0f}% traffic -> save ${capped_cost_savings:.2f}/day and {capped_wh_savings:,.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_economics": {
            "avg_reads": round(avg_cache_reads, 2),
            "break_even_reads": round(cache_break_even, 2),
            "applied": cache_worth_it,
        },
        "reasoning_budget": {
            "traffic_pct": round(reasoning_traffic_pct, 1),
            "cost_pct": round(reasoning_cost_pct, 1),
            "energy_pct": round(reasoning_energy_pct, 1),
            "cap_traffic_pct": target_reasoning_pct,
            "cap_savings_daily": round(capped_cost_savings, 2),
            "cap_wh_savings_daily": round(capped_wh_savings, 1),
        },
    }


if __name__ == "__main__":
    run()
