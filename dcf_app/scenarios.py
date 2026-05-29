"""
Scenario presets for DCF valuation: Bull / Base / Bear.
Each preset defines growth, margin, WACC, and terminal growth assumptions.
"""

from typing import Dict, Any, Optional


SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "bull": {
        "name": "Bull Case",
        "description": (
            "Optimistic outlook: strong revenue growth driven by market expansion, "
            "improving operating leverage, favourable macro conditions, and efficient "
            "capital allocation resulting in high FCF margins."
        ),
        "color": "#2ecc71",
        "revenue_growth_y1_y5": 18.0,
        "revenue_growth_y6_y10": 12.0,
        "fcf_margin": 22.0,
        "wacc": 10.0,
        "terminal_growth": 4.0,
        "margin_of_safety": 15.0,
        "capex_to_revenue": 4.0,
        "risk_free_rate": 6.5,
        "equity_risk_premium": 4.5,
        "beta": 0.9,
        "cost_of_debt": 7.0,
        "debt_ratio": 20.0,
        "tax_rate": 25.0,
    },
    "base": {
        "name": "Base Case",
        "description": (
            "Moderate growth consistent with historical performance, stable FCF margins, "
            "and average discount rate reflecting sector norms. Most likely scenario "
            "under normal operating conditions."
        ),
        "color": "#3498db",
        "revenue_growth_y1_y5": 12.0,
        "revenue_growth_y6_y10": 8.0,
        "fcf_margin": 15.0,
        "wacc": 12.0,
        "terminal_growth": 3.0,
        "margin_of_safety": 20.0,
        "capex_to_revenue": 6.0,
        "risk_free_rate": 6.5,
        "equity_risk_premium": 5.0,
        "beta": 1.0,
        "cost_of_debt": 8.0,
        "debt_ratio": 30.0,
        "tax_rate": 25.0,
    },
    "bear": {
        "name": "Bear Case",
        "description": (
            "Pessimistic outlook: slower revenue growth due to competitive pressure, "
            "margin compression, rising interest rates, and higher capital intensity. "
            "Reflects a challenging operating environment."
        ),
        "color": "#e74c3c",
        "revenue_growth_y1_y5": 6.0,
        "revenue_growth_y6_y10": 4.0,
        "fcf_margin": 10.0,
        "wacc": 15.0,
        "terminal_growth": 2.0,
        "margin_of_safety": 30.0,
        "capex_to_revenue": 9.0,
        "risk_free_rate": 6.5,
        "equity_risk_premium": 6.0,
        "beta": 1.2,
        "cost_of_debt": 10.0,
        "debt_ratio": 40.0,
        "tax_rate": 25.0,
    },
}


def auto_adjust_scenarios(fetched_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Adjust scenario assumptions relative to the company's own historical metrics.

    Scaling rules:
    - Bull  = max(historical_growth * 1.3, preset_bull)  capped at +5pp above preset
    - Base  = historical_growth (if available, else keep preset)
    - Bear  = min(historical_growth * 0.6, preset_bear)  floored at -5pp below preset

    FCF margin scaled similarly from historical profit margins.
    """
    import copy

    adjusted = copy.deepcopy(SCENARIO_PRESETS)

    # --- Revenue growth calibration ---
    hist_growth: Optional[float] = fetched_data.get("historical_revenue_cagr")
    if hist_growth is not None and isinstance(hist_growth, (int, float)) and not _is_nan(hist_growth):
        hist_growth = max(0.0, min(50.0, float(hist_growth)))

        # Base: use historical growth, bounded to [8, 18]
        base_g = max(8.0, min(18.0, hist_growth))
        adjusted["base"]["revenue_growth_y1_y5"] = round(base_g, 1)
        adjusted["base"]["revenue_growth_y6_y10"] = round(max(4.0, base_g * 0.6), 1)

        # Bull: 1.3× historical, capped at 30%
        bull_g = min(30.0, hist_growth * 1.3)
        bull_g = max(adjusted["bull"]["revenue_growth_y1_y5"] - 5, bull_g)
        adjusted["bull"]["revenue_growth_y1_y5"] = round(bull_g, 1)
        adjusted["bull"]["revenue_growth_y6_y10"] = round(max(6.0, bull_g * 0.65), 1)

        # Bear: 0.6× historical, floored at 2%
        bear_g = max(2.0, hist_growth * 0.6)
        bear_g = min(adjusted["bear"]["revenue_growth_y1_y5"] + 5, bear_g)
        adjusted["bear"]["revenue_growth_y1_y5"] = round(bear_g, 1)
        adjusted["bear"]["revenue_growth_y6_y10"] = round(max(1.5, bear_g * 0.55), 1)

    # --- FCF margin calibration ---
    hist_margin: Optional[float] = fetched_data.get("profit_margins")
    if hist_margin is not None and isinstance(hist_margin, (int, float)) and not _is_nan(hist_margin):
        # profit_margins from yfinance is a decimal (0.15 = 15%)
        if abs(hist_margin) < 1.0:
            hist_margin_pct = hist_margin * 100.0
        else:
            hist_margin_pct = hist_margin
        hist_margin_pct = max(0.0, min(50.0, hist_margin_pct))

        # Base FCF margin ≈ operating margin proxy
        base_m = max(5.0, min(35.0, hist_margin_pct * 1.1))  # FCF slightly below net
        adjusted["base"]["fcf_margin"] = round(base_m, 1)
        adjusted["bull"]["fcf_margin"] = round(min(40.0, base_m * 1.35), 1)
        adjusted["bear"]["fcf_margin"] = round(max(3.0, base_m * 0.65), 1)

    # --- Beta calibration for WACC ---
    beta: Optional[float] = fetched_data.get("beta")
    if beta is not None and isinstance(beta, (int, float)) and not _is_nan(beta):
        beta = max(0.3, min(3.0, float(beta)))
        for scenario_key in adjusted:
            adjusted[scenario_key]["beta"] = round(beta, 2)
        # Recompute implied WACC hints (rough)
        rfr = adjusted["base"]["risk_free_rate"]
        erp_base = adjusted["base"]["equity_risk_premium"]
        ke_base = rfr + beta * erp_base
        dr_base = adjusted["base"]["debt_ratio"] / 100.0
        kd_base = adjusted["base"]["cost_of_debt"] * (1 - adjusted["base"]["tax_rate"] / 100)
        implied_wacc_base = (1 - dr_base) * ke_base + dr_base * kd_base
        adjusted["base"]["wacc"] = round(max(8.0, min(20.0, implied_wacc_base)), 1)

        # Bull: lower beta
        bull_beta = max(0.3, beta * 0.85)
        adjusted["bull"]["beta"] = round(bull_beta, 2)
        ke_bull = rfr + bull_beta * (erp_base - 0.5)
        dr_bull = adjusted["bull"]["debt_ratio"] / 100.0
        kd_bull = adjusted["bull"]["cost_of_debt"] * (1 - adjusted["bull"]["tax_rate"] / 100)
        implied_wacc_bull = (1 - dr_bull) * ke_bull + dr_bull * kd_bull
        adjusted["bull"]["wacc"] = round(max(7.0, min(14.0, implied_wacc_bull)), 1)

        # Bear: higher beta
        bear_beta = min(2.5, beta * 1.2)
        adjusted["bear"]["beta"] = round(bear_beta, 2)
        ke_bear = rfr + bear_beta * (erp_base + 1.0)
        dr_bear = adjusted["bear"]["debt_ratio"] / 100.0
        kd_bear = adjusted["bear"]["cost_of_debt"] * (1 - adjusted["bear"]["tax_rate"] / 100)
        implied_wacc_bear = (1 - dr_bear) * ke_bear + dr_bear * kd_bear
        adjusted["bear"]["wacc"] = round(max(12.0, min(22.0, implied_wacc_bear)), 1)

    return adjusted


def get_scenario_names() -> Dict[str, str]:
    return {k: v["name"] for k, v in SCENARIO_PRESETS.items()}


def _is_nan(val) -> bool:
    try:
        import math
        return math.isnan(float(val))
    except Exception:
        return False
