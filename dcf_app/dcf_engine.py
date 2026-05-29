"""
DCF Engine — Two-stage Discounted Cash Flow valuation model.

Stage 1: Years 1-5 at high growth rate
Stage 2: Years 6-10 at lower growth rate
Terminal Value: Gordon Growth Model on Year-10 FCF
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DCFInputs:
    # Base financials
    base_revenue: float = 100.0          # in Cr (India) or USD millions (US)
    base_fcf: float = 15.0               # same units as revenue
    net_debt: float = 0.0                # total debt - cash
    shares_outstanding: float = 100.0   # in crores (India) or millions (US)
    current_price: float = 100.0        # per share

    # Growth assumptions
    revenue_growth_y1_y5: float = 12.0   # % per year, Stage 1
    revenue_growth_y6_y10: float = 8.0   # % per year, Stage 2

    # Margin assumptions
    fcf_margin: float = 15.0            # FCF / Revenue, %
    capex_to_revenue: float = 5.0       # Capex / Revenue, %

    # Discount rate
    wacc: float = 12.0                  # %
    terminal_growth: float = 3.0        # %

    # WACC components (used if compute_wacc_from_components=True)
    risk_free_rate: float = 6.5         # %
    equity_risk_premium: float = 5.0    # %
    beta: float = 1.0
    cost_of_debt: float = 8.0           # %
    debt_ratio: float = 30.0            # D/(D+E), %
    tax_rate: float = 25.0              # %
    compute_wacc_from_components: bool = False

    # Other
    margin_of_safety: float = 20.0      # % discount to IV
    stage1_years: int = 5
    stage2_years: int = 5

    # Currency / display metadata (not used in math)
    currency: str = "INR"
    unit_label: str = "Cr"


@dataclass
class DCFOutputs:
    projected_revenues: List[float] = field(default_factory=list)
    projected_fcfs: List[float] = field(default_factory=list)
    pv_fcfs: List[float] = field(default_factory=list)

    terminal_value: float = 0.0
    pv_terminal_value: float = 0.0

    enterprise_value: float = 0.0
    equity_value: float = 0.0
    intrinsic_value_per_share: float = 0.0
    iv_with_mos: float = 0.0

    upside_downside_pct: float = 0.0
    implied_fcf_yield: float = 0.0
    implied_pe_ratio: float = 0.0

    wacc_used: float = 0.0
    sum_pv_fcfs: float = 0.0


def compute_wacc(inputs: DCFInputs) -> float:
    """Compute WACC from CAPM components or return direct override."""
    if inputs.compute_wacc_from_components:
        equity_ratio = 1.0 - inputs.debt_ratio / 100.0
        debt_ratio = inputs.debt_ratio / 100.0
        cost_of_equity = (
            inputs.risk_free_rate / 100.0
            + inputs.beta * inputs.equity_risk_premium / 100.0
        )
        after_tax_cod = (inputs.cost_of_debt / 100.0) * (1.0 - inputs.tax_rate / 100.0)
        wacc = equity_ratio * cost_of_equity + debt_ratio * after_tax_cod
        return wacc * 100.0  # return as percentage
    return inputs.wacc


def project_cash_flows(inputs: DCFInputs) -> tuple:
    """
    Project revenues and FCFs for Stage 1 + Stage 2.
    Returns (revenues, fcfs) both as lists of length stage1_years + stage2_years.
    """
    revenues = []
    fcfs = []
    rev = inputs.base_revenue
    fcf_margin_dec = inputs.fcf_margin / 100.0
    g1 = inputs.revenue_growth_y1_y5 / 100.0
    g2 = inputs.revenue_growth_y6_y10 / 100.0
    total_years = inputs.stage1_years + inputs.stage2_years

    for yr in range(1, total_years + 1):
        if yr <= inputs.stage1_years:
            rev = rev * (1.0 + g1)
        else:
            rev = rev * (1.0 + g2)
        revenues.append(rev)
        fcfs.append(rev * fcf_margin_dec)

    return revenues, fcfs


def calculate_terminal_value(last_fcf: float, terminal_growth: float, wacc: float) -> float:
    """Gordon Growth Model: TV = FCF_last * (1 + g) / (WACC - g)."""
    wacc_dec = wacc / 100.0
    g_dec = terminal_growth / 100.0
    if wacc_dec <= g_dec:
        # Guard: if WACC <= terminal growth, TV would be infinite/negative
        # Use a minimal spread of 0.5%
        wacc_dec = g_dec + 0.005
    return last_fcf * (1.0 + g_dec) / (wacc_dec - g_dec)


def calculate_npv(cash_flows: List[float], wacc: float) -> tuple:
    """
    Discount a list of annual cash flows at `wacc` (%).
    Returns (pv_list, sum_of_pvs).
    """
    wacc_dec = wacc / 100.0
    pvs = []
    for yr, cf in enumerate(cash_flows, start=1):
        pv = cf / ((1.0 + wacc_dec) ** yr)
        pvs.append(pv)
    return pvs, sum(pvs)


def calculate_intrinsic_value(inputs: DCFInputs) -> DCFOutputs:
    """Full DCF computation. Returns a DCFOutputs instance."""
    outputs = DCFOutputs()

    # Step 1: Effective WACC
    wacc = compute_wacc(inputs)
    outputs.wacc_used = wacc

    # Guard terminal growth
    effective_wacc = wacc
    tg = inputs.terminal_growth
    if tg >= wacc:
        # Warn caller — we'll still compute with a floor spread
        effective_wacc = tg + 0.5  # 0.5% floor spread

    # Step 2: Project revenues & FCFs
    revenues, fcfs = project_cash_flows(inputs)
    outputs.projected_revenues = revenues
    outputs.projected_fcfs = fcfs

    # Step 3: PV of each FCF
    pv_fcfs, sum_pv_fcfs = calculate_npv(fcfs, effective_wacc)
    outputs.pv_fcfs = pv_fcfs
    outputs.sum_pv_fcfs = sum_pv_fcfs

    # Step 4: Terminal value on the LAST projected FCF
    total_years = inputs.stage1_years + inputs.stage2_years
    tv = calculate_terminal_value(fcfs[-1], tg, effective_wacc)
    outputs.terminal_value = tv

    # Step 5: PV of terminal value
    wacc_dec = effective_wacc / 100.0
    pv_tv = tv / ((1.0 + wacc_dec) ** total_years)
    outputs.pv_terminal_value = pv_tv

    # Step 6: Enterprise value
    ev = sum_pv_fcfs + pv_tv
    outputs.enterprise_value = ev

    # Step 7: Equity value
    equity_val = ev - inputs.net_debt
    outputs.equity_value = equity_val

    # Step 8: Intrinsic value per share
    if inputs.shares_outstanding > 0:
        iv_per_share = equity_val / inputs.shares_outstanding
    else:
        iv_per_share = 0.0
    outputs.intrinsic_value_per_share = iv_per_share

    # Step 9: Margin of safety
    mos_dec = inputs.margin_of_safety / 100.0
    outputs.iv_with_mos = iv_per_share * (1.0 - mos_dec)

    # Step 10: Upside/downside
    if inputs.current_price > 0:
        outputs.upside_downside_pct = (iv_per_share / inputs.current_price - 1.0) * 100.0
    else:
        outputs.upside_downside_pct = 0.0

    # Step 11: Implied metrics at current price
    market_cap = inputs.current_price * inputs.shares_outstanding
    if market_cap > 0:
        # FCF yield based on Year-1 projected FCF
        outputs.implied_fcf_yield = (fcfs[0] / market_cap) * 100.0 if fcfs else 0.0
    else:
        outputs.implied_fcf_yield = 0.0

    # Implied P/E: rough — treat FCF as proxy for earnings
    if fcfs and inputs.shares_outstanding > 0:
        eps_y1 = fcfs[0] / inputs.shares_outstanding
        if eps_y1 > 0 and inputs.current_price > 0:
            outputs.implied_pe_ratio = inputs.current_price / eps_y1
        else:
            outputs.implied_pe_ratio = 0.0
    else:
        outputs.implied_pe_ratio = 0.0

    return outputs


def sensitivity_table(
    base_inputs: DCFInputs,
    var1_name: str,
    var1_range: List[float],
    var2_name: str,
    var2_range: List[float],
) -> np.ndarray:
    """
    Compute intrinsic value for every combination of var1 × var2,
    keeping all other inputs at base values.

    var1 → rows (var1_range[i])
    var2 → columns (var2_range[j])
    Returns 2D numpy array shape (len(var1_range), len(var2_range)).
    """
    results = np.zeros((len(var1_range), len(var2_range)))

    for i, v1 in enumerate(var1_range):
        for j, v2 in enumerate(var2_range):
            # Clone inputs
            inp = DCFInputs(
                base_revenue=base_inputs.base_revenue,
                base_fcf=base_inputs.base_fcf,
                net_debt=base_inputs.net_debt,
                shares_outstanding=base_inputs.shares_outstanding,
                current_price=base_inputs.current_price,
                revenue_growth_y1_y5=base_inputs.revenue_growth_y1_y5,
                revenue_growth_y6_y10=base_inputs.revenue_growth_y6_y10,
                fcf_margin=base_inputs.fcf_margin,
                capex_to_revenue=base_inputs.capex_to_revenue,
                wacc=base_inputs.wacc,
                terminal_growth=base_inputs.terminal_growth,
                risk_free_rate=base_inputs.risk_free_rate,
                equity_risk_premium=base_inputs.equity_risk_premium,
                beta=base_inputs.beta,
                cost_of_debt=base_inputs.cost_of_debt,
                debt_ratio=base_inputs.debt_ratio,
                tax_rate=base_inputs.tax_rate,
                compute_wacc_from_components=base_inputs.compute_wacc_from_components,
                margin_of_safety=base_inputs.margin_of_safety,
                stage1_years=base_inputs.stage1_years,
                stage2_years=base_inputs.stage2_years,
                currency=base_inputs.currency,
                unit_label=base_inputs.unit_label,
            )
            # Apply var1
            setattr(inp, var1_name, v1)
            # Apply var2
            setattr(inp, var2_name, v2)

            try:
                out = calculate_intrinsic_value(inp)
                results[i, j] = out.intrinsic_value_per_share
            except Exception:
                results[i, j] = np.nan

    return results


def project_price(
    current_price: float,
    intrinsic_value: float,
    growth_rate: float,
    years: float,
) -> float:
    """
    Project future price assuming gradual mean reversion to IV.
    Uses a blend: fundamental (IV-based) + momentum (growth-based).

    years: can be fractional (e.g., 1/52 for 1 week).
    """
    if years <= 0 or current_price <= 0:
        return current_price

    # Fraction of gap that closes per year (empirically ~30%/yr for 1yr, 70% over 5yr)
    if years < 0.1:  # within a week
        convergence = 0.02 * (years / (1 / 52))  # 2% per week
    elif years <= 1:
        convergence = 0.30 * years
    else:
        convergence = min(0.70, 0.30 + 0.08 * (years - 1))

    # Gap
    gap = intrinsic_value - current_price
    projected = current_price + convergence * gap

    # Compound by growth rate for the horizon
    growth_dec = growth_rate / 100.0
    projected = projected * ((1.0 + growth_dec) ** years)

    return max(projected, 0.0)


def build_price_trajectory(
    current_price: float,
    intrinsic_value: float,
    wacc: float,
    years: int = 5,
) -> List[float]:
    """
    Build yearly price trajectory from year 0 to `years`,
    assuming gradual mean reversion.
    """
    prices = [current_price]
    for yr in range(1, years + 1):
        p = project_price(current_price, intrinsic_value, wacc, yr)
        prices.append(p)
    return prices
