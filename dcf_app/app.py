"""
DCF Valuation Studio — Streamlit App
Full two-stage discounted cash flow valuation for Indian (NSE/BSE) and US stocks.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from dcf_engine import (
    DCFInputs,
    calculate_intrinsic_value,
    sensitivity_table,
    project_price,
    build_price_trajectory,
)
from scenarios import SCENARIO_PRESETS, auto_adjust_scenarios
from data_fetcher import DataFetcher

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DCF Valuation Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #3498db;
    }
    .metric-card.green { border-left-color: #2ecc71; }
    .metric-card.red   { border-left-color: #e74c3c; }
    .big-iv {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .verdict-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .undervalued { background:#2ecc7133; color:#2ecc71; }
    .overvalued  { background:#e74c3c33; color:#e74c3c; }
    .fair        { background:#f39c1233; color:#f39c12; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    if "stock_data" not in st.session_state:
        st.session_state.stock_data = None
    if "dcf_inputs" not in st.session_state:
        st.session_state.dcf_inputs = DCFInputs()
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = "base"
    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = ""
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = SCENARIO_PRESETS


_init_state()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _fmt_number(val: float, unit: str = "", currency: str = "INR") -> str:
    """Format large numbers with Cr/B/M suffixes."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    if currency == "INR" or unit == "Cr":
        if abs(val) >= 100_000:
            return f"₹{val/100_000:.2f}L Cr"
        if abs(val) >= 1_000:
            return f"₹{val/1_000:.1f}K Cr"
        return f"₹{val:.1f} Cr"
    else:
        if abs(val) >= 1_000:
            return f"${val/1_000:.2f}B"
        return f"${val:.1f}M"


def _fmt_price(val: float, currency: str = "INR") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    sym = "₹" if currency == "INR" else "$"
    return f"{sym}{val:,.2f}"


def _color_upside(pct: float) -> str:
    if pct >= 15:
        return "green"
    if pct <= -15:
        return "red"
    return "orange"


def _verdict(upside_pct: float) -> tuple:
    if upside_pct >= 20:
        return "Undervalued", "undervalued", "🟢"
    if upside_pct >= -10:
        return "Fairly Valued", "fair", "🟡"
    return "Overvalued", "overvalued", "🔴"


def _apply_scenario(scenario_key: str):
    """Apply a scenario preset to session_state.dcf_inputs."""
    scenarios = st.session_state.get("scenarios", SCENARIO_PRESETS)
    if scenario_key not in scenarios:
        return
    s = scenarios[scenario_key]
    inp = st.session_state.dcf_inputs
    for field in [
        "revenue_growth_y1_y5", "revenue_growth_y6_y10", "fcf_margin",
        "wacc", "terminal_growth", "margin_of_safety", "capex_to_revenue",
        "risk_free_rate", "equity_risk_premium", "beta", "cost_of_debt",
        "debt_ratio", "tax_rate",
    ]:
        if field in s:
            setattr(inp, field, float(s[field]))
    st.session_state.selected_scenario = scenario_key


def _build_dcf_inputs_from_stock(stock_data: dict) -> DCFInputs:
    """Build a DCFInputs from fetched stock data."""
    currency = stock_data.get("currency", "USD")
    inp = DCFInputs(
        base_revenue=stock_data.get("total_revenue", 100.0) or 100.0,
        base_fcf=stock_data.get("free_cashflow", 15.0) or 15.0,
        net_debt=stock_data.get("net_debt", 0.0) or 0.0,
        shares_outstanding=max(stock_data.get("shares_outstanding", 1.0) or 1.0, 0.001),
        current_price=stock_data.get("current_price", 100.0) or 100.0,
        beta=max(0.1, stock_data.get("beta", 1.0) or 1.0),
        currency=currency,
        unit_label=stock_data.get("unit_label", "M"),
    )

    # Set growth from historical CAGR if available
    hist_cagr = stock_data.get("historical_revenue_cagr")
    if hist_cagr and not math.isnan(float(hist_cagr or 0)):
        inp.revenue_growth_y1_y5 = max(4.0, min(35.0, float(hist_cagr)))
        inp.revenue_growth_y6_y10 = max(3.0, min(20.0, float(hist_cagr) * 0.65))
    else:
        # Derive from yfinance revenue_growth
        rg = stock_data.get("revenue_growth", 12.0) or 12.0
        inp.revenue_growth_y1_y5 = max(4.0, min(35.0, float(rg)))
        inp.revenue_growth_y6_y10 = max(3.0, min(20.0, float(rg) * 0.65))

    # FCF margin
    fcf_m = stock_data.get("fcf_margin", 0.0) or 0.0
    if abs(fcf_m) > 0.1:
        inp.fcf_margin = max(2.0, min(45.0, float(fcf_m)))
    elif inp.base_revenue > 0 and inp.base_fcf != 0:
        inp.fcf_margin = max(2.0, min(45.0, (inp.base_fcf / inp.base_revenue) * 100.0))

    # WACC from beta
    rfr = 6.5 if currency == "INR" else 4.5
    erp = 5.0
    inp.risk_free_rate = rfr
    inp.wacc = round(rfr + inp.beta * erp, 1)
    inp.wacc = max(8.0, min(20.0, inp.wacc))

    return inp


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("📊 DCF Valuation Studio")
        st.caption("Two-stage DCF model | Indian & US stocks")
        st.divider()

        ticker_input = st.text_input(
            "Stock Ticker",
            value=st.session_state.last_ticker,
            placeholder="e.g. RELIANCE, TCS, AAPL, MSFT",
            help="Enter NSE/BSE symbol for Indian stocks (e.g. RELIANCE, TCS) or US ticker (e.g. AAPL).",
        )

        st.caption("Indian stocks: type symbol only (e.g. INFY). US: AAPL, MSFT, GOOGL.")

        source = st.radio(
            "Data Source",
            options=["Auto", "Yahoo Finance", "Screener.in"],
            index=0,
            horizontal=True,
        )
        source_map = {"Auto": "auto", "Yahoo Finance": "yahoo", "Screener.in": "screener"}

        fetch_btn = st.button("🔍 Fetch Data", type="primary", use_container_width=True)

        if fetch_btn and ticker_input.strip():
            with st.spinner(f"Fetching data for {ticker_input.upper()}..."):
                fetcher = DataFetcher()
                data = fetcher.fetch(ticker_input.strip(), source=source_map[source])
            if data:
                st.session_state.stock_data = data
                st.session_state.last_ticker = ticker_input.strip()
                # Build DCF inputs from fetched data
                st.session_state.dcf_inputs = _build_dcf_inputs_from_stock(data)
                # Auto-adjust scenarios
                st.session_state.scenarios = auto_adjust_scenarios(data)
                # Apply base scenario defaults
                _apply_scenario("base")
                st.session_state.selected_scenario = "base"
                st.success(f"Loaded: {data.get('name', ticker_input)}")
                st.rerun()
            else:
                st.error(
                    f"Could not fetch data for '{ticker_input}'. "
                    "Check the ticker symbol and try again."
                )

        # Show loaded stock info
        if st.session_state.stock_data:
            sd = st.session_state.stock_data
            st.divider()
            st.markdown(f"**{sd.get('name', 'N/A')}**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price", _fmt_price(sd.get("current_price", 0), sd.get("currency", "INR")))
            with col2:
                st.metric("Sector", sd.get("sector", "N/A") or "N/A")

            # Scenario presets
            st.divider()
            st.subheader("Scenario Presets")
            s_cols = st.columns(4)
            scenarios = st.session_state.scenarios
            scenario_labels = [("bull", "🐂"), ("base", "📊"), ("bear", "🐻"), ("custom", "✏️")]
            for idx, (key, icon) in enumerate(scenario_labels):
                with s_cols[idx]:
                    is_active = st.session_state.selected_scenario == key
                    label = f"{icon} {'Bull' if key=='bull' else 'Base' if key=='base' else 'Bear' if key=='bear' else 'Custom'}"
                    if st.button(
                        label,
                        key=f"scenario_btn_{key}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        if key != "custom":
                            _apply_scenario(key)
                        else:
                            st.session_state.selected_scenario = "custom"
                        st.rerun()

            st.divider()
            if st.button("🔄 Refresh Data", use_container_width=True):
                if st.session_state.last_ticker:
                    with st.spinner("Refreshing..."):
                        fetcher = DataFetcher()
                        data = fetcher.fetch(
                            st.session_state.last_ticker,
                            source=source_map[source],
                        )
                    if data:
                        st.session_state.stock_data = data
                        st.session_state.dcf_inputs = _build_dcf_inputs_from_stock(data)
                        st.session_state.scenarios = auto_adjust_scenarios(data)
                        st.success("Data refreshed!")
                        st.rerun()


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

def render_overview_tab(sd: dict, dcf_out):
    currency = sd.get("currency", "USD")
    unit = sd.get("unit_label", "M")

    # --- KPI Row 1 ---
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Price", _fmt_price(sd.get("current_price", 0), currency))
        st.metric("Beta", f"{sd.get('beta', 0):.2f}")
    with c2:
        mc = sd.get("market_cap", 0)
        st.metric("Market Cap", _fmt_number(mc, unit, currency))
        st.metric("Trailing P/E", f"{sd.get('trailing_pe', 0):.1f}x" if sd.get("trailing_pe") else "N/A")
    with c3:
        st.metric("Revenue", _fmt_number(sd.get("total_revenue", 0), unit, currency))
        st.metric("Net Income", _fmt_number(sd.get("net_income", 0), unit, currency))
    with c4:
        st.metric("Free Cash Flow", _fmt_number(sd.get("free_cashflow", 0), unit, currency))
        st.metric("ROE", f"{sd.get('roe', 0):.1f}%" if sd.get("roe") else "N/A")

    c1b, c2b, c3b, c4b = st.columns(4)
    with c1b:
        st.metric("FCF Margin", f"{sd.get('fcf_margin', 0):.1f}%")
    with c2b:
        de = (sd.get("total_debt", 0) / max(sd.get("total_cash", 1), 1))
        st.metric("Debt/Cash", f"{de:.2f}x")
    with c3b:
        st.metric("Rev Growth (YoY)", f"{sd.get('revenue_growth', 0):.1f}%")
    with c4b:
        st.metric("Dividend Yield", f"{sd.get('dividend_yield', 0):.2f}%")

    st.divider()

    # --- Historical charts ---
    col_chart, col_summary = st.columns([2, 1])

    with col_chart:
        hist_rev = sd.get("hist_revenues", [])
        hist_ni = sd.get("hist_net_incomes", [])

        if hist_rev:
            n = min(len(hist_rev), len(hist_ni) if hist_ni else len(hist_rev))
            years_label = [f"FY{i+1}" for i in range(n)]
            rev_vals = hist_rev[-n:] if len(hist_rev) > n else hist_rev
            ni_vals = hist_ni[-n:] if hist_ni and len(hist_ni) > n else (hist_ni[-n:] if hist_ni else [])

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=years_label,
                    y=rev_vals,
                    name=f"Revenue ({unit})",
                    marker_color="#3498db",
                    opacity=0.85,
                ),
                secondary_y=False,
            )
            if ni_vals and len(ni_vals) == n:
                fig.add_trace(
                    go.Bar(
                        x=years_label,
                        y=ni_vals,
                        name=f"Net Income ({unit})",
                        marker_color="#2ecc71",
                        opacity=0.85,
                    ),
                    secondary_y=False,
                )

            fig.update_layout(
                title="Historical Revenue & Net Income",
                height=320,
                barmode="group",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig.update_yaxes(title_text=f"Amount ({unit})", gridcolor="#333")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical data not available for this ticker.")

    with col_summary:
        st.subheader("Valuation Summary")
        if dcf_out:
            iv = dcf_out.intrinsic_value_per_share
            cp = sd.get("current_price", 0)
            upside = dcf_out.upside_downside_pct
            verdict_text, verdict_class, verdict_icon = _verdict(upside)

            st.markdown(
                f"""
                <div class="metric-card {'green' if upside>=0 else 'red'}">
                    <div style="font-size:0.85rem;color:#aaa;">Intrinsic Value / Share</div>
                    <div class="big-iv">{_fmt_price(iv, currency)}</div>
                    <div style="font-size:0.85rem;color:#aaa;margin-top:4px;">
                        Current: {_fmt_price(cp, currency)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="margin:12px 0;">
                    <span class="verdict-badge {verdict_class}">{verdict_icon} {verdict_text}</span>
                    &nbsp;
                    <span style="color:{'#2ecc71' if upside>=0 else '#e74c3c'};font-weight:600;font-size:1.1rem;">
                        {'+' if upside>=0 else ''}{upside:.1f}% {'upside' if upside>=0 else 'downside'}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("MoS Price (IV × {:.0f}%)".format(100 - st.session_state.dcf_inputs.margin_of_safety),
                      _fmt_price(dcf_out.iv_with_mos, currency))
            st.metric("Implied FCF Yield", f"{dcf_out.implied_fcf_yield:.2f}%")
            st.metric("WACC Used", f"{dcf_out.wacc_used:.2f}%")
        else:
            st.info("Run DCF model to see valuation summary.")

    # --- Description ---
    desc = sd.get("description", "")
    if desc:
        st.divider()
        with st.expander("Company Description", expanded=False):
            st.write(desc)

    # --- Data source info ---
    src = sd.get("source", "")
    screener_url = sd.get("screener_url", "")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.caption(f"Data source: **{src}** | Currency: **{currency}** | Units: **{unit}**")
    with col_s2:
        if screener_url:
            st.caption(f"[View on Screener.in]({screener_url})")


# ---------------------------------------------------------------------------
# Tab 2: DCF Model
# ---------------------------------------------------------------------------

def render_dcf_tab(sd: dict):
    currency = sd.get("currency", "USD")
    unit = sd.get("unit_label", "M")
    inp = st.session_state.dcf_inputs

    st.subheader("DCF Model Inputs")
    if st.session_state.selected_scenario != "custom":
        st.info(
            f"Active Scenario: **{st.session_state.selected_scenario.title()}** — "
            "Adjust any slider to switch to Custom."
        )

    col_left, col_mid, col_right = st.columns(3)

    # --- LEFT: Growth & Margin ---
    with col_left:
        st.markdown("**Growth & Margin**")

        new_g1 = st.slider(
            "Stage 1 Rev Growth Y1-Y5 (%)",
            min_value=0.0, max_value=40.0,
            value=float(inp.revenue_growth_y1_y5),
            step=0.5,
            key="slider_g1",
        )
        new_g2 = st.slider(
            "Stage 2 Rev Growth Y6-Y10 (%)",
            min_value=0.0, max_value=30.0,
            value=float(inp.revenue_growth_y6_y10),
            step=0.5,
            key="slider_g2",
        )
        new_fcf_m = st.slider(
            "FCF Margin (% of Revenue)",
            min_value=0.0, max_value=50.0,
            value=float(inp.fcf_margin),
            step=0.5,
            key="slider_fcf_m",
        )
        new_tg = st.slider(
            "Terminal Growth Rate (%)",
            min_value=0.0, max_value=8.0,
            value=float(inp.terminal_growth),
            step=0.25,
            key="slider_tg",
        )

    # --- MIDDLE: Discount rate ---
    with col_mid:
        st.markdown("**Discount Rate**")

        new_wacc = st.slider(
            "WACC — direct override (%)",
            min_value=5.0, max_value=25.0,
            value=float(inp.wacc),
            step=0.25,
            key="slider_wacc",
        )

        with st.expander("⚙️ Compute WACC from components"):
            new_rfr = st.slider(
                "Risk-Free Rate (%)", 2.0, 10.0,
                value=float(inp.risk_free_rate), step=0.25, key="slider_rfr",
            )
            new_erp = st.slider(
                "Equity Risk Premium (%)", 3.0, 10.0,
                value=float(inp.equity_risk_premium), step=0.25, key="slider_erp",
            )
            new_beta = st.number_input(
                "Beta", min_value=0.1, max_value=4.0,
                value=float(inp.beta), step=0.05, key="ni_beta",
            )
            new_cod = st.slider(
                "Cost of Debt (%)", 3.0, 15.0,
                value=float(inp.cost_of_debt), step=0.25, key="slider_cod",
            )
            new_dr = st.slider(
                "Debt Ratio D/(D+E) (%)", 0.0, 60.0,
                value=float(inp.debt_ratio), step=1.0, key="slider_dr",
            )
            use_components = st.checkbox(
                "Use component-computed WACC",
                value=inp.compute_wacc_from_components,
                key="cb_wacc_comp",
            )
            if use_components:
                er = 1.0 - new_dr / 100.0
                ke = new_rfr / 100.0 + new_beta * new_erp / 100.0
                kd = (new_cod / 100.0) * (1 - inp.tax_rate / 100.0)
                computed_wacc = (er * ke + (new_dr / 100.0) * kd) * 100.0
                st.info(f"Computed WACC: **{computed_wacc:.2f}%**")

        new_tax = st.slider(
            "Tax Rate (%)", 0.0, 40.0,
            value=float(inp.tax_rate), step=1.0, key="slider_tax",
        )
        new_capex = st.slider(
            "Capex as % of Revenue", 0.0, 20.0,
            value=float(inp.capex_to_revenue), step=0.5, key="slider_capex",
        )

    # --- RIGHT: Balance Sheet ---
    with col_right:
        st.markdown("**Balance Sheet & Other**")

        new_nd = st.number_input(
            f"Net Debt ({unit})",
            value=float(inp.net_debt),
            step=100.0,
            key="ni_nd",
            help="Total Debt minus Cash. Negative = net cash position.",
        )
        new_shares = st.number_input(
            f"Shares Outstanding ({'Cr' if currency=='INR' else 'M'})",
            value=float(max(inp.shares_outstanding, 0.001)),
            min_value=0.001,
            step=1.0,
            key="ni_shares",
        )
        new_mos = st.slider(
            "Margin of Safety (%)", 0.0, 40.0,
            value=float(inp.margin_of_safety), step=1.0, key="slider_mos",
        )
        new_base_rev = st.number_input(
            f"Base Revenue — Last FY ({unit})",
            value=float(max(inp.base_revenue, 0.01)),
            min_value=0.01,
            step=100.0,
            key="ni_base_rev",
        )
        new_base_fcf = st.number_input(
            f"Base FCF — Last FY ({unit})",
            value=float(inp.base_fcf),
            step=10.0,
            key="ni_base_fcf",
        )

    # --- Detect changes → mark custom ---
    def _changed():
        st.session_state.selected_scenario = "custom"

    # Build updated inputs
    updated_inp = DCFInputs(
        base_revenue=new_base_rev,
        base_fcf=new_base_fcf,
        net_debt=new_nd,
        shares_outstanding=new_shares,
        current_price=inp.current_price,
        revenue_growth_y1_y5=new_g1,
        revenue_growth_y6_y10=new_g2,
        fcf_margin=new_fcf_m,
        capex_to_revenue=new_capex,
        wacc=new_wacc,
        terminal_growth=new_tg,
        risk_free_rate=new_rfr if "slider_rfr" in st.session_state else inp.risk_free_rate,
        equity_risk_premium=new_erp if "slider_erp" in st.session_state else inp.equity_risk_premium,
        beta=new_beta if "ni_beta" in st.session_state else inp.beta,
        cost_of_debt=new_cod if "slider_cod" in st.session_state else inp.cost_of_debt,
        debt_ratio=new_dr if "slider_dr" in st.session_state else inp.debt_ratio,
        tax_rate=new_tax,
        compute_wacc_from_components=use_components if "cb_wacc_comp" in st.session_state else inp.compute_wacc_from_components,
        margin_of_safety=new_mos,
        currency=inp.currency,
        unit_label=inp.unit_label,
    )

    # Check if any slider changed from the stored inputs
    _fields_to_check = [
        "revenue_growth_y1_y5", "revenue_growth_y6_y10", "fcf_margin",
        "wacc", "terminal_growth", "margin_of_safety", "capex_to_revenue", "tax_rate",
        "net_debt", "shares_outstanding", "base_revenue", "base_fcf",
    ]
    for f in _fields_to_check:
        if abs(getattr(updated_inp, f, 0) - getattr(inp, f, 0)) > 0.0001:
            st.session_state.selected_scenario = "custom"
            break

    st.session_state.dcf_inputs = updated_inp

    # --- WACC guard ---
    if updated_inp.terminal_growth >= updated_inp.wacc:
        st.warning(
            f"Terminal Growth ({updated_inp.terminal_growth}%) is >= WACC ({updated_inp.wacc}%). "
            "This would produce an infinite or negative Terminal Value. A floor spread of 0.5% will be applied."
        )

    st.divider()

    # --- Compute and display results ---
    try:
        dcf_out = calculate_intrinsic_value(updated_inp)
    except Exception as e:
        st.error(f"DCF calculation error: {e}")
        return dcf_out if "dcf_out" in dir() else None

    # Big IV display
    iv = dcf_out.intrinsic_value_per_share
    cp = updated_inp.current_price
    upside = dcf_out.upside_downside_pct
    color = "#2ecc71" if upside >= 0 else "#e74c3c"

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    with res_col1:
        st.markdown(
            f"""
            <div class="metric-card {'green' if upside>=0 else 'red'}">
                <div style="font-size:0.8rem;color:#aaa;">Intrinsic Value / Share</div>
                <div style="font-size:2.2rem;font-weight:700;color:{color};">{_fmt_price(iv, currency)}</div>
                <div style="font-size:0.8rem;color:#aaa;">Current: {_fmt_price(cp, currency)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#aaa;">MoS Price ({100-updated_inp.margin_of_safety:.0f}% of IV)</div>
                <div style="font-size:2.2rem;font-weight:700;">{_fmt_price(dcf_out.iv_with_mos, currency)}</div>
                <div style="font-size:0.8rem;color:#aaa;">MoS: {updated_inp.margin_of_safety:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_col3:
        verdict_text, verdict_class, verdict_icon = _verdict(upside)
        st.markdown(
            f"""
            <div class="metric-card {'green' if upside>=0 else 'red'}">
                <div style="font-size:0.8rem;color:#aaa;">Upside / Downside</div>
                <div style="font-size:2.2rem;font-weight:700;color:{color};">{'+' if upside>=0 else ''}{upside:.1f}%</div>
                <div style="font-size:0.8rem;margin-top:4px;"><span class="verdict-badge {verdict_class}">{verdict_icon} {verdict_text}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#aaa;">Enterprise Value</div>
                <div style="font-size:2.2rem;font-weight:700;">{_fmt_number(dcf_out.enterprise_value, unit, currency)}</div>
                <div style="font-size:0.8rem;color:#aaa;">Equity: {_fmt_number(dcf_out.equity_value, unit, currency)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Breakdown
    st.subheader("Valuation Breakdown")
    bk_col1, bk_col2 = st.columns([1, 2])
    with bk_col1:
        total_years = updated_inp.stage1_years + updated_inp.stage2_years
        breakdown_data = {
            "Component": [
                f"PV of FCFs (Yrs 1-{total_years})",
                f"PV of Terminal Value (Yr {total_years})",
                "= Enterprise Value",
                "− Net Debt",
                "= Equity Value",
                f"÷ Shares ({updated_inp.shares_outstanding:.1f} {'Cr' if currency=='INR' else 'M'})",
                "= Intrinsic Value / Share",
                f"× MoS ({100-updated_inp.margin_of_safety:.0f}%)",
                "= MoS-Adjusted IV",
            ],
            "Value": [
                _fmt_number(dcf_out.sum_pv_fcfs, unit, currency),
                _fmt_number(dcf_out.pv_terminal_value, unit, currency),
                _fmt_number(dcf_out.enterprise_value, unit, currency),
                _fmt_number(updated_inp.net_debt, unit, currency),
                _fmt_number(dcf_out.equity_value, unit, currency),
                "",
                _fmt_price(iv, currency),
                "",
                _fmt_price(dcf_out.iv_with_mos, currency),
            ],
        }
        df_bk = pd.DataFrame(breakdown_data)
        st.dataframe(df_bk, use_container_width=True, hide_index=True)

        # Waterfall chart
        pv_share = dcf_out.sum_pv_fcfs / updated_inp.shares_outstanding if updated_inp.shares_outstanding else 0
        tv_share = dcf_out.pv_terminal_value / updated_inp.shares_outstanding if updated_inp.shares_outstanding else 0
        nd_share = updated_inp.net_debt / updated_inp.shares_outstanding if updated_inp.shares_outstanding else 0
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["PV FCFs", "PV TV", "EV/share", "-Net Debt/share", "IV/share"],
            y=[pv_share, tv_share, 0, -nd_share, 0],
            connector={"line": {"color": "#555"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}},
        ))
        fig_wf.update_layout(
            height=280, showlegend=False,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with bk_col2:
        # Projected cash flows table
        st.markdown("**Projected Cash Flows (10-Year Forecast)**")
        total_yrs = updated_inp.stage1_years + updated_inp.stage2_years
        yr_labels = [f"Year {i+1}" for i in range(total_yrs)]
        stage_labels = (
            ["Stage 1"] * updated_inp.stage1_years +
            ["Stage 2"] * updated_inp.stage2_years
        )
        cf_df = pd.DataFrame({
            "Year": yr_labels,
            "Stage": stage_labels,
            f"Revenue ({unit})": [f"{r:.1f}" for r in dcf_out.projected_revenues],
            f"FCF ({unit})": [f"{f:.1f}" for f in dcf_out.projected_fcfs],
            f"PV of FCF ({unit})": [f"{p:.1f}" for p in dcf_out.pv_fcfs],
        })
        st.dataframe(cf_df, use_container_width=True, hide_index=True)

        # FCF bar chart
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=yr_labels,
            y=dcf_out.projected_fcfs,
            name="Projected FCF",
            marker_color=["#3498db"] * updated_inp.stage1_years + ["#9b59b6"] * updated_inp.stage2_years,
        ))
        fig_cf.add_trace(go.Scatter(
            x=yr_labels, y=dcf_out.pv_fcfs,
            name="PV of FCF", mode="lines+markers",
            line=dict(color="#f39c12", width=2), marker=dict(size=6),
        ))
        fig_cf.update_layout(
            title="Projected FCF vs PV",
            height=280,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig_cf.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_cf, use_container_width=True)

    return dcf_out


# ---------------------------------------------------------------------------
# Tab 3: Scenarios
# ---------------------------------------------------------------------------

def render_scenarios_tab(sd: dict, base_dcf_out):
    currency = sd.get("currency", "USD")
    unit = sd.get("unit_label", "M")
    cp = sd.get("current_price", 0)
    base_inp = st.session_state.dcf_inputs
    scenarios = st.session_state.scenarios

    st.subheader("Bull / Base / Bear Scenario Comparison")

    scenario_results = {}
    scenario_cols = st.columns(3)

    for idx, scenario_key in enumerate(["bull", "base", "bear"]):
        s = scenarios[scenario_key]
        # Build DCF inputs for this scenario
        s_inp = DCFInputs(
            base_revenue=base_inp.base_revenue,
            base_fcf=base_inp.base_fcf,
            net_debt=base_inp.net_debt,
            shares_outstanding=base_inp.shares_outstanding,
            current_price=cp,
            revenue_growth_y1_y5=s.get("revenue_growth_y1_y5", 12.0),
            revenue_growth_y6_y10=s.get("revenue_growth_y6_y10", 8.0),
            fcf_margin=s.get("fcf_margin", 15.0),
            wacc=s.get("wacc", 12.0),
            terminal_growth=s.get("terminal_growth", 3.0),
            margin_of_safety=s.get("margin_of_safety", 20.0),
            capex_to_revenue=s.get("capex_to_revenue", 6.0),
            risk_free_rate=s.get("risk_free_rate", base_inp.risk_free_rate),
            equity_risk_premium=s.get("equity_risk_premium", base_inp.equity_risk_premium),
            beta=s.get("beta", base_inp.beta),
            cost_of_debt=s.get("cost_of_debt", base_inp.cost_of_debt),
            debt_ratio=s.get("debt_ratio", base_inp.debt_ratio),
            tax_rate=s.get("tax_rate", base_inp.tax_rate),
            currency=base_inp.currency,
            unit_label=base_inp.unit_label,
        )
        try:
            s_out = calculate_intrinsic_value(s_inp)
        except Exception:
            continue

        scenario_results[scenario_key] = (s, s_inp, s_out)
        iv = s_out.intrinsic_value_per_share
        upside = s_out.upside_downside_pct
        color = s.get("color", "#3498db")

        with scenario_cols[idx]:
            verdict_text, verdict_class, _ = _verdict(upside)
            st.markdown(
                f"""
                <div style="border:2px solid {color};border-radius:12px;padding:16px;margin-bottom:12px;">
                    <div style="font-size:1.1rem;font-weight:700;color:{color};">{s['name']}</div>
                    <div style="font-size:0.8rem;color:#aaa;margin:6px 0 10px;">{s['description'][:100]}...</div>
                    <div style="font-size:2rem;font-weight:700;">{_fmt_price(iv, currency)}</div>
                    <div style="font-size:1rem;color:{'#2ecc71' if upside>=0 else '#e74c3c'};font-weight:600;">
                        {'+' if upside>=0 else ''}{upside:.1f}% vs current
                    </div>
                    <div style="margin-top:8px;"><span class="verdict-badge {verdict_class}">{verdict_text}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**Assumptions:**")
            assumption_data = {
                "Assumption": [
                    "Rev Growth Y1-5", "Rev Growth Y6-10",
                    "FCF Margin", "WACC",
                    "Terminal Growth", "Margin of Safety",
                ],
                "Value": [
                    f"{s_inp.revenue_growth_y1_y5:.1f}%",
                    f"{s_inp.revenue_growth_y6_y10:.1f}%",
                    f"{s_inp.fcf_margin:.1f}%",
                    f"{s_inp.wacc:.1f}%",
                    f"{s_inp.terminal_growth:.1f}%",
                    f"{s_inp.margin_of_safety:.0f}%",
                ],
            }
            st.dataframe(pd.DataFrame(assumption_data), use_container_width=True, hide_index=True)

            if upside >= 20:
                st.success("Strong Buy")
            elif upside >= 5:
                st.success("Buy / Accumulate")
            elif upside >= -10:
                st.warning("Hold / Fair Value")
            else:
                st.error("Avoid / Sell")

    st.divider()

    # Comparison chart
    if scenario_results:
        labels = [v[0]["name"] for v in scenario_results.values()]
        iv_vals = [v[2].intrinsic_value_per_share for v in scenario_results.values()]
        colors = [v[0].get("color", "#3498db") for v in scenario_results.values()]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=labels, y=iv_vals, name="Intrinsic Value",
            marker_color=colors,
            text=[_fmt_price(v, currency) for v in iv_vals],
            textposition="outside",
        ))
        fig_bar.add_hline(
            y=cp, line_dash="dash", line_color="white",
            annotation_text=f"Current Price: {_fmt_price(cp, currency)}",
            annotation_position="top right",
        )
        fig_bar.update_layout(
            title="Intrinsic Value Across Scenarios",
            height=400,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            showlegend=False,
            yaxis=dict(title=f"Intrinsic Value ({currency})", gridcolor="#333"),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Sensitivity Analysis
# ---------------------------------------------------------------------------

def render_sensitivity_tab(sd: dict):
    currency = sd.get("currency", "USD")
    inp = st.session_state.dcf_inputs
    cp = sd.get("current_price", inp.current_price)

    st.subheader("Sensitivity Analysis — Intrinsic Value Heatmap")

    var_options = {
        "WACC (%)": "wacc",
        "Terminal Growth (%)": "terminal_growth",
        "FCF Margin (%)": "fcf_margin",
        "Revenue Growth Y1-5 (%)": "revenue_growth_y1_y5",
        "Revenue Growth Y6-10 (%)": "revenue_growth_y6_y10",
        "Margin of Safety (%)": "margin_of_safety",
    }

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        x_label = st.selectbox("X-axis Variable (columns)", list(var_options.keys()), index=0)
    with col_s2:
        remaining = [k for k in var_options if k != x_label]
        y_label = st.selectbox("Y-axis Variable (rows)", remaining, index=1)

    x_field = var_options[x_label]
    y_field = var_options[y_label]

    # Range sliders
    col_r1, col_r2 = st.columns(2)

    def _default_range(field, base_val):
        if field in ("wacc",):
            return (max(5.0, base_val - 4.0), min(25.0, base_val + 4.0))
        if field == "terminal_growth":
            return (max(0.5, base_val - 2.0), min(7.0, base_val + 2.0))
        if field in ("fcf_margin",):
            return (max(2.0, base_val - 8.0), min(45.0, base_val + 8.0))
        if "growth" in field:
            return (max(2.0, base_val - 8.0), min(35.0, base_val + 8.0))
        if field == "margin_of_safety":
            return (0.0, 40.0)
        return (max(0.0, base_val - 5.0), base_val + 5.0)

    x_base = getattr(inp, x_field)
    y_base = getattr(inp, y_field)
    x_rng = _default_range(x_field, x_base)
    y_rng = _default_range(y_field, y_base)

    with col_r1:
        x_min, x_max = st.slider(
            f"{x_label} range",
            min_value=0.0, max_value=50.0,
            value=(float(round(x_rng[0], 1)), float(round(x_rng[1], 1))),
            step=0.5, key="sens_x_range",
        )
    with col_r2:
        y_min, y_max = st.slider(
            f"{y_label} range",
            min_value=0.0, max_value=50.0,
            value=(float(round(y_rng[0], 1)), float(round(y_rng[1], 1))),
            step=0.5, key="sens_y_range",
        )

    n_points = st.select_slider("Grid points", options=[5, 7, 10, 12, 15], value=10)

    x_range = np.linspace(x_min, x_max, n_points).tolist()
    y_range = np.linspace(y_min, y_max, n_points).tolist()

    with st.spinner("Computing sensitivity table..."):
        # var1=y (rows), var2=x (columns)
        table = sensitivity_table(inp, y_field, y_range, x_field, x_range)

    # Build heatmap
    x_labels = [f"{v:.1f}%" for v in x_range]
    y_labels = [f"{v:.1f}%" for v in y_range]

    # Color: diverging around current price
    color_matrix = [[iv - cp for iv in row] for row in table]

    # Annotations
    annotations = []
    for i, row in enumerate(table):
        for j, iv_val in enumerate(row):
            is_base = (abs(x_range[j] - x_base) < (x_max - x_min) / n_points / 2 and
                       abs(y_range[i] - y_base) < (y_max - y_min) / n_points / 2)
            text = f"<b>{_fmt_price(iv_val, currency)}</b>" if is_base else _fmt_price(iv_val, currency)
            annotations.append(
                dict(
                    x=j, y=i, text=text,
                    showarrow=False,
                    font=dict(size=9, color="white" if abs(iv_val - cp) > cp * 0.15 else "black"),
                    xref="x", yref="y",
                )
            )

    fig_heat = go.Figure(go.Heatmap(
        z=color_matrix,
        x=x_labels, y=y_labels,
        colorscale=[
            [0.0, "#7f0000"],
            [0.2, "#d73027"],
            [0.4, "#fdae61"],
            [0.5, "#ffffbf"],
            [0.6, "#a6d96a"],
            [0.8, "#1a9641"],
            [1.0, "#004d00"],
        ],
        zmid=0,
        colorbar=dict(
            title=f"IV − Price ({currency})",
            tickformat=".0f",
            len=0.9,
        ),
        showscale=True,
    ))

    fig_heat.update_layout(
        title=f"Intrinsic Value: {y_label} (rows) × {x_label} (cols)<br>"
              f"<sub>Green = IV > Current Price (undervalued) | Red = IV < Current Price (overvalued)</sub>",
        xaxis=dict(title=x_label, tickvals=list(range(n_points)), ticktext=x_labels, tickangle=-30),
        yaxis=dict(title=y_label, tickvals=list(range(n_points)), ticktext=y_labels),
        height=520,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        annotations=annotations,
        margin=dict(l=60, r=20, t=80, b=60),
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        f"Base case: {x_label} = {x_base:.1f}%, {y_label} = {y_base:.1f}% | "
        f"Current Price: {_fmt_price(cp, currency)}"
    )


# ---------------------------------------------------------------------------
# Tab 5: Price Projections
# ---------------------------------------------------------------------------

def render_projections_tab(sd: dict, base_dcf_out):
    if base_dcf_out is None:
        st.info("Run the DCF model first (Tab 2) to see price projections.")
        return

    currency = sd.get("currency", "USD")
    cp = sd.get("current_price", 0)
    inp = st.session_state.dcf_inputs
    scenarios = st.session_state.scenarios
    iv_base = base_dcf_out.intrinsic_value_per_share
    wacc_base = base_dcf_out.wacc_used

    st.subheader("Price Projections")
    st.markdown(
        "Projections assume **gradual mean reversion** toward intrinsic value. "
        "Short-term prices reflect market sentiment; long-term prices approach fundamental value."
    )

    # Compute projections for each scenario
    def _proj_for_scenario(scenario_key):
        s = scenarios[scenario_key]
        s_inp = DCFInputs(
            base_revenue=inp.base_revenue,
            base_fcf=inp.base_fcf,
            net_debt=inp.net_debt,
            shares_outstanding=inp.shares_outstanding,
            current_price=cp,
            revenue_growth_y1_y5=s.get("revenue_growth_y1_y5", 12.0),
            revenue_growth_y6_y10=s.get("revenue_growth_y6_y10", 8.0),
            fcf_margin=s.get("fcf_margin", 15.0),
            wacc=s.get("wacc", 12.0),
            terminal_growth=s.get("terminal_growth", 3.0),
            margin_of_safety=0.0,  # no MoS for projections
            capex_to_revenue=s.get("capex_to_revenue", 6.0),
            currency=inp.currency,
            unit_label=inp.unit_label,
        )
        try:
            s_out = calculate_intrinsic_value(s_inp)
            return s_out, s_inp
        except Exception:
            return None, None

    bull_out, bull_inp = _proj_for_scenario("bull")
    bear_out, bear_inp = _proj_for_scenario("bear")

    iv_bull = bull_out.intrinsic_value_per_share if bull_out else iv_base * 1.3
    iv_bear = bear_out.intrinsic_value_per_share if bear_out else iv_base * 0.7
    wacc_bull = bull_out.wacc_used if bull_out else wacc_base - 2
    wacc_bear = bear_out.wacc_used if bear_out else wacc_base + 3

    # --- 1 WEEK projection ---
    # Assumes 2% of the gap between current price and IV closes per week
    def _week_proj(iv):
        if cp <= 0:
            return cp
        gap = iv - cp
        return cp + 0.02 * gap

    p1w_base = _week_proj(iv_base)
    p1w_bull = _week_proj(iv_bull)
    p1w_bear = _week_proj(iv_bear)

    # --- 1 YEAR projection ---
    # Method: 30% convergence toward IV
    def _1yr_proj(iv, wacc):
        if cp <= 0:
            return cp
        gap = iv - cp
        return cp + 0.30 * gap

    p1y_base = _1yr_proj(iv_base, wacc_base)
    p1y_bull = _1yr_proj(iv_bull, wacc_bull)
    p1y_bear = _1yr_proj(iv_bear, wacc_bear)

    # --- 5 YEAR projection ---
    # 70% convergence + compounding with growth rate
    def _5yr_proj(iv, growth_rate):
        if cp <= 0:
            return cp
        gap = iv - cp
        price_at_5 = (cp + 0.70 * gap) * ((1 + growth_rate / 100.0) ** 5)
        return max(price_at_5, 0.0)

    p5y_base = _5yr_proj(iv_base, inp.revenue_growth_y6_y10)
    p5y_bull = _5yr_proj(
        iv_bull,
        scenarios["bull"].get("revenue_growth_y6_y10", 12.0),
    )
    p5y_bear = _5yr_proj(
        iv_bear,
        scenarios["bear"].get("revenue_growth_y6_y10", 4.0),
    )

    def _ret_pct(proj):
        if cp and cp > 0:
            return (proj / cp - 1) * 100.0
        return 0.0

    def _cagr(proj, yrs):
        if cp and cp > 0 and proj > 0:
            return ((proj / cp) ** (1.0 / yrs) - 1) * 100.0
        return 0.0

    # --- Display ---
    st.subheader("Projected Prices at Key Horizons")
    h_cols = st.columns(3)

    with h_cols[0]:
        st.markdown("### 1 Week")
        st.caption("Market-mood driven; assumes 2% gap closure per week.")
        st.metric("Bear", _fmt_price(p1w_bear, currency), f"{_ret_pct(p1w_bear):+.2f}%",
                  delta_color="normal")
        st.metric("Base", _fmt_price(p1w_base, currency), f"{_ret_pct(p1w_base):+.2f}%",
                  delta_color="normal")
        st.metric("Bull", _fmt_price(p1w_bull, currency), f"{_ret_pct(p1w_bull):+.2f}%",
                  delta_color="normal")
        st.caption("Note: 1-week is model-indicative only. Short-term prices are dominated by market forces.")

    with h_cols[1]:
        st.markdown("### 1 Year")
        st.caption("Assumes 30% convergence toward intrinsic value over 12 months.")
        c1a, c1b = st.columns(2)
        with c1a:
            st.metric("Bear", _fmt_price(p1y_bear, currency), f"{_ret_pct(p1y_bear):+.1f}%")
        with c1b:
            st.metric("CAGR", f"{_cagr(p1y_bear, 1):.1f}%")
        c2a, c2b = st.columns(2)
        with c2a:
            st.metric("Base", _fmt_price(p1y_base, currency), f"{_ret_pct(p1y_base):+.1f}%")
        with c2b:
            st.metric("CAGR", f"{_cagr(p1y_base, 1):.1f}%")
        c3a, c3b = st.columns(2)
        with c3a:
            st.metric("Bull", _fmt_price(p1y_bull, currency), f"{_ret_pct(p1y_bull):+.1f}%")
        with c3b:
            st.metric("CAGR", f"{_cagr(p1y_bull, 1):.1f}%")

    with h_cols[2]:
        st.markdown("### 5 Years")
        st.caption("Assumes 70% convergence + FCF growth compounding.")
        c1a, c1b = st.columns(2)
        with c1a:
            st.metric("Bear", _fmt_price(p5y_bear, currency), f"{_ret_pct(p5y_bear):+.1f}%")
        with c1b:
            st.metric("CAGR", f"{_cagr(p5y_bear, 5):.1f}%")
        c2a, c2b = st.columns(2)
        with c2a:
            st.metric("Base", _fmt_price(p5y_base, currency), f"{_ret_pct(p5y_base):+.1f}%")
        with c2b:
            st.metric("CAGR", f"{_cagr(p5y_base, 5):.1f}%")
        c3a, c3b = st.columns(2)
        with c3a:
            st.metric("Bull", _fmt_price(p5y_bull, currency), f"{_ret_pct(p5y_bull):+.1f}%")
        with c3b:
            st.metric("CAGR", f"{_cagr(p5y_bull, 5):.1f}%")

    st.divider()

    # 5-year trajectory chart
    st.subheader("5-Year Price Trajectory")

    years_axis = list(range(0, 6))
    base_traj = [cp] + [_1yr_proj(iv_base, wacc_base) * ((1 + inp.revenue_growth_y6_y10 / 100) ** (y - 1))
                        if y >= 2 else _1yr_proj(iv_base, wacc_base) for y in range(1, 6)]
    bull_traj = [cp] + [_1yr_proj(iv_bull, wacc_bull) * ((1 + scenarios["bull"].get("revenue_growth_y6_y10", 12) / 100) ** (y - 1))
                        if y >= 2 else _1yr_proj(iv_bull, wacc_bull) for y in range(1, 6)]
    bear_traj = [cp] + [_1yr_proj(iv_bear, wacc_bear) * ((1 + scenarios["bear"].get("revenue_growth_y6_y10", 4) / 100) ** (y - 1))
                        if y >= 2 else _1yr_proj(iv_bear, wacc_bear) for y in range(1, 6)]

    # Use proper compounding for years 2-5
    for i in range(2, 6):
        base_traj[i] = base_traj[1] * ((1 + inp.revenue_growth_y6_y10 / 100) ** (i - 1))
        bull_traj[i] = bull_traj[1] * ((1 + scenarios["bull"].get("revenue_growth_y6_y10", 12) / 100) ** (i - 1))
        bear_traj[i] = bear_traj[1] * ((1 + scenarios["bear"].get("revenue_growth_y6_y10", 4) / 100) ** (i - 1))

    # Cap at 2× IV to avoid outliers
    base_traj = [min(v, iv_base * 2) for v in base_traj]
    bull_traj = [min(v, iv_bull * 2) for v in bull_traj]
    bear_traj = [max(0, min(v, iv_bear * 2)) for v in bear_traj]

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=years_axis, y=bull_traj, name="Bull",
        mode="lines+markers", line=dict(color="#2ecc71", width=2, dash="solid"),
        marker=dict(size=7),
    ))
    fig_traj.add_trace(go.Scatter(
        x=years_axis, y=base_traj, name="Base",
        mode="lines+markers", line=dict(color="#3498db", width=3),
        marker=dict(size=8),
    ))
    fig_traj.add_trace(go.Scatter(
        x=years_axis, y=bear_traj, name="Bear",
        mode="lines+markers", line=dict(color="#e74c3c", width=2, dash="dot"),
        marker=dict(size=7),
    ))
    # Add IV lines
    fig_traj.add_hline(y=iv_base, line_dash="dash", line_color="#3498db",
                       annotation_text=f"Base IV: {_fmt_price(iv_base, currency)}", opacity=0.5)
    fig_traj.add_hline(y=cp, line_dash="dot", line_color="white",
                       annotation_text=f"Current: {_fmt_price(cp, currency)}", opacity=0.7)

    # Shaded region bull-bear
    fig_traj.add_trace(go.Scatter(
        x=years_axis + years_axis[::-1],
        y=bull_traj + bear_traj[::-1],
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Range",
    ))

    fig_traj.update_layout(
        title="5-Year Price Trajectory (Mean Reversion Model)",
        xaxis=dict(title="Years from Now", tickvals=years_axis, gridcolor="#333"),
        yaxis=dict(title=f"Price ({currency})", gridcolor="#333"),
        height=420,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # Days to reach IV
    st.divider()
    st.subheader("Days to Reach Intrinsic Value")
    if cp > 0 and iv_base > 0 and cp != iv_base:
        # Using continuous mean reversion: price(t) = IV - (IV-cp)*exp(-k*t)
        # At 30%/yr convergence rate: k = -ln(1-0.3) ≈ 0.357/yr
        if iv_base > cp:
            k = 0.357  # per year
            # When does price reach 95% of the way to IV?
            target = cp + 0.95 * (iv_base - cp)
            # t = -ln((IV - target)/(IV - cp)) / k
            try:
                t_years = -math.log((iv_base - target) / (iv_base - cp)) / k
                days = int(t_years * 365)
                st.metric(
                    "Estimated days to reach 95% of IV (Base)",
                    f"{days:,} days (~{t_years:.1f} years)",
                    help="Based on 30%/yr mean reversion rate at base case IV.",
                )
            except (ValueError, ZeroDivisionError):
                st.metric("Days to IV", "N/A")
        else:
            st.info("Current price is already above intrinsic value — the stock appears overvalued.")
    else:
        st.info("Enter valid price and IV to see days-to-IV estimate.")

    st.caption(
        "Methodology: 1-week uses 2% gap closure; 1-year uses 30% convergence; 5-year uses 70% convergence + "
        "FCF compounding. These are estimates, not guarantees. Actual prices depend on market conditions, "
        "sentiment, and macro factors outside this model."
    )


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    sd = st.session_state.stock_data

    if sd is None:
        # Landing page
        st.title("📊 DCF Valuation Studio")
        st.markdown(
            """
            **A professional two-stage Discounted Cash Flow valuation tool for Indian (NSE/BSE) and US stocks.**

            ### How to use
            1. **Enter a stock ticker** in the sidebar (e.g., `RELIANCE`, `TCS`, `INFY` for Indian stocks or `AAPL`, `MSFT` for US)
            2. **Click "Fetch Data"** — the app pulls financials from Yahoo Finance and/or Screener.in automatically
            3. **Review the pre-filled DCF model** and tweak any assumptions via sliders
            4. **Apply Bull/Base/Bear scenarios** for quick what-if analysis
            5. **Explore the sensitivity heatmap** to see how IV changes with WACC and growth assumptions

            ### Features
            - Two-stage DCF (Stage 1: Y1-5, Stage 2: Y6-10) with Gordon Growth terminal value
            - Auto-populated from live financial data
            - Scenario analysis (Bull/Base/Bear) with auto-calibration to company history
            - 10×10 sensitivity heatmap (any two variables)
            - Price projections for 1 week, 1 year, and 5 years

            ### Disclaimer
            *This tool is for educational and informational purposes only. It does not constitute financial advice.
            Always conduct your own research and consult a qualified financial advisor before making investment decisions.*
            """
        )

        st.divider()

        # Quick demo with sample data
        st.subheader("Quick Demo — try these tickers:")
        demo_cols = st.columns(5)
        demo_tickers = ["RELIANCE", "TCS", "INFY", "AAPL", "MSFT"]
        for i, t in enumerate(demo_tickers):
            with demo_cols[i]:
                if st.button(t, use_container_width=True):
                    st.session_state.last_ticker = t
                    st.rerun()
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🧮 DCF Model",
        "🎭 Scenarios",
        "🔥 Sensitivity",
        "🔮 Projections",
    ])

    # Compute base DCF output (used across tabs)
    try:
        base_dcf_out = calculate_intrinsic_value(st.session_state.dcf_inputs)
    except Exception as e:
        base_dcf_out = None
        st.warning(f"DCF computation issue: {e}")

    with tab1:
        render_overview_tab(sd, base_dcf_out)

    with tab2:
        base_dcf_out = render_dcf_tab(sd)

    with tab3:
        render_scenarios_tab(sd, base_dcf_out)

    with tab4:
        render_sensitivity_tab(sd)

    with tab5:
        render_projections_tab(sd, base_dcf_out)


if __name__ == "__main__":
    main()
