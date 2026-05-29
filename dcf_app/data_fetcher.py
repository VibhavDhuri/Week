"""
Data fetcher module for DCF Valuation Studio.

Supports:
  - Yahoo Finance (yfinance) for US and Indian (NSE/BSE) stocks
  - Screener.in for Indian stocks (scraping)
  - Auto-merge mode: tries both and merges results
"""

import re
import math
import logging
from typing import Any, Dict, Optional, List

import requests
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def parse_indian_number(s: str) -> Optional[float]:
    """
    Parse Indian number formats used on Screener.in.

    Examples:
        "1,234.56 Cr"  -> 1234.56   (already in Crores)
        "123.45 L"     -> 1.2345    (Lakhs → Crores, divide by 100)
        "12.3%"        -> 12.3
        "4.5x"         -> 4.5
        "1,23,456"     -> 123456.0
        "--"           -> None
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("--", "-", "N/A", "n/a", ""):
        return None

    # Extract unit suffix
    unit = ""
    if s.endswith(" Cr"):
        unit = "Cr"
        s = s[:-3].strip()
    elif s.endswith(" L"):
        unit = "L"
        s = s[:-2].strip()
    elif s.endswith("%"):
        unit = "%"
        s = s[:-1].strip()
    elif s.endswith("x"):
        unit = "x"
        s = s[:-1].strip()

    # Remove commas (Indian grouping: 1,23,456)
    s = s.replace(",", "")

    try:
        val = float(s)
    except ValueError:
        return None

    if unit == "L":
        val = val / 100.0  # Lakhs to Crores
    # "Cr", "%", "x", "" → keep as-is
    return val


def _revenue_cagr(rev_list: List[float]) -> Optional[float]:
    """Compute CAGR from a list of annual revenues (oldest → newest)."""
    rev_list = [r for r in rev_list if r and r > 0]
    if len(rev_list) < 2:
        return None
    n = len(rev_list) - 1
    cagr = (rev_list[-1] / rev_list[0]) ** (1.0 / n) - 1.0
    return round(cagr * 100.0, 2)  # return as %


def _format_unit(value: float, currency: str) -> str:
    """Return a display unit label based on currency."""
    if currency in ("INR", "INR"):
        return "Cr"
    return "M"  # USD millions


# ---------------------------------------------------------------------------
# Yahoo Finance Fetcher
# ---------------------------------------------------------------------------

class YahooFetcher:
    """Fetches financial data from Yahoo Finance via yfinance."""

    TICKER_SUFFIXES = ["", ".NS", ".BO"]

    def fetch(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Try ticker as-is, then with .NS, then .BO.
        Returns normalized dict or None if all attempts fail.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return None

        ticker_upper = ticker.strip().upper()
        last_error = None

        for suffix in self.TICKER_SUFFIXES:
            t = ticker_upper + suffix
            try:
                stock = yf.Ticker(t)
                info = stock.info or {}
                # Validate: need at least a current price or market cap
                price = _safe_float(
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose"),
                    default=0.0,
                )
                if price <= 0:
                    continue

                data = self._normalize(stock, info, t)
                if data:
                    return data
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            logger.warning("YahooFetcher failed for %s: %s", ticker, last_error)
        return None

    def _normalize(self, stock, info: Dict[str, Any], ticker_used: str) -> Dict[str, Any]:
        """Normalize yfinance data into a standard dict."""
        currency = info.get("currency", "USD")
        is_indian = currency == "INR" or ticker_used.endswith((".NS", ".BO"))

        # Prices
        current_price = _safe_float(
            info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        )

        # Shares & market cap
        shares_outstanding = _safe_float(info.get("sharesOutstanding"))
        market_cap = _safe_float(info.get("marketCap"))

        # If Indian stock, convert from absolute to Crores
        # yfinance returns INR amounts in absolute (not crores)
        if is_indian and shares_outstanding > 1e7:
            shares_cr = shares_outstanding / 1e7  # absolute → crores
        else:
            shares_cr = shares_outstanding / 1e6  # absolute → millions

        # Revenue & earnings
        total_revenue = _safe_float(info.get("totalRevenue"))
        net_income = _safe_float(info.get("netIncomeToCommon"))
        free_cashflow = _safe_float(info.get("freeCashflow"))
        operating_cashflow = _safe_float(info.get("operatingCashflow"))
        ebitda = _safe_float(info.get("ebitda"))

        # Scale to Cr (Indian) or M (USD)
        if is_indian:
            scale = 1e7  # 1 crore = 10M
            total_revenue_scaled = total_revenue / scale
            net_income_scaled = net_income / scale
            fcf_scaled = free_cashflow / scale
            ocf_scaled = operating_cashflow / scale
            ebitda_scaled = ebitda / scale
            market_cap_scaled = market_cap / scale
        else:
            scale = 1e6  # millions
            total_revenue_scaled = total_revenue / scale
            net_income_scaled = net_income / scale
            fcf_scaled = free_cashflow / scale
            ocf_scaled = operating_cashflow / scale
            ebitda_scaled = ebitda / scale
            market_cap_scaled = market_cap / scale

        # Debt & cash
        total_debt = _safe_float(info.get("totalDebt"))
        total_cash = _safe_float(info.get("totalCash"))
        net_debt_abs = total_debt - total_cash
        net_debt_scaled = net_debt_abs / scale if scale else 0.0

        # Ratios (already dimensionless from yfinance)
        profit_margins = _safe_float(info.get("profitMargins"))
        operating_margins = _safe_float(info.get("operatingMargins"))
        revenue_growth = _safe_float(info.get("revenueGrowth"))
        earnings_growth = _safe_float(info.get("earningsGrowth"))
        roe = _safe_float(info.get("returnOnEquity"))
        roa = _safe_float(info.get("returnOnAssets"))

        # Book value
        book_value = _safe_float(info.get("bookValue"))  # per share

        # Fetch multi-year financials
        hist_revenues, hist_net_incomes, hist_fcfs = self._fetch_historical(stock, is_indian, scale)
        historical_revenue_cagr = _revenue_cagr(hist_revenues)

        # FCF: prefer actual FCF, fallback to OCF - Capex estimate
        if fcf_scaled == 0.0 and ocf_scaled != 0.0:
            fcf_scaled = ocf_scaled * 0.7  # rough estimate

        # FCF margin
        fcf_margin_pct = 0.0
        if total_revenue_scaled > 0 and fcf_scaled != 0.0:
            fcf_margin_pct = (fcf_scaled / total_revenue_scaled) * 100.0

        return {
            # Identity
            "name": info.get("longName") or info.get("shortName") or ticker_used,
            "ticker_used": ticker_used,
            "source": "yahoo",
            "currency": currency,
            "unit_label": "Cr" if is_indian else "M",

            # Prices
            "current_price": current_price,
            "market_cap": market_cap_scaled,

            # Shares
            "shares_outstanding": shares_cr if is_indian else shares_outstanding / 1e6,

            # Income statement
            "total_revenue": total_revenue_scaled,
            "net_income": net_income_scaled,
            "ebitda": ebitda_scaled,

            # Cash flow
            "free_cashflow": fcf_scaled,
            "operating_cashflow": ocf_scaled,

            # Derived
            "net_debt": net_debt_scaled,
            "total_debt": total_debt / scale,
            "total_cash": total_cash / scale,
            "book_value": book_value,

            # Ratios
            "beta": _safe_float(info.get("beta"), default=1.0),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "profit_margins": profit_margins,
            "operating_margins": operating_margins,
            "revenue_growth": revenue_growth * 100.0 if revenue_growth else 0.0,
            "earnings_growth": earnings_growth * 100.0 if earnings_growth else 0.0,
            "roe": roe * 100.0 if roe else 0.0,
            "roa": roa * 100.0 if roa else 0.0,
            "dividend_yield": _safe_float(info.get("dividendYield")) * 100.0,
            "fcf_margin": fcf_margin_pct,

            # Sector/Industry
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "description": info.get("longBusinessSummary", ""),

            # Historical
            "hist_revenues": hist_revenues,
            "hist_net_incomes": hist_net_incomes,
            "hist_fcfs": hist_fcfs,
            "historical_revenue_cagr": historical_revenue_cagr,
        }

    def _fetch_historical(self, stock, is_indian: bool, scale: float):
        """Fetch multi-year historical revenues, net incomes, and FCFs."""
        hist_revenues = []
        hist_net_incomes = []
        hist_fcfs = []

        try:
            financials = stock.financials  # income statement, annual
            if financials is not None and not financials.empty:
                # Columns are dates (newest first), rows are line items
                rev_row = None
                ni_row = None
                for idx in financials.index:
                    idx_lower = str(idx).lower()
                    if "total revenue" in idx_lower or "revenue" in idx_lower:
                        rev_row = financials.loc[idx]
                    if "net income" in idx_lower:
                        ni_row = financials.loc[idx]

                if rev_row is not None:
                    vals = [_safe_float(v) / scale for v in reversed(rev_row.values)]
                    hist_revenues = [v for v in vals if v != 0.0]

                if ni_row is not None:
                    vals = [_safe_float(v) / scale for v in reversed(ni_row.values)]
                    hist_net_incomes = [v for v in vals if v != 0.0]

        except Exception as exc:
            logger.debug("Failed to fetch financials: %s", exc)

        try:
            cashflow = stock.cashflow  # annual
            if cashflow is not None and not cashflow.empty:
                fcf_row = None
                ocf_row = None
                capex_row = None
                for idx in cashflow.index:
                    idx_lower = str(idx).lower()
                    if "free cash flow" in idx_lower:
                        fcf_row = cashflow.loc[idx]
                    if "operating" in idx_lower and "cash" in idx_lower:
                        ocf_row = cashflow.loc[idx]
                    if "capital expenditure" in idx_lower or "capex" in idx_lower:
                        capex_row = cashflow.loc[idx]

                if fcf_row is not None:
                    vals = [_safe_float(v) / scale for v in reversed(fcf_row.values)]
                    hist_fcfs = [v for v in vals if v != 0.0]
                elif ocf_row is not None and capex_row is not None:
                    ocf_vals = list(reversed(ocf_row.values))
                    cap_vals = list(reversed(capex_row.values))
                    for o, c in zip(ocf_vals, cap_vals):
                        fcf = (_safe_float(o) + _safe_float(c)) / scale  # capex is negative
                        if fcf != 0.0:
                            hist_fcfs.append(fcf)

        except Exception as exc:
            logger.debug("Failed to fetch cashflow: %s", exc)

        return hist_revenues, hist_net_incomes, hist_fcfs


# ---------------------------------------------------------------------------
# Screener.in Fetcher
# ---------------------------------------------------------------------------

class ScreenerFetcher:
    """Scrapes financial data from Screener.in for Indian stocks."""

    BASE_URL = "https://www.screener.in/company/{symbol}/{mode}/"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def fetch(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Scrape Screener.in. Try consolidated first, then standalone."""
        symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
        data = self._scrape(symbol, "consolidated")
        if data is None:
            data = self._scrape(symbol, "")
        return data

    def _scrape(self, symbol: str, mode: str) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL.format(symbol=symbol, mode=mode) if mode else \
              f"https://www.screener.in/company/{symbol}/"
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200:
                return None
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            return self._parse(soup, symbol, url)
        except Exception as exc:
            logger.warning("ScreenerFetcher._scrape error for %s: %s", symbol, exc)
            return None

    def _parse(self, soup, symbol: str, url: str) -> Optional[Dict[str, Any]]:
        """Parse the Screener page."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        data: Dict[str, Any] = {
            "name": symbol,
            "ticker_used": symbol,
            "source": "screener",
            "currency": "INR",
            "unit_label": "Cr",
            "screener_url": url,
        }

        # --- Company name ---
        name_tag = soup.find("h1", class_="company-name") or soup.find("h1")
        if name_tag:
            data["name"] = name_tag.get_text(strip=True)

        # --- Top ratios ---
        top_ratios = soup.find(id="top-ratios")
        if top_ratios:
            for li in top_ratios.find_all("li"):
                name_span = li.find("span", class_="name")
                value_span = li.find("span", class_="number") or li.find("span", class_="value")
                if not name_span or not value_span:
                    continue
                key = name_span.get_text(strip=True).lower()
                raw_val = value_span.get_text(strip=True)
                val = parse_indian_number(raw_val)

                if "market cap" in key:
                    data["market_cap"] = val
                elif "current price" in key or "stock p/e" in key and "current" in key:
                    if "current price" in key:
                        data["current_price"] = val
                elif "stock p/e" in key:
                    data["trailing_pe"] = val
                elif "book value" in key:
                    data["book_value"] = val
                elif "dividend yield" in key:
                    data["dividend_yield"] = val
                elif "roce" in key:
                    data["roe"] = val  # approximate with ROCE
                elif "roe" in key:
                    data["roe"] = val
                elif "face value" in key:
                    data["face_value"] = val
                elif "eps" in key:
                    data["eps"] = val

        # Also check company-ratios
        ratios_section = soup.find(id="company-ratios") or soup.find("ul", class_="company-ratios")
        if ratios_section:
            for li in ratios_section.find_all("li"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    key = spans[0].get_text(strip=True).lower()
                    raw_val = spans[-1].get_text(strip=True)
                    val = parse_indian_number(raw_val)
                    if "current price" in key and val:
                        data["current_price"] = val
                    elif "52 week high" in key and val:
                        data["52w_high"] = val
                    elif "52 week low" in key and val:
                        data["52w_low"] = val

        # Parse current price from the big number if top-ratios missed it
        if not data.get("current_price"):
            price_tag = soup.find("span", class_="number", id=re.compile(r"(?i)price|current"))
            if price_tag:
                data["current_price"] = parse_indian_number(price_tag.get_text(strip=True))

        # --- P&L Table ---
        pl_data = self._parse_table(soup, "profit-loss")
        bs_data = self._parse_table(soup, "balance-sheet")
        cf_data = self._parse_table(soup, "cash-flow")

        data["pl_table"] = pl_data
        data["bs_table"] = bs_data
        data["cf_table"] = cf_data

        # Extract key annual figures from tables
        years, revenues = self._extract_row(pl_data, ["sales", "revenue", "net sales"])
        _, net_profits = self._extract_row(pl_data, ["net profit", "pat"])
        _, operating_profits = self._extract_row(pl_data, ["operating profit", "ebitda", "ebit"])
        _, fcfs = self._extract_row(cf_data, ["free cash flow", "fcf"])
        _, ocfs = self._extract_row(cf_data, ["cash from operations", "operating activities", "operating"])
        _, capex_vals = self._extract_row(cf_data, ["capital expenditure", "capex", "investing"])

        # Most recent values (last column = latest)
        data["total_revenue"] = revenues[-1] if revenues else 0.0
        data["net_income"] = net_profits[-1] if net_profits else 0.0
        data["ebitda"] = operating_profits[-1] if operating_profits else 0.0

        # FCF
        if fcfs:
            data["free_cashflow"] = fcfs[-1]
        elif ocfs and capex_vals:
            data["free_cashflow"] = ocfs[-1] - abs(capex_vals[-1])
        elif ocfs:
            data["free_cashflow"] = ocfs[-1] * 0.75
        else:
            data["free_cashflow"] = 0.0

        data["operating_cashflow"] = ocfs[-1] if ocfs else 0.0

        # Historical for CAGR
        data["hist_revenues"] = revenues if revenues else []
        data["hist_net_incomes"] = net_profits if net_profits else []
        data["hist_fcfs"] = fcfs if fcfs else []
        data["historical_revenue_cagr"] = _revenue_cagr(revenues)

        # Margins
        if data["total_revenue"] and data["total_revenue"] > 0:
            data["profit_margins"] = (data["net_income"] / data["total_revenue"])
            if data["free_cashflow"]:
                data["fcf_margin"] = (data["free_cashflow"] / data["total_revenue"]) * 100.0
            else:
                data["fcf_margin"] = 0.0
        else:
            data["profit_margins"] = 0.0
            data["fcf_margin"] = 0.0

        # Balance sheet
        _, total_borrowings = self._extract_row(bs_data, ["borrowings", "total debt", "long term borrowing"])
        _, cash_equivalents = self._extract_row(bs_data, ["cash", "cash equivalents", "investments"])
        _, equity = self._extract_row(bs_data, ["equity", "shareholders equity", "total equity"])

        total_debt = total_borrowings[-1] if total_borrowings else 0.0
        total_cash = cash_equivalents[-1] if cash_equivalents else 0.0
        data["total_debt"] = total_debt
        data["total_cash"] = total_cash
        data["net_debt"] = total_debt - total_cash

        # Shares outstanding estimate from market cap and price
        if data.get("market_cap") and data.get("current_price") and data["current_price"] > 0:
            # market cap is in Cr, price is per share in INR
            # shares (Cr) = market_cap_Cr / (price * 1e7 / 1e7) → market_cap / price gives crores
            shares_cr = data["market_cap"] / data["current_price"]
            data["shares_outstanding"] = shares_cr  # in Crores

        # Revenue growth (YoY latest)
        if len(revenues) >= 2 and revenues[-2] > 0:
            data["revenue_growth"] = ((revenues[-1] / revenues[-2]) - 1.0) * 100.0
        else:
            data["revenue_growth"] = 0.0

        data["beta"] = data.get("beta", 1.0)

        return data

    def parse_table(self, soup, table_id: str) -> Dict[str, Any]:
        """Public wrapper for _parse_table."""
        return self._parse_table(soup, table_id)

    def _parse_table(self, soup, table_id: str) -> Dict[str, Any]:
        """
        Parse a financial table from Screener.in by its section id.
        Returns dict: {row_label: [values_per_year], 'years': [year_labels]}
        """
        result: Dict[str, Any] = {"years": []}
        section = soup.find(id=table_id)
        if not section:
            return result

        table = section.find("table")
        if not table:
            # Try tbody directly
            table = section

        thead = table.find("thead")
        if thead:
            header_cells = thead.find_all("th")
            result["years"] = [th.get_text(strip=True) for th in header_cells[1:]]

        tbody = table.find("tbody")
        if not tbody:
            return result

        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            row_label = cells[0].get_text(strip=True).lower().strip()
            values = []
            for td in cells[1:]:
                raw = td.get_text(strip=True)
                val = parse_indian_number(raw)
                values.append(val if val is not None else 0.0)
            if row_label and values:
                result[row_label] = values

        return result

    def _extract_row(self, table: Dict[str, Any], keywords: List[str]):
        """
        Find the first row in a parsed table matching any of the keywords (substring match).
        Returns (years, values) tuple.
        """
        years = table.get("years", [])
        for key in keywords:
            for row_label, values in table.items():
                if row_label == "years":
                    continue
                if key.lower() in row_label.lower():
                    return years, [v for v in values if isinstance(v, (int, float))]
        return years, []


# ---------------------------------------------------------------------------
# Main DataFetcher
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    Unified fetcher that tries Yahoo Finance and/or Screener.in
    and returns a clean StockData dict ready for DCF inputs.
    """

    def __init__(self):
        self.yahoo = YahooFetcher()
        self.screener = ScreenerFetcher()

    def fetch(self, ticker: str, source: str = "auto") -> Optional[Dict[str, Any]]:
        """
        Fetch stock data.

        source: 'auto' | 'yahoo' | 'screener'
        Returns a StockData dict or None.
        """
        ticker = ticker.strip().upper()
        yahoo_data = None
        screener_data = None

        if source in ("auto", "yahoo"):
            yahoo_data = self.yahoo.fetch(ticker)

        is_indian = self._is_indian(ticker, yahoo_data)

        if source in ("auto", "screener") and is_indian:
            clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
            screener_data = self.screener.fetch(clean_ticker)

        if source == "yahoo":
            return self._finalize(yahoo_data) if yahoo_data else None

        if source == "screener":
            return self._finalize(screener_data) if screener_data else None

        # auto: merge
        if yahoo_data and screener_data:
            merged = self._merge(yahoo_data, screener_data)
            return self._finalize(merged)
        elif yahoo_data:
            return self._finalize(yahoo_data)
        elif screener_data:
            return self._finalize(screener_data)
        else:
            return None

    def _is_indian(self, ticker: str, yahoo_data: Optional[Dict]) -> bool:
        if ".NS" in ticker or ".BO" in ticker:
            return True
        if yahoo_data and yahoo_data.get("currency") == "INR":
            return True
        # heuristic: if ticker has no dot and no digits, could be Indian
        if "." not in ticker and not any(c.isdigit() for c in ticker):
            return True
        return False

    def _merge(self, yahoo: Dict[str, Any], screener: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge Yahoo and Screener data. Prefer Screener for Indian financials
        (more accurate Cr figures), Yahoo for prices and ratios.
        """
        merged = dict(yahoo)  # start with Yahoo

        # Prefer Screener for fundamental data
        for key in [
            "total_revenue", "net_income", "free_cashflow", "operating_cashflow",
            "ebitda", "net_debt", "total_debt", "total_cash", "hist_revenues",
            "hist_net_incomes", "hist_fcfs", "historical_revenue_cagr", "roe",
            "fcf_margin", "profit_margins",
        ]:
            screener_val = screener.get(key)
            if screener_val is not None and screener_val != 0.0:
                merged[key] = screener_val

        # Screener sector/name sometimes better
        if screener.get("name") and len(screener.get("name", "")) > len(merged.get("name", "")):
            merged["name"] = screener["name"]

        merged["source"] = "yahoo+screener"
        merged["screener_url"] = screener.get("screener_url", "")

        return merged

    def _finalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all required fields exist with sensible defaults.
        Compute derived fields needed for DCF.
        """
        d = data.copy()

        # Ensure defaults
        defaults = {
            "name": "Unknown",
            "ticker_used": "",
            "source": "unknown",
            "currency": "USD",
            "unit_label": "M",
            "current_price": 0.0,
            "market_cap": 0.0,
            "shares_outstanding": 0.0,
            "total_revenue": 0.0,
            "net_income": 0.0,
            "free_cashflow": 0.0,
            "operating_cashflow": 0.0,
            "ebitda": 0.0,
            "net_debt": 0.0,
            "total_debt": 0.0,
            "total_cash": 0.0,
            "book_value": 0.0,
            "beta": 1.0,
            "trailing_pe": 0.0,
            "forward_pe": 0.0,
            "profit_margins": 0.0,
            "operating_margins": 0.0,
            "revenue_growth": 0.0,
            "earnings_growth": 0.0,
            "roe": 0.0,
            "roa": 0.0,
            "dividend_yield": 0.0,
            "fcf_margin": 0.0,
            "sector": "",
            "industry": "",
            "description": "",
            "hist_revenues": [],
            "hist_net_incomes": [],
            "hist_fcfs": [],
            "historical_revenue_cagr": None,
            "screener_url": "",
        }

        for k, v in defaults.items():
            if k not in d or d[k] is None:
                d[k] = v
            elif isinstance(d[k], float) and (math.isnan(d[k]) or math.isinf(d[k])):
                d[k] = v

        # Derive shares if missing
        if d["shares_outstanding"] == 0.0 and d["market_cap"] > 0 and d["current_price"] > 0:
            d["shares_outstanding"] = d["market_cap"] / d["current_price"]

        # Derive FCF margin if missing
        if d["fcf_margin"] == 0.0 and d["total_revenue"] > 0 and d["free_cashflow"] != 0.0:
            d["fcf_margin"] = (d["free_cashflow"] / d["total_revenue"]) * 100.0

        # Derive profit_margins (as decimal for compatibility)
        if d["profit_margins"] == 0.0 and d["total_revenue"] > 0 and d["net_income"] != 0.0:
            d["profit_margins"] = d["net_income"] / d["total_revenue"]

        # Historical revenue CAGR
        if d["historical_revenue_cagr"] is None and d["hist_revenues"]:
            d["historical_revenue_cagr"] = _revenue_cagr(d["hist_revenues"])

        # Convert profit_margins to percentage if stored as decimal
        if abs(d["profit_margins"]) < 1.0 and d["profit_margins"] != 0.0:
            d["profit_margins_pct"] = d["profit_margins"] * 100.0
        else:
            d["profit_margins_pct"] = d["profit_margins"]

        return d
