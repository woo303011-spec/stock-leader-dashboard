import hashlib
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="S&P 500 Sector Leaders (1926-2026)", layout="wide")


START_YEAR = 1926
END_YEAR = 2026
INITIAL_CAPITAL = 10_000.0

SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]


@dataclass
class LeaderEvent:
    year: int
    ticker: str
    name: str
    base_market_cap: float
    base_return: float


LEADER_SEEDS: Dict[str, List[LeaderEvent]] = {
    "Communication Services": [
        LeaderEvent(1926, "RCA_OLD", "RCA (legacy)", 3.0e9, 0.06),
        LeaderEvent(1945, "ATT_OLD", "AT&T (legacy)", 7.0e9, 0.07),
        LeaderEvent(2015, "GOOGL", "Alphabet", 5.2e11, 0.15),
    ],
    "Consumer Discretionary": [
        LeaderEvent(1926, "GM_OLD", "General Motors (legacy)", 1.8e9, 0.08),
        LeaderEvent(2018, "AMZN", "Amazon", 7.6e11, 0.20),
    ],
    "Consumer Staples": [
        LeaderEvent(1926, "PG", "Procter & Gamble", 1.4e9, 0.07),
        LeaderEvent(2005, "WMT", "Walmart", 1.9e11, 0.10),
    ],
    "Energy": [
        LeaderEvent(1926, "SOCONY", "Standard Oil of New York", 2.4e9, 0.09),
        LeaderEvent(2002, "XOM", "Exxon Mobil", 2.8e11, 0.10),
    ],
    "Financials": [
        LeaderEvent(1926, "AIG_OLD", "AIG (legacy)", 1.2e9, 0.08),
        LeaderEvent(2010, "BRK-B", "Berkshire Hathaway", 2.1e11, 0.12),
    ],
    "Health Care": [
        LeaderEvent(1926, "MRK", "Merck", 1.3e9, 0.09),
        LeaderEvent(2021, "LLY", "Eli Lilly", 2.45e11, 0.18),
    ],
    "Industrials": [
        LeaderEvent(1926, "USX_OLD", "U.S. Steel (legacy)", 2.2e9, 0.07),
        LeaderEvent(2023, "CAT", "Caterpillar", 1.45e11, 0.14),
    ],
    "Information Technology": [
        LeaderEvent(1926, "IBM", "IBM", 1.8e9, 0.10),
        LeaderEvent(2023, "NVDA", "NVIDIA", 1.1e12, 0.26),
    ],
    "Materials": [
        LeaderEvent(1926, "DD_OLD", "DuPont (legacy)", 1.6e9, 0.08),
        LeaderEvent(2019, "LIN", "Linde", 9.8e10, 0.13),
    ],
    "Real Estate": [
        LeaderEvent(1926, "REIT_OLD", "Legacy Realty Trust", 6.0e8, 0.06),
        LeaderEvent(2022, "WELL", "Welltower", 5.8e10, 0.11),
    ],
    "Utilities": [
        LeaderEvent(1926, "UTIL_OLD", "Legacy Utility Holdings", 8.0e8, 0.06),
        LeaderEvent(2007, "NEE", "NextEra Energy", 5.3e10, 0.10),
    ],
}


def deterministic_noise(*keys: str) -> float:
    key = "|".join(keys).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    n = int(digest[:8], 16) / 0xFFFFFFFF
    return (n * 2.0) - 1.0  # [-1, 1]


def active_event_for_year(events: List[LeaderEvent], year: int) -> LeaderEvent:
    active = events[0]
    for e in events:
        if e.year <= year:
            active = e
        else:
            break
    return active


@st.cache_data
def build_sector_history() -> pd.DataFrame:
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        for sector in SECTORS:
            event = active_event_for_year(LEADER_SEEDS[sector], year)
            noise_ret = deterministic_noise(str(year), sector, event.ticker) * 0.055
            annual_return = max(-0.65, min(1.10, event.base_return + noise_ret))
            cap_growth = 1.0 + max(0.0, year - event.year) * 0.028
            market_cap = event.base_market_cap * cap_growth
            rows.append(
                {
                    "Year": year,
                    "Sector": sector,
                    "Ticker": event.ticker,
                    "CompanyName": event.name,
                    "MarketCap": float(round(market_cap, 2)),
                    "AnnualReturn": float(round(annual_return, 6)),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data
def build_sp500_returns() -> Dict[int, float]:
    out = {}
    for year in range(START_YEAR, END_YEAR + 1):
        base = 0.085 + deterministic_noise("spx", str(year)) * 0.03
        if 1929 <= year <= 1932:
            base -= 0.28
        if year in (1973, 1974):
            base -= 0.18
        if 2000 <= year <= 2002:
            base -= 0.16
        if year == 2008:
            base -= 0.33
        if year == 2020:
            base -= 0.07
        out[year] = max(-0.60, min(0.55, base))
    return out


def run_backtest(history: pd.DataFrame, benchmark: Dict[int, float]):
    sector_weight = 1.0 / len(SECTORS)
    by_year = {y: g.sort_values("Sector") for y, g in history.groupby("Year")}
    years = sorted(by_year.keys())

    strategy_value = INITIAL_CAPITAL
    benchmark_value = INITIAL_CAPITAL
    curve = []
    changes = []
    prev_map = None

    for year in years:
        year_df = by_year[year]
        cur_map = {r.Sector: r.Ticker for r in year_df.itertuples()}

        if prev_map is not None:
            for sec in SECTORS:
                if prev_map.get(sec) != cur_map.get(sec):
                    old_row = year_df[year_df["Sector"] == sec].iloc[0]
                    changes.append(
                        {
                            "Year": year,
                            "Sector": sec,
                            "OutTicker": prev_map.get(sec),
                            "InTicker": cur_map.get(sec),
                            "InCompanyName": old_row["CompanyName"],
                        }
                    )

        growth = 0.0
        for r in year_df.itertuples():
            growth += sector_weight * (1.0 + float(r.AnnualReturn))

        strategy_value *= growth
        benchmark_value *= (1.0 + benchmark[year])
        curve.append(
            {
                "Year": year,
                "StrategyValue": round(strategy_value, 2),
                "BenchmarkValue": round(benchmark_value, 2),
                "StrategyAnnualReturnPct": round((growth - 1.0) * 100.0, 2),
                "BenchmarkAnnualReturnPct": round(benchmark[year] * 100.0, 2),
                "ChangeCount": 0,
            }
        )
        prev_map = cur_map

    curve_df = pd.DataFrame(curve)
    change_df = pd.DataFrame(changes)
    if not change_df.empty:
        counts = change_df.groupby("Year").size().to_dict()
        curve_df["ChangeCount"] = curve_df["Year"].map(counts).fillna(0).astype(int)

    return curve_df, change_df


def decade_membership_table(history: pd.DataFrame) -> pd.DataFrame:
    years = [1926, 1936, 1946, 1956, 1966, 1976, 1986, 1996, 2006, 2016, 2026]
    parts = []
    for y in years:
        sub = history[history["Year"] == y][["Sector", "Ticker"]].copy()
        sub = sub.rename(columns={"Ticker": f"{y}"})
        parts.append(sub.set_index("Sector"))
    out = pd.concat(parts, axis=1).reset_index()
    return out


def event_window_stats(curve_df: pd.DataFrame) -> pd.DataFrame:
    event_windows = [
        ("1929 대공황", 1929, 1933),
        ("1970s 오일쇼크", 1973, 1975),
        ("2000 닷컴버블", 2000, 2002),
        ("2008 금융위기", 2007, 2009),
        ("2020 코로나", 2020, 2021),
    ]
    by_year = {int(r.Year): r for r in curve_df.itertuples()}
    rows = []
    for name, s, e in event_windows:
        if s in by_year and e in by_year:
            sv0 = float(by_year[s].StrategyValue)
            sv1 = float(by_year[e].StrategyValue)
            bv0 = float(by_year[s].BenchmarkValue)
            bv1 = float(by_year[e].BenchmarkValue)
            rows.append(
                {
                    "Event": name,
                    "Start": s,
                    "End": e,
                    "StrategyWindowReturnPct": round((sv1 / sv0 - 1.0) * 100.0, 2),
                    "SP500WindowReturnPct": round((bv1 / bv0 - 1.0) * 100.0, 2),
                }
            )
    return pd.DataFrame(rows)


st.title("S&P 500 11개 섹터 대장주 교체 전략 (1926-2026)")
st.caption("가상/정적 역사 데이터 기반. 섹터 대장주 교체를 연초 리밸런싱으로 반영합니다.")

history_df = build_sector_history()
spx_returns = build_sp500_returns()
curve_df, change_df = run_backtest(history_df, spx_returns)
decade_df = decade_membership_table(history_df)
event_df = event_window_stats(curve_df)

log_scale = st.checkbox("로그 스케일", value=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=curve_df["Year"],
        y=curve_df["StrategyValue"],
        mode="lines",
        name="Sector Leaders Strategy",
        line=dict(width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=curve_df["Year"],
        y=curve_df["BenchmarkValue"],
        mode="lines",
        name="S&P 500 Benchmark",
        line=dict(width=2),
    )
)

marker_df = curve_df[curve_df["ChangeCount"] > 0]
fig.add_trace(
    go.Scatter(
        x=marker_df["Year"],
        y=marker_df["StrategyValue"],
        mode="markers+text",
        name="Leader Changes",
        text=marker_df["ChangeCount"].astype(str),
        textposition="top center",
        marker=dict(size=8, color="orange"),
    )
)

fig.update_layout(
    title="전략 누적 수익률 vs S&P 500",
    xaxis_title="Year",
    yaxis_title="Portfolio Value (USD)",
    yaxis_type="log" if log_scale else "linear",
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("전략 최종자산", f"${curve_df.iloc[-1]['StrategyValue']:,.0f}")
col2.metric("S&P500 최종자산", f"${curve_df.iloc[-1]['BenchmarkValue']:,.0f}")
col3.metric(
    "초과성과",
    f"{((curve_df.iloc[-1]['StrategyValue'] / curve_df.iloc[-1]['BenchmarkValue']) - 1.0) * 100.0:.2f}%",
)

st.subheader("섹터 대장주 교체 이벤트")
if change_df.empty:
    st.info("교체 이벤트가 없습니다.")
else:
    st.dataframe(change_df, use_container_width=True, hide_index=True)

st.subheader("10년 단위 섹터 편입 종목")
st.dataframe(decade_df, use_container_width=True, hide_index=True)

st.subheader("주요 경제 이벤트 구간 성과")
if event_df.empty:
    st.info("이벤트 구간 데이터가 없습니다.")
else:
    st.dataframe(event_df, use_container_width=True, hide_index=True)

st.caption("참고: 100년 데이터는 정적/가상 시뮬레이션 기반이며 교육/연구 목적입니다.")
