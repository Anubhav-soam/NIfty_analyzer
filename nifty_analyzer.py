"""Core analysis engine for NIFTY 50 market insights.

Disclaimer: Educational use only. Not financial advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Signal:
    strategy: str
    action: str
    score: int
    confidence: float
    reasoning: str
    entry: float | None = None
    stop_loss: float | None = None
    target: float | None = None


class NiftyAnalyzer:
    def __init__(self, symbol: str = "^NSEI") -> None:
        self.symbol = symbol

    def fetch_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        df = yf.download(self.symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("No data fetched. Check internet connection or symbol.")
        return df.dropna().copy()

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["Close"]

        out["Ret1D"] = close.pct_change()
        out["SMA20"] = close.rolling(20).mean()
        out["SMA50"] = close.rolling(50).mean()
        out["SMA200"] = close.rolling(200).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out["RSI14"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["MACD"] = ema12 - ema26
        out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

        std20 = close.rolling(20).std()
        out["BB_upper"] = out["SMA20"] + 2 * std20
        out["BB_lower"] = out["SMA20"] - 2 * std20

        tr = pd.concat(
            [
                out["High"] - out["Low"],
                (out["High"] - close.shift()).abs(),
                (out["Low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["ATR14"] = tr.rolling(14).mean()

        out["Vol20"] = out["Ret1D"].rolling(20).std() * np.sqrt(252) * 100
        return out

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
        window = df.tail(lookback)
        return {
            "support": float(window["Low"].min()),
            "resistance": float(window["High"].max()),
        }

    @staticmethod
    def _confidence_from_score(score: int) -> float:
        return max(0.5, min(0.9, 0.5 + score * 0.08))

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(latest["Close"])
        atr = float(latest.get("ATR14", 0) or 0)
        sr = self.detect_support_resistance(df)

        bull_score = 0
        bear_score = 0
        reasons_bull: List[str] = []
        reasons_bear: List[str] = []

        if latest["SMA50"] > latest["SMA200"]:
            bull_score += 1
            reasons_bull.append("50DMA above 200DMA")
        else:
            bear_score += 1
            reasons_bear.append("50DMA below 200DMA")

        if close > latest["SMA20"] > latest["SMA50"]:
            bull_score += 1
            reasons_bull.append("price above key moving averages")
        if close < latest["SMA20"] < latest["SMA50"]:
            bear_score += 1
            reasons_bear.append("price below key moving averages")

        if latest["MACD"] > latest["MACD_signal"]:
            bull_score += 1
            reasons_bull.append("MACD above signal")
        else:
            bear_score += 1
            reasons_bear.append("MACD below signal")

        if latest["RSI14"] > 55:
            bull_score += 1
            reasons_bull.append("RSI in bullish momentum zone")
        elif latest["RSI14"] < 45:
            bear_score += 1
            reasons_bear.append("RSI in weak momentum zone")

        signals: List[Signal] = []
        if bull_score >= 3:
            conf = self._confidence_from_score(bull_score)
            signals.append(
                Signal(
                    strategy="Trend Following",
                    action="ENTER LONG",
                    score=bull_score,
                    confidence=conf,
                    reasoning=", ".join(reasons_bull),
                    entry=close,
                    stop_loss=close - 1.5 * atr if atr else close * 0.98,
                    target=close + 3 * atr if atr else close * 1.04,
                )
            )
        if bear_score >= 3:
            conf = self._confidence_from_score(bear_score)
            signals.append(
                Signal(
                    strategy="Trend Following",
                    action="EXIT LONG / CONSIDER SHORT",
                    score=bear_score,
                    confidence=conf,
                    reasoning=", ".join(reasons_bear),
                    entry=close,
                    stop_loss=close + 1.5 * atr if atr else close * 1.02,
                    target=close - 3 * atr if atr else close * 0.96,
                )
            )

        if prev["Close"] < prev["BB_lower"] and latest["Close"] > latest["BB_lower"] and latest["RSI14"] < 35:
            signals.append(
                Signal(
                    strategy="Mean Reversion",
                    action="ENTER LONG (BOUNCE)",
                    score=2,
                    confidence=0.66,
                    reasoning="price re-entered Bollinger band from oversold zone",
                    entry=close,
                    stop_loss=min(sr["support"], close - atr),
                    target=float(latest["SMA20"]),
                )
            )
        if prev["Close"] > prev["BB_upper"] and latest["Close"] < latest["BB_upper"] and latest["RSI14"] > 65:
            signals.append(
                Signal(
                    strategy="Mean Reversion",
                    action="BOOK PROFITS / TACTICAL SHORT",
                    score=2,
                    confidence=0.64,
                    reasoning="price re-entered from overbought Bollinger stretch",
                    entry=close,
                    stop_loss=max(sr["resistance"], close + atr),
                    target=float(latest["SMA20"]),
                )
            )

        if not signals:
            signals.append(
                Signal(
                    strategy="Wait and Watch",
                    action="NO TRADE",
                    score=max(bull_score, bear_score),
                    confidence=0.55,
                    reasoning="mixed signals, wait for trend + momentum alignment",
                )
            )
        return signals

    @staticmethod
    def backtest_trend_strategy(df: pd.DataFrame) -> Dict[str, float]:
        bt = df.copy()
        bt["trend_long"] = ((bt["SMA50"] > bt["SMA200"]) & (bt["MACD"] > bt["MACD_signal"]) & (bt["RSI14"] > 50)).astype(int)
        bt["position"] = bt["trend_long"].shift(1).fillna(0)
        bt["strategy_ret"] = bt["position"] * bt["Ret1D"]

        cum_market = float((1 + bt["Ret1D"].fillna(0)).prod() - 1)
        cum_strategy = float((1 + bt["strategy_ret"].fillna(0)).prod() - 1)

        equity = (1 + bt["strategy_ret"].fillna(0)).cumprod()
        peak = equity.cummax()
        max_dd = float(((equity / peak) - 1).min()) if len(equity) else 0.0

        trades = int((bt["position"].diff() == 1).sum())
        return {
            "market_return_pct": cum_market * 100,
            "strategy_return_pct": cum_strategy * 100,
            "strategy_max_drawdown_pct": max_dd * 100,
            "entries": trades,
        }

    @staticmethod
    def build_insights(df: pd.DataFrame) -> Dict[str, float | str]:
        latest = df.iloc[-1]
        yearly_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        rolling_high = df["High"].rolling(252).max().iloc[-1]
        drawdown = (latest["Close"] / rolling_high - 1) * 100 if rolling_high else np.nan

        regime = "Bullish" if latest["SMA50"] > latest["SMA200"] else "Bearish"
        if 45 <= latest["RSI14"] <= 55:
            momentum = "Neutral"
        elif latest["RSI14"] > 55:
            momentum = "Positive"
        else:
            momentum = "Weak"

        return {
            "close": float(latest["Close"]),
            "yearly_return_pct": float(yearly_return),
            "drawdown_from_1y_high_pct": float(drawdown),
            "market_regime": regime,
            "momentum": momentum,
            "annualized_volatility_pct": float(latest.get("Vol20", np.nan)),
        }


class AnalysisService:
    def __init__(self, symbol: str = "^NSEI") -> None:
        self.analyzer = NiftyAnalyzer(symbol=symbol)

    def run(self) -> Dict[str, Any]:
        hist = self.analyzer.fetch_data(period="1y", interval="1d")
        hist = self.analyzer.add_indicators(hist).dropna()
        live = self.analyzer.fetch_data(period="1d", interval="5m")

        signals = self.analyzer.generate_signals(hist)
        insights = self.analyzer.build_insights(hist)
        backtest = self.analyzer.backtest_trend_strategy(hist)
        sr = self.analyzer.detect_support_resistance(hist)

        chart_df = hist.tail(260).reset_index()
        if "Date" not in chart_df.columns:
            chart_df = chart_df.rename(columns={chart_df.columns[0]: "Date"})
        chart = {
            "dates": chart_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "close": chart_df["Close"].round(2).tolist(),
            "sma50": chart_df["SMA50"].round(2).fillna(0).tolist(),
            "sma200": chart_df["SMA200"].round(2).fillna(0).tolist(),
        }

        return {
            "insights": insights,
            "intraday_last_price": float(live["Close"].iloc[-1]),
            "support_resistance": sr,
            "signals": [asdict(s) for s in signals],
            "backtest": backtest,
            "chart": chart,
        }
