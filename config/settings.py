"""Centralized configuration for a safer, research-backed trading bot baseline.

Why this version?
    - Fixes the sizing math so risk-per-trade and position caps are consistent.
    - Prices in expected slippage / regulatory-fee friction for sizing.
    - Separates gross exposure from actual risk semantics.
    - Adds mark-to-market drawdown halts, correlation caps, and EOD flattening.
    - Keeps paper trading as the default and requires an explicit second flag
      before live trading can be armed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = PROJECT_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if not value:
        return default.copy()
    return [item.strip().upper() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")

    # ------------------------------------------------------------------
    # Deployment mode
    # Two-step arming for live trading:
    #   1) PAPER_TRADING=false
    #   2) ALLOW_LIVE_TRADING=true
    # ------------------------------------------------------------------
    paper_trading: bool = _env_bool("PAPER_TRADING", True)
    allow_live_trading: bool = _env_bool("ALLOW_LIVE_TRADING", False)

    # Match this to the exact feed used in backtesting.
    # For many Paper / Basic setups, real-time equities data is IEX-only.
    market_data_feed: str = os.getenv("ALPACA_MARKET_DATA_FEED", "iex").strip().lower()

    # ------------------------------------------------------------------
    # Trading universe
    # By default prefer liquid ETFs instead of concentrated single-name tech
    # exposure. Override with SYMBOLS=SPY,AAPL,MSFT,... if you want custom names.
    # ------------------------------------------------------------------
    symbols: List[str] = field(
        default_factory=lambda: _env_list("SYMBOLS", ["QQQ", "IWM", "TLT", "GLD"])
    )
    min_price: float = _env_float("MIN_PRICE", 5.0)
    min_avg_daily_dollar_volume: float = _env_float(
        "MIN_AVG_DAILY_DOLLAR_VOLUME",
        50_000_000.0,
    )
    max_bid_ask_spread_bps: float = _env_float("MAX_BID_ASK_SPREAD_BPS", 8.0)

    # ------------------------------------------------------------------
    # Risk model
    # Core formula:
    #   position_pct ~= risk_per_trade_pct / worst_case_stop_pct
    #
    # where:
    #   effective_stop_pct = stop_loss_pct + round_trip_friction_pct
    #   worst_case_stop_pct = effective_stop_pct (intraday only)
    #                      or effective_stop_pct * gap_risk_multiplier_overnight
    #
    # These defaults imply a risk-based position size of roughly 9.35% of equity,
    # which fits within the 10% max_position_pct cap.
    # ------------------------------------------------------------------
    max_position_pct: float = _env_float("MAX_POSITION_PCT", 0.10)
    max_gross_exposure_pct: float = _env_float("MAX_GROSS_EXPOSURE_PCT", 0.40)
    risk_per_trade_pct: float = _env_float("RISK_PER_TRADE_PCT", 0.0015)

    stop_loss_pct: float = _env_float("STOP_LOSS_PCT", 0.015)
    take_profit_pct: float = _env_float("TAKE_PROFIT_PCT", 0.030)

    expected_slippage_bps_per_side: float = _env_float(
        "EXPECTED_SLIPPAGE_BPS_PER_SIDE",
        5.0,
    )
    expected_fees_bps_per_side: float = _env_float(
        "EXPECTED_FEES_BPS_PER_SIDE",
        0.2,
    )

    # If you intentionally carry overnight, size as though losses can exceed
    # the intraday stop distance because gaps can jump over stops.
    gap_risk_multiplier_overnight: float = _env_float(
        "GAP_RISK_MULTIPLIER_OVERNIGHT",
        2.0,
    )

    # Drawdown halts should include OPEN + REALIZED P&L, not just closed trades.
    max_daily_loss_pct: float = _env_float("MAX_DAILY_LOSS_PCT", 0.02)
    max_weekly_loss_pct: float = _env_float("MAX_WEEKLY_LOSS_PCT", 0.05)
    include_open_pnl_in_loss_limits: bool = _env_bool(
        "INCLUDE_OPEN_PNL_IN_LOSS_LIMITS",
        True,
    )

    # Concentration controls
    max_open_positions: int = _env_int("MAX_OPEN_POSITIONS", 4)
    max_high_correlation_positions: int = _env_int("MAX_HIGH_CORRELATION_POSITIONS", 2)
    max_pairwise_correlation: float = _env_float("MAX_PAIRWISE_CORRELATION", 0.80)
    correlation_lookback_bars: int = _env_int("CORRELATION_LOOKBACK_BARS", 60)

    # ------------------------------------------------------------------
    # Execution controls
    # ------------------------------------------------------------------
    use_bracket_orders: bool = _env_bool("USE_BRACKET_ORDERS", True)

    # "stop" prioritizes execution certainty after trigger.
    # "stop_limit" prioritizes price control but may not fill.
    stop_order_type: str = os.getenv("STOP_ORDER_TYPE", "stop").strip().lower()

    max_orders_per_minute: int = _env_int("MAX_ORDERS_PER_MINUTE", 20)
    cooldown_after_consecutive_losses: int = _env_int(
        "COOLDOWN_AFTER_CONSECUTIVE_LOSSES",
        3,
    )
    cooldown_minutes: int = _env_int("COOLDOWN_MINUTES", 60)

    flatten_at_eod: bool = _env_bool("FLATTEN_AT_EOD", True)
    flatten_before_close_minutes: int = _env_int("FLATTEN_BEFORE_CLOSE_MINUTES", 15)
    allow_extended_hours: bool = _env_bool("ALLOW_EXTENDED_HOURS", False)

    # ------------------------------------------------------------------
    # Strategy defaults
    # The literature review favored momentum/trend families over retail-style
    # oscillator/crossover rules after costs, so the bot now defaults to the
    # evidence-backed momentum path while keeping older strategies available for
    # comparison and rollback.
    # ------------------------------------------------------------------
    strategy_mode: str = os.getenv("STRATEGY_MODE", "evidence_momentum").strip().lower()

    short_sma_window: int = _env_int("SHORT_SMA_WINDOW", 30)
    long_sma_window: int = _env_int("LONG_SMA_WINDOW", 100)
    regime_filter_window: int = _env_int("REGIME_FILTER_WINDOW", 200)
    pullback_lookback_bars: int = _env_int("PULLBACK_LOOKBACK_BARS", 3)
    trend_slope_lookback_bars: int = _env_int("TREND_SLOPE_LOOKBACK_BARS", 5)
    atr_window: int = _env_int("ATR_WINDOW", 14)
    min_atr_pct: float = _env_float("MIN_ATR_PCT", 0.002)
    max_atr_pct: float = _env_float("MAX_ATR_PCT", 0.030)
    pullback_reclaim_tolerance_pct: float = _env_float(
        "PULLBACK_RECLAIM_TOLERANCE_PCT",
        0.001,
    )
    relative_strength_lookback_bars: int = _env_int(
        "RELATIVE_STRENGTH_LOOKBACK_BARS",
        20,
    )
    max_relative_strength_rank: int = _env_int(
        "MAX_RELATIVE_STRENGTH_RANK",
        3,
    )
    min_relative_strength_return: float = _env_float(
        "MIN_RELATIVE_STRENGTH_RETURN",
        0.0,
    )
    bar_timeframe_minutes: int = _env_int("BAR_TIMEFRAME_MINUTES", 60)

    mean_reversion_market_proxy: str = os.getenv(
        "MEAN_REVERSION_MARKET_PROXY",
        "SPY",
    ).strip().upper()
    mr_bollinger_window: int = _env_int("MR_BOLLINGER_WINDOW", 20)
    mr_bollinger_std_mult: float = _env_float("MR_BOLLINGER_STD_MULT", 2.0)
    mr_keltner_window: int = _env_int("MR_KELTNER_WINDOW", 20)
    mr_keltner_atr_mult: float = _env_float("MR_KELTNER_ATR_MULT", 1.5)
    mr_rsi_length: int = _env_int("MR_RSI_LENGTH", 7)
    mr_rsi_entry_threshold: float = _env_float("MR_RSI_ENTRY_THRESHOLD", 30.0)
    mr_rsi_exit_threshold: float = _env_float("MR_RSI_EXIT_THRESHOLD", 58.0)
    mr_regime_window: int = _env_int("MR_REGIME_WINDOW", 200)
    mr_regime_buffer_pct: float = _env_float("MR_REGIME_BUFFER_PCT", 0.02)
    mr_regime_slope_lookback: int = _env_int("MR_REGIME_SLOPE_LOOKBACK", 10)
    mr_regime_max_down_slope_pct: float = _env_float(
        "MR_REGIME_MAX_DOWN_SLOPE_PCT",
        0.01,
    )
    mr_require_asset_regime: bool = _env_bool("MR_REQUIRE_ASSET_REGIME", False)
    mr_asset_regime_buffer_pct: float = _env_float(
        "MR_ASSET_REGIME_BUFFER_PCT",
        0.05,
    )
    mr_atr_length: int = _env_int("MR_ATR_LENGTH", 14)
    mr_atr_stop_mult: float = _env_float("MR_ATR_STOP_MULT", 1.8)
    mr_atr_trailing_stop_mult: float = _env_float(
        "MR_ATR_TRAILING_STOP_MULT",
        2.2,
    )
    mr_atr_target_mult: float = _env_float("MR_ATR_TARGET_MULT", 1.1)
    mr_mean_exit_atr_buffer: float = _env_float("MR_MEAN_EXIT_ATR_BUFFER", 0.15)
    mr_max_holding_bars: int = _env_int("MR_MAX_HOLDING_BARS", 18)
    mr_require_reversal_bar: bool = _env_bool("MR_REQUIRE_REVERSAL_BAR", True)
    mr_entry_zscore_threshold: float = _env_float("MR_ENTRY_ZSCORE_THRESHOLD", 0.9)
    mr_entry_reclaim_atr_buffer: float = _env_float(
        "MR_ENTRY_RECLAIM_ATR_BUFFER",
        0.10,
    )
    mr_allow_short: bool = _env_bool("MR_ALLOW_SHORT", False)
    mr_circuit_breaker_drawdown_pct: float = _env_float(
        "MR_CIRCUIT_BREAKER_DRAWDOWN_PCT",
        0.25,
    )

    evidence_market_proxy: str = os.getenv(
        "EVIDENCE_MARKET_PROXY",
        "SPY",
    ).strip().upper()
    em_short_lookback_sessions: int = _env_int("EM_SHORT_LOOKBACK_SESSIONS", 21)
    em_medium_lookback_sessions: int = _env_int("EM_MEDIUM_LOOKBACK_SESSIONS", 63)
    em_long_lookback_sessions: int = _env_int("EM_LONG_LOOKBACK_SESSIONS", 126)
    em_skip_recent_sessions: int = _env_int("EM_SKIP_RECENT_SESSIONS", 5)
    em_long_sma_sessions: int = _env_int("EM_LONG_SMA_SESSIONS", 200)
    em_high_lookback_sessions: int = _env_int("EM_HIGH_LOOKBACK_SESSIONS", 252)
    em_vol_lookback_sessions: int = _env_int("EM_VOL_LOOKBACK_SESSIONS", 21)
    em_min_high_ratio: float = _env_float("EM_MIN_HIGH_RATIO", 0.85)
    em_min_momentum_score: float = _env_float("EM_MIN_MOMENTUM_SCORE", 0.0)
    em_require_market_regime: bool = _env_bool("EM_REQUIRE_MARKET_REGIME", True)
    em_market_regime_buffer_pct: float = _env_float("EM_MARKET_REGIME_BUFFER_PCT", 0.02)
    em_rebalance_interval_sessions: int = _env_int("EM_REBALANCE_INTERVAL_SESSIONS", 5)
    em_top_n: int = _env_int("EM_TOP_N", 2)
    em_atr_length: int = _env_int("EM_ATR_LENGTH", 14)
    em_atr_stop_mult: float = _env_float("EM_ATR_STOP_MULT", 2.0)
    em_atr_trailing_stop_mult: float = _env_float("EM_ATR_TRAILING_STOP_MULT", 3.0)
    em_circuit_breaker_drawdown_pct: float = _env_float(
        "EM_CIRCUIT_BREAKER_DRAWDOWN_PCT",
        0.25,
    )

    # ------------------------------------------------------------------
    # Operational controls
    # ------------------------------------------------------------------
    use_websocket_market_data: bool = _env_bool("USE_WEBSOCKET_MARKET_DATA", True)
    poll_interval_seconds: int = _env_int("POLL_INTERVAL_SECONDS", 15)
    heartbeat_timeout_seconds: int = _env_int("HEARTBEAT_TIMEOUT_SECONDS", 180)
    use_leverage: bool = _env_bool("USE_LEVERAGE", False)
    runtime_state_path: Path = STATE_DIR / "runtime_state.json"

    # Touch this file to halt the bot at the next loop iteration:
    #   touch logs/STOP
    kill_switch_path: Path = LOGS_DIR / "STOP"

    @property
    def base_url(self) -> str:
        return (
            "https://paper-api.alpaca.markets"
            if self.paper_trading
            else "https://api.alpaca.markets"
        )

    @property
    def max_portfolio_risk_pct(self) -> float:
        """
        Backward-compatible alias for older code that used the misleading name
        max_portfolio_risk_pct for what is really a gross exposure cap.
        """
        return self.max_gross_exposure_pct

    @property
    def round_trip_friction_pct(self) -> float:
        """
        Estimated total friction for one entry and one exit, expressed as
        a decimal. Example: 10 bps = 0.0010
        """
        return 2.0 * (
            self.expected_slippage_bps_per_side + self.expected_fees_bps_per_side
        ) / 10_000.0

    @property
    def effective_stop_pct(self) -> float:
        """
        Stop distance used for sizing. Intentionally larger than the raw trigger.
        """
        return self.stop_loss_pct + self.round_trip_friction_pct

    @property
    def worst_case_stop_pct(self) -> float:
        """
        If you do NOT flatten at end of day, size as though gaps can make the
        realized loss materially larger than the intraday stop.
        """
        if self.flatten_at_eod:
            return self.effective_stop_pct
        return self.effective_stop_pct * self.gap_risk_multiplier_overnight

    @property
    def reward_to_risk(self) -> float:
        return self.take_profit_pct / self.stop_loss_pct

    @property
    def risk_implied_position_pct(self) -> float:
        return self.risk_per_trade_pct / self.worst_case_stop_pct

    def target_position_pct(self, stop_distance_pct: float | None = None) -> float:
        effective_stop = (
            self.worst_case_stop_pct
            if stop_distance_pct is None
            else stop_distance_pct
        )
        if effective_stop <= 0:
            raise ValueError("stop_distance_pct must be > 0")
        return min(self.risk_per_trade_pct / effective_stop, self.max_position_pct)

    def target_position_notional(
        self,
        equity: float,
        stop_distance_pct: float | None = None,
    ) -> float:
        if equity <= 0:
            raise ValueError("equity must be > 0")
        return equity * self.target_position_pct(stop_distance_pct)

    def shares_for_entry(
        self,
        equity: float,
        entry_price: float,
        stop_distance_pct: float | None = None,
    ) -> int:
        """
        Integer-share sizing helper.
        If you later add fractional-share support, this method can be relaxed.
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        notional = self.target_position_notional(equity, stop_distance_pct)
        return max(int(notional / entry_price), 0)

    def validate(self) -> None:
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your environment or .env file."
            )

        if not self.paper_trading and not self.allow_live_trading:
            raise RuntimeError(
                "Live trading is blocked. Keep PAPER_TRADING=true by default, or "
                "explicitly set ALLOW_LIVE_TRADING=true after paper validation."
            )

        if self.market_data_feed not in {"iex", "sip"}:
            raise ValueError("ALPACA_MARKET_DATA_FEED must be 'iex' or 'sip'")
        if self.strategy_mode not in {"evidence_momentum", "mean_reversion"}:
            raise ValueError(
                "STRATEGY_MODE must be 'evidence_momentum' or 'mean_reversion'"
            )

        if not self.symbols:
            raise ValueError("At least one symbol is required")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Duplicate symbols are not allowed")

        if self.stop_order_type not in {"stop", "stop_limit"}:
            raise ValueError("STOP_ORDER_TYPE must be 'stop' or 'stop_limit'")

        if self.short_sma_window >= self.long_sma_window:
            raise ValueError("short_sma_window must be < long_sma_window")
        if self.long_sma_window >= self.regime_filter_window:
            raise ValueError("long_sma_window must be < regime_filter_window")
        if not (2 <= self.pullback_lookback_bars <= 100):
            raise ValueError("pullback_lookback_bars must be in [2, 100]")
        if not (1 <= self.trend_slope_lookback_bars <= 100):
            raise ValueError("trend_slope_lookback_bars must be in [1, 100]")
        if not (2 <= self.atr_window <= 100):
            raise ValueError("atr_window must be in [2, 100]")
        if not (0 <= self.min_atr_pct < self.max_atr_pct < 0.20):
            raise ValueError("Require 0 <= min_atr_pct < max_atr_pct < 0.20")
        if not (0 <= self.pullback_reclaim_tolerance_pct <= 0.02):
            raise ValueError("pullback_reclaim_tolerance_pct must be in [0, 0.02]")
        if self.relative_strength_lookback_bars < 1:
            raise ValueError("relative_strength_lookback_bars must be >= 1")
        if self.max_relative_strength_rank < 0:
            raise ValueError("max_relative_strength_rank must be >= 0")
        if not (-1.0 <= self.min_relative_strength_return <= 1.0):
            raise ValueError("min_relative_strength_return must be in [-1.0, 1.0]")
        if self.bar_timeframe_minutes not in {1, 5, 15, 30, 60}:
            raise ValueError("bar_timeframe_minutes must be one of {1, 5, 15, 30, 60}")
        if not self.mean_reversion_market_proxy:
            raise ValueError("mean_reversion_market_proxy is required")
        if self.mr_bollinger_window < 2:
            raise ValueError("mr_bollinger_window must be >= 2")
        if not (0.5 <= self.mr_bollinger_std_mult <= 5.0):
            raise ValueError("mr_bollinger_std_mult must be in [0.5, 5.0]")
        if self.mr_keltner_window < 2:
            raise ValueError("mr_keltner_window must be >= 2")
        if not (0.5 <= self.mr_keltner_atr_mult <= 5.0):
            raise ValueError("mr_keltner_atr_mult must be in [0.5, 5.0]")
        if self.mr_rsi_length < 2:
            raise ValueError("mr_rsi_length must be >= 2")
        if not (0.0 < self.mr_rsi_entry_threshold < self.mr_rsi_exit_threshold < 100.0):
            raise ValueError(
                "Require 0 < mr_rsi_entry_threshold < mr_rsi_exit_threshold < 100"
            )
        if self.mr_regime_window < max(self.mr_bollinger_window, self.mr_keltner_window):
            raise ValueError(
                "mr_regime_window must be >= the Bollinger and Keltner windows"
            )
        if not (0.0 <= self.mr_regime_buffer_pct < 0.20):
            raise ValueError("mr_regime_buffer_pct must be in [0.0, 0.20)")
        if self.mr_regime_slope_lookback < 1:
            raise ValueError("mr_regime_slope_lookback must be >= 1")
        if not (0.0 <= self.mr_regime_max_down_slope_pct < 0.20):
            raise ValueError("mr_regime_max_down_slope_pct must be in [0.0, 0.20)")
        if not (0.0 <= self.mr_asset_regime_buffer_pct < 0.25):
            raise ValueError("mr_asset_regime_buffer_pct must be in [0.0, 0.25)")
        if self.mr_atr_length < 2:
            raise ValueError("mr_atr_length must be >= 2")
        if not (0.1 <= self.mr_atr_stop_mult <= 10.0):
            raise ValueError("mr_atr_stop_mult must be in [0.1, 10.0]")
        if not (0.1 <= self.mr_atr_trailing_stop_mult <= 10.0):
            raise ValueError("mr_atr_trailing_stop_mult must be in [0.1, 10.0]")
        if not (0.1 <= self.mr_atr_target_mult <= 10.0):
            raise ValueError("mr_atr_target_mult must be in [0.1, 10.0]")
        if not (0.0 <= self.mr_mean_exit_atr_buffer <= 5.0):
            raise ValueError("mr_mean_exit_atr_buffer must be in [0.0, 5.0]")
        if self.mr_max_holding_bars < 1:
            raise ValueError("mr_max_holding_bars must be >= 1")
        if not (0.0 <= self.mr_entry_zscore_threshold <= 5.0):
            raise ValueError("mr_entry_zscore_threshold must be in [0.0, 5.0]")
        if not (0.0 <= self.mr_entry_reclaim_atr_buffer <= 5.0):
            raise ValueError("mr_entry_reclaim_atr_buffer must be in [0.0, 5.0]")
        if not (0.01 <= self.mr_circuit_breaker_drawdown_pct <= 0.50):
            raise ValueError("mr_circuit_breaker_drawdown_pct must be in [0.01, 0.50]")
        if not self.evidence_market_proxy:
            raise ValueError("evidence_market_proxy is required")
        if self.em_short_lookback_sessions < 1:
            raise ValueError("em_short_lookback_sessions must be >= 1")
        if self.em_medium_lookback_sessions <= self.em_short_lookback_sessions:
            raise ValueError(
                "em_medium_lookback_sessions must be > em_short_lookback_sessions"
            )
        if self.em_long_lookback_sessions <= self.em_medium_lookback_sessions:
            raise ValueError(
                "em_long_lookback_sessions must be > em_medium_lookback_sessions"
            )
        if self.em_skip_recent_sessions < 0:
            raise ValueError("em_skip_recent_sessions must be >= 0")
        if self.em_long_sma_sessions < self.em_long_lookback_sessions:
            raise ValueError(
                "em_long_sma_sessions should be >= em_long_lookback_sessions"
            )
        if self.em_high_lookback_sessions < self.em_long_lookback_sessions:
            raise ValueError(
                "em_high_lookback_sessions should be >= em_long_lookback_sessions"
            )
        if self.em_vol_lookback_sessions < 2:
            raise ValueError("em_vol_lookback_sessions must be >= 2")
        if not (0.5 <= self.em_min_high_ratio <= 1.05):
            raise ValueError("em_min_high_ratio must be in [0.5, 1.05]")
        if not (-5.0 <= self.em_min_momentum_score <= 5.0):
            raise ValueError("em_min_momentum_score must be in [-5.0, 5.0]")
        if not (0.0 <= self.em_market_regime_buffer_pct < 0.20):
            raise ValueError("em_market_regime_buffer_pct must be in [0.0, 0.20)")
        if self.em_rebalance_interval_sessions < 1:
            raise ValueError("em_rebalance_interval_sessions must be >= 1")
        if self.em_top_n < 1:
            raise ValueError("em_top_n must be >= 1")
        if self.em_atr_length < 2:
            raise ValueError("em_atr_length must be >= 2")
        if not (0.1 <= self.em_atr_stop_mult <= 10.0):
            raise ValueError("em_atr_stop_mult must be in [0.1, 10.0]")
        if not (0.1 <= self.em_atr_trailing_stop_mult <= 10.0):
            raise ValueError("em_atr_trailing_stop_mult must be in [0.1, 10.0]")
        if not (0.01 <= self.em_circuit_breaker_drawdown_pct <= 0.50):
            raise ValueError("em_circuit_breaker_drawdown_pct must be in [0.01, 0.50]")

        if not (0 < self.min_price):
            raise ValueError("min_price must be > 0")
        if not (0 < self.min_avg_daily_dollar_volume):
            raise ValueError("min_avg_daily_dollar_volume must be > 0")
        if not (0 < self.max_bid_ask_spread_bps):
            raise ValueError("max_bid_ask_spread_bps must be > 0")

        if not (0 < self.max_position_pct <= 0.25):
            raise ValueError("max_position_pct must be in (0, 0.25]")
        if not (0 < self.max_gross_exposure_pct <= 1.0):
            raise ValueError("max_gross_exposure_pct must be in (0, 1.0]")
        if self.max_gross_exposure_pct > self.max_open_positions * self.max_position_pct:
            raise ValueError(
                "max_gross_exposure_pct cannot exceed "
                "max_open_positions * max_position_pct"
            )

        if not (0 < self.risk_per_trade_pct < self.max_daily_loss_pct):
            raise ValueError(
                "risk_per_trade_pct must be > 0 and smaller than max_daily_loss_pct"
            )
        if not (0 < self.stop_loss_pct < 0.10):
            raise ValueError("stop_loss_pct must be in (0, 0.10)")
        if not (self.take_profit_pct > self.stop_loss_pct):
            raise ValueError("take_profit_pct should be > stop_loss_pct")
        if not (0 < self.max_daily_loss_pct < self.max_weekly_loss_pct < 0.20):
            raise ValueError(
                "Require 0 < max_daily_loss_pct < max_weekly_loss_pct < 0.20"
            )

        if not (0 <= self.expected_slippage_bps_per_side <= 50):
            raise ValueError("expected_slippage_bps_per_side looks unreasonable")
        if not (0 <= self.expected_fees_bps_per_side <= 10):
            raise ValueError("expected_fees_bps_per_side looks unreasonable")
        if not (1.0 <= self.gap_risk_multiplier_overnight <= 5.0):
            raise ValueError("gap_risk_multiplier_overnight must be in [1.0, 5.0]")

        if not (1 <= self.max_open_positions <= 50):
            raise ValueError("max_open_positions must be in [1, 50]")
        if not (1 <= self.max_high_correlation_positions <= self.max_open_positions):
            raise ValueError(
                "max_high_correlation_positions must be in "
                "[1, max_open_positions]"
            )
        if not (0.0 < self.max_pairwise_correlation <= 1.0):
            raise ValueError("max_pairwise_correlation must be in (0, 1]")
        if not (10 <= self.correlation_lookback_bars <= 500):
            raise ValueError("correlation_lookback_bars must be in [10, 500]")

        if not (1 <= self.max_orders_per_minute <= 200):
            raise ValueError("max_orders_per_minute must be in [1, 200]")
        if not (1 <= self.cooldown_after_consecutive_losses <= 20):
            raise ValueError("cooldown_after_consecutive_losses must be in [1, 20]")
        if not (1 <= self.cooldown_minutes <= 24 * 60):
            raise ValueError("cooldown_minutes must be in [1, 1440]")

        if self.flatten_at_eod and self.flatten_before_close_minutes < 1:
            raise ValueError(
                "flatten_before_close_minutes must be >= 1 when flatten_at_eod is true"
            )

        if not (1 <= self.poll_interval_seconds <= self.bar_timeframe_minutes * 60):
            raise ValueError(
                "poll_interval_seconds must be >= 1 and <= bar_timeframe_minutes * 60"
            )
        if self.heartbeat_timeout_seconds <= self.poll_interval_seconds:
            raise ValueError(
                "heartbeat_timeout_seconds must be > poll_interval_seconds"
            )

        if self.risk_implied_position_pct > self.max_position_pct + 1e-12:
            raise ValueError(
                "Sizing is inconsistent: risk_per_trade_pct / worst_case_stop_pct "
                "exceeds max_position_pct. Lower risk_per_trade_pct, widen stop "
                "assumptions, or raise the cap only with evidence."
            )


# Shared instance, imported elsewhere as:
#   from config.settings import settings
settings = Settings()
