# Algorithmic Trading Bot (Alpaca)

A modular, risk-controlled algorithmic trading skeleton for Alpaca, designed
for **paper-first** development. The included strategy (SMA crossover) is an
example only — the value is in the surrounding architecture: risk gates,
bracket orders, kill-switch, position sizing, logging, and backtesting.

> **WARNING** — Trading real money carries the risk of significant loss.
> A profitable backtest does not imply future profits. Validate this system
> in paper mode for weeks, review every trade, and only then consider live
> trading with capital you can afford to lose.

---

## Quickstart

```bash
# 1. Install dependencies (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure credentials (PAPER keys to start)
cp .env.example .env
# Edit .env: paste your paper API key and secret from
#   https://app.alpaca.markets/paper/dashboard/overview

# 3. Run a backtest
python run_backtest.py

# 4. Run paper trading
python main.py

# 5. Halt the bot at any time without restarting
touch logs/STOP
```

## File map

| Path | Purpose |
|------|---------|
| `config/settings.py` | All tunable parameters; loads creds from env |
| `src/broker/alpaca_client.py` | Single gateway to Alpaca's API |
| `src/data/market_data.py` | Bar fetching with liquidity filter |
| `src/strategy/base.py` | Strategy abstract base class |
| `src/strategy/sma_crossover.py` | Example SMA crossover strategy |
| `src/risk/risk_manager.py` | Pre-trade gates and position sizing |
| `src/execution/order_executor.py` | Bracket-order submission |
| `src/backtest/backtester.py` | Event-driven backtester |
| `src/monitoring/logger.py` | Loguru setup |
| `main.py` | Live/paper trading orchestrator |
| `run_backtest.py` | Backtest entry point |
| `tests/` | Pytest tests (start with risk manager) |

## Risk parameters (defaults — review before changing)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `max_position_pct` | 10% | Cap of equity per single position |
| `max_daily_loss_pct` | 2% | Halts the bot for the day if breached |
| `stop_loss_pct` | 2% | Per-trade stop loss |
| `take_profit_pct` | 4% | Per-trade take profit (1:2 R:R) |
| `risk_per_trade_pct` | 0.5% | Equity at risk per trade |
| `max_open_positions` | 5 | Concurrent positions cap |
| `cooldown_after_consecutive_losses` | 3 | Pauses for `cooldown_minutes` |
| `min_avg_daily_volume` | 1M shares | Liquidity floor |
| `use_leverage` | False | Must explicitly enable |

## Pre-flight checklist before going LIVE

1. ✅ Run the backtester on at least 1 year of data; review trades and drawdowns.
2. ✅ Run paper trading for **at least 4 weeks** in real market conditions.
3. ✅ Test the kill-switch (`touch logs/STOP`) and confirm it halts the bot.
4. ✅ Test the daily-loss halt by lowering `max_daily_loss_pct` and forcing a loss in paper.
5. ✅ Confirm bracket orders are placed (visible in Alpaca dashboard).
6. ✅ Set up log monitoring and alerts (e.g. notifications on any `ERROR` or `WARNING`).
7. ✅ Start with the smallest equity you'd be okay losing entirely.
8. ✅ Sit at the computer for the first full session. Watch every trade.

## Common mistakes this scaffold helps avoid

- Hardcoded API keys in code (use `.env`).
- Acting on the currently-forming bar (look-ahead bias).
- Forgetting bracket orders, leaving open trades unprotected if the bot crashes.
- Risk parameters scattered across files — here they are all in one settings module.
- Single point of failure in the strategy layer — every order goes through risk gates.
- Implicit leverage on margin accounts — we size against equity by default.
- Trading illiquid names — liquidity filter rejects below `min_avg_daily_volume`.
- No way to stop the bot mid-session — kill-switch file solves this.

## Disclaimer

This code is provided for educational purposes only. Nothing here is
investment advice. Markets carry risk. Past performance, simulated or
otherwise, is not indicative of future results. You are solely
responsible for any trades placed with this software.
