import os
from decimal import Decimal
import pandas as pd
import yfinance as yf
from typing import Tuple

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
from nautilus_trader.model.data import BarSpecification, BarType, BarAggregation, Bar
from nautilus_trader.core.rust.model import PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.model.currencies import USD

def download_uvxy(start="2024-01-01", end="2025-01-01") -> pd.DataFrame:
    df = yf.download("UVXY", start=start, end=end, group_by="ticker", auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for UVXY between {start}–{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1].lower() for col in df.columns]
    else:
        df.columns = df.columns.str.lower()
    needed = ['open','high','low','close','volume']
    missing = set(needed) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = df[needed].dropna()
    df["ts_event"] = pd.to_datetime(df.index).tz_localize("UTC").astype("int64")
    df["ts_init"]  = df["ts_event"]
    return df

def df_to_navigator_bars(df: pd.DataFrame, inst_id: InstrumentId) -> Tuple[list, BarType]:
    spec     = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
    bar_type = BarType(inst_id, spec)
    bars = []
    for _, row in df.iterrows():
        o = f"{row['open']:.9f}"; h = f"{row['high']:.9f}"
        l = f"{row['low']:.9f}";  c = f"{row['close']:.9f}"
        bars.append(Bar(
            bar_type=bar_type,
            open= Price.from_str(o),
            high= Price.from_str(h),
            low=  Price.from_str(l),
            close=Price.from_str(c),
            volume=Quantity.from_int(int(row['volume'])),
            ts_event=int(row["ts_event"]),
            ts_init=int(row["ts_init"]),
            is_revision=False,
        ))
    return bars, bar_type

def main():
    # 1. Download and prep data
    print("Downloading UVXY data…")
    df = download_uvxy()

    # 2. Define instrument identity and full Equity instrument
    inst_id = InstrumentId(symbol=Symbol("UVXY"), venue=Venue("YF"))
    equity = Equity(
        instrument_id=inst_id,
        raw_symbol=Symbol("UVXY"),
        currency=USD,
        price_precision=4,             # 4 decimal places
        price_increment=Price.from_str("0.0001"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )

    # 3. Convert bars
    print("Converting data to bars…")
    bars, bar_type = df_to_navigator_bars(df, inst_id)

    # 4. Initialize engine
    print("Initializing BacktestEngine…")
    cfg = BacktestEngineConfig(trader_id="BACKTESTER-UVXY")
    engine = BacktestEngine(config=cfg)

    # 5. Register venue & instrument & data
    print("Registering venue & instrument…")
    engine.add_venue(
        venue=Venue("YF"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(Decimal("1000000"), USD)],
    )
    engine.add_instrument(equity)
    engine.add_data(bars)

    # 6. Configure and add EMA-cross strategy
    print("Adding EMA-cross strategy…")
    st_cfg = EMACrossConfig(
        instrument_id=inst_id,
        bar_type=bar_type,
        fast_ema_period=10,
        slow_ema_period=20,
        trade_size=Quantity.from_int(1000),
    )
    engine.add_strategy(EMACross(st_cfg))

    # 7. Run
    print("Running backtest…")
    engine.run()

    print("Backtest complete.")
    report = engine.trader.generate_account_report(Venue("YF"))
    print("Account Report:\n", report)

    engine.dispose()

if __name__ == "__main__":
    main()
