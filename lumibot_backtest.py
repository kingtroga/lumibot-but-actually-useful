from lumibot.strategies import Strategy
from lumibot.backtesting import PandasDataBacktesting
from lumibot.entities import Asset, Data
import pandas as pd
import numpy as np
import quantstats as qs

CASH_USAGE = 0.95

class SignalBasedStrategy(Strategy):
    """Executes pre-computed trading signals using Lumibot."""

    _trades_storage = []
    _max_exposure_storage = 0

    parameters = {
        "symbol": "AAPL",
        "signals_df": None,
        "init_cash": 100000,
        "leverage": 1,
        "fees": 0,
    }

    def initialize(self):
        SignalBasedStrategy._trades_storage = []
        SignalBasedStrategy._max_exposure_storage = 0

        self.sleeptime = "1M"
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.position_type = None
        self.current_trade = None

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        current_dt = self.get_datetime()
        signals = self.parameters["signals_df"]

        try:
            if current_dt not in signals.index:
                return

            s = signals.loc[current_dt]
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
            position = self.get_position(asset)
            price = self.get_last_price(asset)
            if price is None or price == 0:
                return

            if s.get('long_entries') and position is None:
                self._enter_long(asset, current_dt, price, s)
            elif s.get('long_exits') and position and self.position_type == 'long':
                self._close_trade(current_dt, price, position, "Signal Exit")
            elif s.get('short_entries') and position is None:
                self._enter_short(asset, current_dt, price, s)
            elif s.get('short_exits') and position and self.position_type == 'short':
                self._close_trade(current_dt, price, position, "Signal Exit")
            elif position and self.entry_price is not None:
                if self.position_type == 'long':
                    if price <= self.stop_loss_price:
                        self._close_trade(current_dt, price, position, "Stop Loss")
                    elif price >= self.take_profit_price:
                        self._close_trade(current_dt, price, position, "Take Profit")
                else:
                    if price >= self.stop_loss_price:
                        self._close_trade(current_dt, price, position, "Stop Loss")
                    elif price <= self.take_profit_price:
                        self._close_trade(current_dt, price, position, "Take Profit")

        except Exception as e:
            logger.error(f'Error at {current_dt} processing signals: {e}', exc_info=True)

    def _enter_long(self, asset, dt, price, s):
        leverage = self.parameters.get("leverage", 1)
        qty = int((self.cash * CASH_USAGE * leverage) / price)
        if qty <= 0:
            return
        try:
            order = self.create_order(asset, qty, "buy")
            self.submit_order(order)

            exposure = (qty * price) / self.parameters["init_cash"] * 100
            SignalBasedStrategy._max_exposure_storage = max(
                SignalBasedStrategy._max_exposure_storage, exposure
            )

            self.entry_price = price
            self.stop_loss_price = price * (1 - s.get('long_sl', 0.02))
            self.take_profit_price = price * (1 + s.get('long_tp', 0.05))
            self.position_type = 'long'
            self.current_trade = {
                'entry_date': dt, 'entry_price': price, 'quantity': qty,
                'type': 'long', 'entry_fees': self.parameters.get('fees', 0)
            }
        except Exception as e:
            logger.error(f'Failed to enter long at {dt}: {e}', exc_info=True)

    def _enter_short(self, asset, dt, price, s):
        leverage = self.parameters.get("leverage", 1)
        qty = int((self.cash * 0.95 * leverage) / price)
        if qty <= 0:
            return
        try:
            order = self.create_order(asset, qty, "sell")
            self.submit_order(order)

            exposure = (qty * price) / self.parameters["init_cash"] * 100
            SignalBasedStrategy._max_exposure_storage = max(
                SignalBasedStrategy._max_exposure_storage, exposure
            )

            self.entry_price = price
            self.stop_loss_price = price * (1 + s.get('short_sl', 0.02))
            self.take_profit_price = price * (1 - s.get('short_tp', 0.05))
            self.position_type = 'short'
            self.current_trade = {
                'entry_date': dt, 'entry_price': price, 'quantity': qty,
                'type': 'short', 'entry_fees': self.parameters.get('fees', 0)
            }
        except Exception as e:
            logger.error(f'Failed to enter short at {dt}: {e}', exc_info=True)

    def _close_trade(self, dt, price, position, reason):
        if self.position_type == 'long':
            pnl = (price - self.entry_price) * position.quantity
            self.sell_all()
        else:
            pnl = (self.entry_price - price) * position.quantity
            asset = Asset(symbol=self.parameters["symbol"], asset_type=Asset.AssetType.STOCK)
            self.submit_order(self.create_order(asset, position.quantity, "buy"))

        if self.current_trade:
            exit_fees = self.parameters.get('fees', 0)
            total_fees = self.current_trade['entry_fees'] + exit_fees
            net_pnl = pnl - total_fees
            self.current_trade.update({
                'exit_date': dt, 'exit_price': price, 'exit_fees': exit_fees,
                'pnl': net_pnl,
                'return': (net_pnl / (self.entry_price * self.current_trade['quantity'])) * 100,
                'exit_reason': reason,
            })
            SignalBasedStrategy._trades_storage.append(self.current_trade.copy())
            self.current_trade = None

        self._reset_position()

    def _reset_position(self):
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.position_type = None


def run_backtest(signals_df, symbol="AAPL", start_date=None, end_date=None,
                 init_cash=100000, leverage=1, fees=0, freq="1m"):
    import datetime as dt
    if start_date is None:
        start_date = signals_df.index[0]
    if end_date is None:
        end_date = signals_df.index[-1]

    ohlcv = signals_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

    if ohlcv.index.tz is None:
        ohlcv.index = ohlcv.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(ohlcv.index.tz) != 'America/New_York':
        ohlcv.index = ohlcv.index.tz_convert('America/New_York')

    signals = signals_df.copy()
    if signals.index.tz is None:
        signals.index = signals.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(signals.index.tz) != 'America/New_York':
        signals.index = signals.index.tz_convert('America/New_York')

    if hasattr(start_date, 'tz'):
        if start_date.tz is None:
            start_date = start_date.tz_localize('UTC').tz_convert('America/New_York')
        elif str(start_date.tz) != 'America/New_York':
            start_date = start_date.tz_convert('America/New_York')
    if hasattr(end_date, 'tz'):
        if end_date.tz is None:
            end_date = end_date.tz_localize('UTC').tz_convert('America/New_York')
        elif str(end_date.tz) != 'America/New_York':
            end_date = end_date.tz_convert('America/New_York')

    asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
    timestep = "minute" if freq == "1m" else "day"
    hours = ohlcv.index.hour.unique()
    has_ext_hours = any((h < 9 or h >= 16) for h in hours)

    if has_ext_hours:
        data_object = Data(
            asset=asset, df=ohlcv, timestep=timestep,
            trading_hours_start=dt.time(0, 0),
            trading_hours_end=dt.time(23, 59)
        )
    else:
        data_object = Data(asset=asset, df=ohlcv, timestep=timestep)

    pandas_data = {asset: data_object}

    try:
        SignalBasedStrategy.run_backtest(
            PandasDataBacktesting, start_date, end_date,
            pandas_data=pandas_data,
            parameters={
                "symbol": symbol, "signals_df": signals,
                "init_cash": init_cash, "leverage": leverage, "fees": fees,
            },
            budget=init_cash,
            show_plot=False, show_tearsheet=False, show_indicators=False,
        )
        trades = SignalBasedStrategy._trades_storage
        max_exposure = SignalBasedStrategy._max_exposure_storage
    except Exception as e:
        logger.error(f'Backtest failed: {e}', exc_info=True)
        return [], _empty_stats()

    for t in trades:
        if 'entry_date' in t and hasattr(t['entry_date'], 'tz'):
            t['entry_date'] = t['entry_date'].tz_convert('UTC')
        if 'exit_date' in t and hasattr(t['exit_date'], 'tz'):
            t['exit_date'] = t['exit_date'].tz_convert('UTC')

    stats = _calculate_stats(trades, init_cash, signals_df, leverage, fees, max_exposure)
    return trades, stats


def _calculate_stats(trades, init_cash, signals_df, leverage, fees, max_exposure):
    if not trades:
        return _empty_stats()

    trades_df = pd.DataFrame(trades)
    portfolio = pd.Series(init_cash, index=signals_df.index)
    current_value = init_cash

    for _, t in trades_df.iterrows():
        if t['exit_date'] in portfolio.index:
            current_value += t['pnl']
            portfolio.loc[t['exit_date']:] = current_value

    returns = portfolio.pct_change().fillna(0)
    start_value, end_value = init_cash, portfolio.iloc[-1]
    total_return = ((end_value - start_value) / start_value) * 100
    time_diff = (returns.index[1] - returns.index[0]).total_seconds() if len(returns) > 1 else 86400
    periods = 252 * 390 if time_diff < 3600 else 252

    max_drawdown = abs(qs.stats.max_drawdown(returns) * 100)
    dd_series = qs.stats.to_drawdown_series(returns)
    dd_details = qs.stats.drawdown_details(dd_series)
    max_dd_duration = pd.Timedelta(days=dd_details['days'].max()) if not dd_details.empty else pd.Timedelta(0)

    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] < 0]
    win_rate = qs.stats.win_rate(returns) * 100
    best_trade = max(t['return'] for t in trades)
    worst_trade = min(t['return'] for t in trades)
    avg_win = np.mean([t['return'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['return'] for t in losing]) if losing else 0

    def avg_duration(lst):
        if not lst: return pd.Timedelta(0)
        return sum([t['exit_date'] - t['entry_date'] for t in lst], pd.Timedelta(0)) / len(lst)

    avg_win_dur = avg_duration(winning)
    avg_loss_dur = avg_duration(losing)
    profit_factor = qs.stats.profit_factor(returns)
    if np.isinf(profit_factor):
        wins = sum(t['pnl'] for t in winning)
        losses = abs(sum(t['pnl'] for t in losing))
        profit_factor = wins / losses if losses > 0 else np.inf

    expectancy = np.mean([t['pnl'] for t in trades])
    total_fees = len(trades) * 2 * fees
    total_duration = signals_df.index[-1] - signals_df.index[0]

    sharpe = qs.stats.sharpe(returns, rf=0, periods=periods, annualize=True)
    calmar = qs.stats.calmar(returns, periods=periods)
    sortino = qs.stats.sortino(returns, rf=0, periods=periods, annualize=True)
    omega = qs.stats.omega(returns, rf=0.0, required_return=0.0, periods=periods)
    in_position_time = sum([(t['exit_date'] - t['entry_date']).total_seconds() for t in trades])
    position_coverage = (in_position_time / total_duration.total_seconds()) * 100

    return {
        'Start Index': signals_df.index[0],
        'End Index': signals_df.index[-1],
        'Total Duration': str(total_duration),
        'Start Value': start_value,
        'End Value': end_value,
        'Total Return [%]': total_return,
        'Position Coverage [%]': position_coverage,
        'Max Gross Exposure [%]': max_exposure * leverage,
        'Max Drawdown [%]': max_drawdown,
        'Max Drawdown Duration': str(max_dd_duration),
        'Total Trades': len(trades),
        'Win Rate [%]': win_rate,
        'Best Trade [%]': best_trade,
        'Worst Trade [%]': worst_trade,
        'Avg Winning Trade [%]': avg_win,
        'Avg Losing Trade [%]': avg_loss,
        'Avg Winning Trade Duration': str(avg_win_dur),
        'Avg Losing Trade Duration': str(avg_loss_dur),
        'Profit Factor': profit_factor,
        'Expectancy': expectancy,
        'Sharpe Ratio': sharpe,
        'Calmar Ratio': calmar,
        'Omega Ratio': omega,
        'Sortino Ratio': sortino,
        'Total Fees Paid': total_fees,
    }


def _empty_stats():
    return {k: 0 for k in [
        'Start Value', 'End Value', 'Total Return [%]', 'Position Coverage [%]',
        'Max Gross Exposure [%]', 'Max Drawdown [%]', 'Total Trades',
        'Win Rate [%]', 'Profit Factor', 'Expectancy', 'Sharpe Ratio',
        'Calmar Ratio', 'Omega Ratio', 'Sortino Ratio', 'Total Fees Paid'
    ]}
