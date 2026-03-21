# 作用：

# 定义 Backtrader 数据源扩展
# 定义第一版事件驱动策略：
# 只做多
# 事件信号触发后，下一交易日开盘买入
# 持有固定天数或触发止盈/止损后卖出
# 同时只持有一笔

from __future__ import annotations

import math
from typing import Any

import backtrader as bt


class EventSignalData(bt.feeds.PandasData):
    """
    扩展 PandasData，增加：
    - event_signal: 是否触发做多信号（1/0）
    - signal_prob: 模型预测上涨概率
    - signal_threshold: 该信号对应的判定阈值
    - event_type_num: 数值型事件类型
    """
    lines = ("event_signal", "signal_prob", "signal_threshold", "event_type_num")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("event_signal", "event_signal"),
        ("signal_prob", "signal_prob"),
        ("signal_threshold", "signal_threshold"),
        ("event_type_num", "event_type_num"),
    )


class EventPredictionStrategy(bt.Strategy):
    params = dict(
        hold_days=3,
        prob_threshold=None,      # 若为 None，则使用每条信号自带的 signal_threshold
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        signal_meta_map=None,     # {signal_date: {...}}
    )

    def __init__(self):
        self.order = None

        self.entry_bar = None
        self.entry_price = None
        self.entry_size = None
        self.entry_context: dict[str, Any] | None = None
        self.pending_entry_context: dict[str, Any] | None = None
        self.last_exit_context: dict[str, Any] | None = None
        self.current_exit_reason: str | None = None

        self.trade_log: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []

        self.signal_meta_map = self.p.signal_meta_map or {}

    def _get_current_date_str(self) -> str:
        return bt.num2date(self.data.datetime[0]).date().isoformat()

    def _safe_line_value(self, line) -> float | None:
        value = float(line[0])
        if math.isnan(value):
            return None
        return value

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            executed_date = bt.num2date(order.executed.dt).date().isoformat()

            if order.isbuy():
                self.entry_bar = len(self)
                self.entry_price = float(order.executed.price)
                self.entry_size = int(order.executed.size)

                entry_context = dict(self.pending_entry_context or {})
                entry_context["entry_date"] = executed_date
                entry_context["entry_price"] = self.entry_price
                entry_context["entry_size"] = self.entry_size
                self.entry_context = entry_context

            elif order.issell():
                self.last_exit_context = {
                    "exit_date": executed_date,
                    "exit_price": float(order.executed.price),
                    "exit_reason": self.current_exit_reason or "manual",
                }

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
            self.pending_entry_context = None
            self.current_exit_reason = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        entry_ctx = self.entry_context or {}
        exit_ctx = self.last_exit_context or {}

        entry_price = float(entry_ctx.get("entry_price", 0.0))
        exit_price = float(exit_ctx.get("exit_price", 0.0))
        pnl_pct = ((exit_price - entry_price) / entry_price) if entry_price else 0.0

        self.trade_log.append(
            {
                "signal_date": entry_ctx.get("signal_date"),
                "entry_date": entry_ctx.get("entry_date"),
                "exit_date": exit_ctx.get("exit_date"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": entry_ctx.get("entry_size"),
                "hold_days": trade.barlen,
                "pnl": float(trade.pnl),
                "pnl_comm": float(trade.pnlcomm),
                "return_pct": float(pnl_pct),
                "signal_prob": entry_ctx.get("signal_prob"),
                "signal_threshold": entry_ctx.get("signal_threshold"),
                "event_title": entry_ctx.get("event_title"),
                "event_source": entry_ctx.get("event_source"),
                "event_type_raw": entry_ctx.get("event_type_raw"),
                "exit_reason": exit_ctx.get("exit_reason"),
            }
        )

        self.entry_bar = None
        self.entry_price = None
        self.entry_size = None
        self.entry_context = None
        self.pending_entry_context = None
        self.last_exit_context = None
        self.current_exit_reason = None

    def next(self):
        current_date_str = self._get_current_date_str()

        # 记录资金曲线
        self.equity_curve.append(
            {
                "date": current_date_str,
                "portfolio_value": float(self.broker.getvalue()),
                "cash": float(self.broker.getcash()),
                "close": float(self.data.close[0]),
            }
        )

        if self.order:
            return

        # 已有持仓：检查平仓条件
        if self.position:
            current_close = float(self.data.close[0])
            pnl_pct = ((current_close - self.entry_price) / self.entry_price) if self.entry_price else 0.0
            hold_days = len(self) - self.entry_bar if self.entry_bar is not None else 0

            if pnl_pct <= -float(self.p.stop_loss_pct):
                self.current_exit_reason = "stop_loss"
                self.order = self.close()
                return

            if pnl_pct >= float(self.p.take_profit_pct):
                self.current_exit_reason = "take_profit"
                self.order = self.close()
                return

            if hold_days >= int(self.p.hold_days):
                self.current_exit_reason = "hold_days"
                self.order = self.close()
                return

            return

        # 空仓：检查是否生成做多信号
        event_signal = self._safe_line_value(self.data.event_signal)
        signal_prob = self._safe_line_value(self.data.signal_prob)
        signal_threshold = self._safe_line_value(self.data.signal_threshold)

        if event_signal is None or signal_prob is None:
            return

        if self.p.prob_threshold is not None:
            decision_threshold = float(self.p.prob_threshold)
        else:
            decision_threshold = float(signal_threshold) if signal_threshold is not None else 0.50

        if int(event_signal) == 1 and float(signal_prob) >= decision_threshold:
            signal_meta = self.signal_meta_map.get(current_date_str, {})

            self.pending_entry_context = {
                "signal_date": current_date_str,
                "signal_prob": float(signal_prob),
                "signal_threshold": float(decision_threshold),
                "event_title": signal_meta.get("event_title", ""),
                "event_source": signal_meta.get("event_source", ""),
                "event_type_raw": signal_meta.get("event_type_raw", 0),
            }

            self.order = self.buy()