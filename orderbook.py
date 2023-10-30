import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from numba import njit, float64, int64
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType, Tuple, float64


INVALID_MIN = -sys.maxsize
INVALID_MAX = sys.maxsize

BUY = 1
SELL = 0

COL_EXCH_TIMESTAMP = 0
COL_LOCAL_TIMESTAMP = 1
COL_SIDE = 2
COL_PRICE = 3
COL_QTY = 4


@njit
def best_below(book, start, end):
    for t in range(start - 1, end - 1, -1):
        if t in book and book[t] > 0:
            return t
    return INVALID_MIN


@njit
def best_above(book, start, end):
    for t in range(start + 1, end + 1):
        if t in book and book[t] > 0:
            return t
    return INVALID_MAX


@jitclass
class Orderbook:
    """
    Numba optimized orderbook data structure. The price-amount key-vales pairs on
    bids/asks are stored in a hashmap rather than red-black-tree for simplifed
    implementation. Prices are indexed in integer units of tick size, so the hashmap is
    typed (int -> float).

    Originally lifted from https://github.com/nkaz001/hftbacktest MarketDepth class,
    with many modifications since then.
    """
    tick_size: float64
    lot_size: float64
    asks: DictType(int64, float64) # type: ignore
    bids: DictType(int64, float64) # type: ignore
    best_bid_tick: int64
    best_ask_tick: int64
    worst_bid_tick: int64
    worst_ask_tick: int64
    last_ts: int64

    def __init__(self, tick_size, lot_size):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.asks = Dict.empty(int64, float64)
        self.bids = Dict.empty(int64, float64)
        self.best_bid_tick = INVALID_MIN
        self.best_ask_tick = INVALID_MAX
        self.worst_bid_tick = INVALID_MAX
        self.worst_ask_tick = INVALID_MIN
        self.last_ts = INVALID_MIN

    ## Raw metrics
    @property
    def best_bid(self):
        return self.tick_size * self.best_bid_tick

    @property
    def best_ask(self):
        return self.tick_size * self.best_ask_tick

    @property
    def mid_price(self):
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self):
        if self.mid_price == 0.:
            return 0.
        return (self.best_bid - self.best_ask) / self.mid_price

    def bid_qty_till(self, bps: float64) -> float64:
        till_tick = max(
            round((self.best_bid * (1 - bps)) / self.tick_size),
            self.worst_bid_tick,
        )
        r = 0
        for t in range(till_tick, self.best_bid_tick):
            # Todo we should not need to set default?
            r += self.bids.get(t, 0)
        return r

    def ask_qty_till(self, bps: float64) -> float64:
        till_tick = min(
            round((self.best_ask * (1 + bps)) / self.tick_size),
            self.worst_ask_tick,
        )
        r = 0
        for t in range(self.best_ask_tick, till_tick):
            # Todo we should not need to set default?
            r += self.asks.get(t, 0)
        return r

    ## Derived metrics
    def imbalance(self, bps) -> float64:
        bids_qty = self.bid_qty_till(bps)
        asks_qty = self.ask_qty_till(bps)
        if (bids_qty + asks_qty) > 0:
            return (bids_qty - asks_qty) / (bids_qty + asks_qty)
        else:
            return 0.

    def pressure_bid(self, narrow_bps, wide_bps) -> float64:
        wide_pressure = self.bid_qty_till(wide_bps)
        if wide_pressure > 0:
            return self.bid_qty_till(narrow_bps) / wide_pressure
        else:
            return 0.

    def pressure_ask(self, narrow_bps, wide_bps) -> float64:
        wide_pressure = self.ask_qty_till(wide_bps)
        if wide_pressure > 0:
            return self.ask_qty_till(narrow_bps) / wide_pressure
        else:
            return 0.

    ## Orderbook ops
    def update_bids(self, price: float64, qty: float64) -> None:
        price_tick: int = round(price / self.tick_size)

        if round(qty / self.lot_size) == 0:
            # Todo we should not need to check level existence
            if price_tick in self.bids:
                del self.bids[price_tick]
            if price_tick == self.best_bid_tick:
                self.best_bid_tick = best_below(
                    self.bids, self.best_bid_tick, self.worst_bid_tick
                )
                if self.best_bid_tick == INVALID_MIN:
                    self.worst_bid_tick = INVALID_MAX
        else:
            self.bids[price_tick] = qty
            if price_tick > self.best_bid_tick:
                self.best_bid_tick = price_tick
                # this path shouldn't happen, if the exchange sent events in order
                if self.best_bid_tick >= self.best_ask_tick:
                    self.best_ask_tick = best_above(
                        self.asks, self.best_bid_tick, self.worst_ask_tick
                    )
            if price_tick < self.worst_bid_tick:
                self.worst_bid_tick = price_tick

    def update_asks(self, price: float64, qty: float64) -> None:
        price_tick: int = round(price / self.tick_size)

        if round(qty // self.lot_size) == 0:
            # Todo we should not need to check level existence
            if price_tick in self.asks:
                del self.asks[price_tick]
            if price_tick == self.best_ask_tick:
                self.best_ask_tick = best_above(
                    self.asks, self.best_ask_tick, self.worst_ask_tick
                )
                if self.best_ask_tick == INVALID_MAX:
                    self.worst_ask_tick = INVALID_MIN
        else:
            self.asks[price_tick] = qty
            if price_tick > self.worst_ask_tick:
                self.worst_ask_tick = price_tick
            if price_tick < self.best_ask_tick:
                self.best_ask_tick = price_tick
                # this path shouldn't happen, if the exchange sent events in order
                if self.best_ask_tick <= self.best_bid_tick:
                    self.best_bid_tick = best_below(
                        self.bids, self.best_ask_tick, self.worst_bid_tick
                    )


@njit
def process_one(evt, book):
    """Apply one event onto an orderbook."""
    # Process a depth event
    if evt[COL_SIDE] == BUY:
        book.update_bids(evt[COL_PRICE], evt[COL_QTY])
    else:
        book.update_asks(evt[COL_PRICE], evt[COL_QTY])
    book.last_ts = evt[COL_LOCAL_TIMESTAMP]


@njit
def process(events, book):
    """Plain function to process an array of events onto an orderbook."""
    for evt in events:
        process_one(evt, book)
    return book


metrics_tup_ty = Tuple((float64,) * 9)

@njit
def collect_metrics(book, events, collect_freq=5 * 60 * 1e6):
    out = List.empty_list(metrics_tup_ty, allocated=1_000)
    for evt in events:
        curr_ts = evt[COL_LOCAL_TIMESTAMP]
        prev_ts = book.last_ts
        if (
            prev_ts != INVALID_MIN
            and round(prev_ts // collect_freq) < round(curr_ts // collect_freq)
        ):
            out.append((
                prev_ts,
                book.best_bid,
                book.best_ask,
                book.bid_qty_till(.0025),
                book.bid_qty_till(.005),
                book.bid_qty_till(.0075),
                book.ask_qty_till(.0025),
                book.ask_qty_till(.005),
                book.ask_qty_till(.0075),
            ))
        process_one(evt, book)
    return out


def numba_metrics_to_df(buf):
    df = pd.DataFrame(buf)
    df.columns = [
        'Timestamp',
        'Best bid',
        'Best ask',
        'Bid volume till 25bps',
        'Bid volume till 50bps',
        'Bid volume till 75bps',
        'Ask volume till 25bps',
        'Ask volume till 50bps',
        'Ask volume till 75bps',
    ]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='us')
    df = df.set_index('Timestamp')
    df = df.resample('5T', closed='right', label='right').last()
    return df
