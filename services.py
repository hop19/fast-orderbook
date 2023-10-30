"""
This module contains utilities working with third party services. I.e. Binance and
Tardis.
"""

import datetime as dt
import io
import json
import pathlib

import pandas as pd
import requests
import toolz as tz
from numpy.typing import NDArray
from pandas import DataFrame


def fmt_date_url(d: dt.date) -> str:
    return d.isoformat().replace("-", "/")


def tardis_url(d, sym, dtype, exchange='binance-futures'):
    d = fmt_date_url(d)
    return f'https://datasets.tardis.dev/v1/{exchange}/{dtype}/{d}/{sym}.csv.gz'


def find_filter(sinfo, filter_name):
    return tz.first(filter(lambda f: f['filterType'] == filter_name, sinfo['filters']))


def bnc_fut_info():
    if pathlib.Path('data/binfo').exists():
        return json.load(open('data/binfo'))
    else:
        return requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo').json()


def is_bnc_trading_universe(sinfo):
    return (
        sinfo['contractType'] == 'PERPETUAL'
        and sinfo['quoteAsset'] == 'USDT'
        and sinfo['underlyingType'] == 'COIN'
        and sinfo['baseAsset'] != 'USDC'
    )


def bnc_fut_trading_param(sinfo):
    return {
        "lot_size": float(find_filter(sinfo, 'LOT_SIZE')['stepSize']),
        "tick_size": float(find_filter(sinfo, 'PRICE_FILTER')['tickSize']),
    }


def bnc_fut_universe_trading_param(binfo):
    return {
        sinfo['symbol']: bnc_fut_trading_param(sinfo)
        for sinfo in binfo['symbols']
        if is_bnc_trading_universe(sinfo)
    }


def get_tardis_dataset_info(sym, dtype='incremental_book_L2'):
    url = tardis_url(dt.date(2019, 1, 1), sym, dtype)
    return requests.get(url).json()['datasetInfo']


def get_listing_date(sym, dtype='incremental_book_L2'):
    """Get the first and last day when tardis data is available for a data type."""
    r = get_tardis_dataset_info(sym, dtype)
    return r['availableSince'], r['availableTo']


def convert_tardis(arg: str | bytes | DataFrame) -> NDArray:
    if isinstance(arg, str):
        if arg.endswith('.csv.gz'):
            df = pd.read_csv(arg, compression='gzip')
        elif arg.endswith('.csv'):
            df = pd.read_csv(arg)
        else:
            raise ValueError('Unknown file extension')
    elif isinstance(arg, bytes):
        df = pd.read_csv(io.BytesIO(arg), compression='gzip')
    else:
        df = arg

    return df[['timestamp', 'local_timestamp', 'side', 'price', 'amount']].assign(
        side=df['side'] == 'bid',
    ).astype(float).values
