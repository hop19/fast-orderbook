Usage example (see `process.py` for most update to date version):

```python
import boto3
from numba.typed import List

from orderbook import Orderbook, collect_metrics, metrics_tup_ty
from services import *

session = boto3.session.Session()
s3 = session.resource('s3')

# Perp metadata and initialise Orderbook object + metrics buffer
BUCKET = 'hop19-public-shared'
sym, d = 'BTCUSDT', '2023/10/01'

binfo = bnc_fut_universe_trading_param(bnc_fut_info())
tick_size, lot_size = binfo[sym]['tick_size'], binfo[sym]['lot_size']
book = Orderbook(tick_size, lot_size)
buf = List.empty_list(metrics_tup_ty, allocated=100_000)

# Fetch a day of BTC raw diff and process it
events = convert_tardis(pd.read_csv(
    f's3://{BUCKET}/tardis-incremental-book/{d}/{sym}.csv.gz',
    compression='gzip',
))
buf.extend(collect_metrics(book, events))
df = numba_metrics_to_df(buf)
```
