Usage example:

```python
from numba.typed import List

from orderbook import Orderbook, collect_metrics, metrics_tup_ty 
from services import *

tick_size, lot_size = 0.0001, 0.01
book = Orderbook(tick_size, lot_size)
buf = List.empty_list(metrics_tup_ty, allocated=100_000)

events = convert_tardis(pd.read_csv( f'{sym}-{d}.csv.gz', compression='gzip'))
buf.extend(collect_metrics(book, events))

df = numba_metrics_to_df(buf)
```
