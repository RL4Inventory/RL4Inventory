# Use base-stock policy

The main is the `inventory_train_base_stock_policy.py`.

use oracle bs:

  run inventory_train_base_stock.py

​    --buyin_gap ?: how often to place an order 

​    --stop_order ?: the factor that controls the stop order point

  by default, the `update_interval` is exactly `evaluation_len`

​        the `start_step` is exactly 0

use dynamic bs:

  run inventory_train_base_stock.py

​    --update_interval 1: how often to update the base-stock levels

​    --start_step -7: where to start the base-stock levels pickup amid the horizon

​    --buyin_gap ?

​    --stop_order ?

use static bs:

​	run inventory_train_base_stock.py

​    --update_interval 1 

​    --start_step -7

​    --static [necessary]