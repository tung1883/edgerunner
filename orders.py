
from dataclasses import dataclass
from typing import Literal

@dataclass
class Order:
    ticker:  str
    side:    Literal['buy', 'sell']
    qty:     float
    order_type: Literal['market', 'limit'] = 'market'
    limit_price: float = 0.0

    def can_fill(self, price):
        if self.order_type == 'market':
            return True
        return (self.side == 'buy'  and price <= self.limit_price) or                (self.side == 'sell' and price >= self.limit_price)


class OrderBook:
    def __init__(self):
        self.open_orders = []
        self.filled = []

    def submit(self, ticker, qty, order_type="market", limit_price=None):
        order = {"ticker": ticker, "qty": qty, "type": order_type, "limit": limit_price, "status": "open"}
        self.open_orders.append(order)
        return len(self.open_orders) - 1

    def fill(self, idx, fill_price):
        o = self.open_orders[idx]
        if o["type"] == "limit" and fill_price > o["limit"]:
            return False
        o["status"] = "filled"
        o["fill_price"] = fill_price
        self.filled.append(o)
        return True

    def cancel(self, idx):
        self.open_orders[idx]["status"] = "cancelled"
