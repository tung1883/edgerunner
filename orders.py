
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

