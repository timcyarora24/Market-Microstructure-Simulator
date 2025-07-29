import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import heapq
from datetime import datetime, timedelta
import uuid

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Represents a single order in the order book"""
    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    trader_id: str = "anonymous"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_market_order(self) -> bool:
        return self.order_type == OrderType.MARKET
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

@dataclass
class Trade:
    """Represents an executed trade"""
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: float
    timestamp: datetime
    aggressor_side: OrderSide

class PriceLevel:
    """Represents a price level in the order book"""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: deque = deque()
        self.total_quantity: float = 0.0
    
    def add_order(self, order: Order):
        """Add order to this price level"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order from this price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                removed_order = self.orders[i]
                del self.orders[i]
                self.total_quantity -= removed_order.remaining_quantity
                return removed_order
        return None
    
    def get_first_order(self) -> Optional[Order]:
        """Get the first order in the queue (FIFO)"""
        return self.orders[0] if self.orders else None
    
    def is_empty(self) -> bool:
        """Check if price level is empty"""
        return len(self.orders) == 0

class LimitOrderBook:
    """
    Complete limit order book implementation with advanced features
    """
    
    def __init__(self, tick_size: float = 0.01):
        """
        Initialize limit order book
        
        Args:
            tick_size: Minimum price increment
        """
        self.tick_size = tick_size
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        
        # Price levels (using heaps for efficient best price retrieval)
        self.bid_levels: Dict[float, PriceLevel] = {}  # Buy orders
        self.ask_levels: Dict[float, PriceLevel] = {}  # Sell orders
        
        # Price queues for efficient access to best prices
        self.bid_prices = []  # Max heap (negative values)
        self.ask_prices = []  # Min heap
        
        # Trade history
        self.trades: List[Trade] = []
        
        # Market data
        self.last_price: Optional[float] = None
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.mid_price: Optional[float] = None
        self.spread: Optional[float] = None
        
        # Statistics
        self.total_volume: float = 0.0
        self.trade_count: int = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add order to the book and return any resulting trades
        
        Args:
            order: Order to add
            
        Returns:
            List of trades executed
        """
        trades = []
        
        # Store order
        self.orders[order.order_id] = order
        
        if order.is_market_order:
            trades = self._execute_market_order(order)
        else:
            trades = self._add_limit_order(order)
        
        # Update market data
        self._update_market_data()
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
            return False
        
        # Remove from appropriate price level
        if order.is_buy:
            if order.price in self.bid_levels:
                self.bid_levels[order.price].remove_order(order_id)
                if self.bid_levels[order.price].is_empty():
                    del self.bid_levels[order.price]
        else:
            if order.price in self.ask_levels:
                self.ask_levels[order.price].remove_order(order_id)
                if self.ask_levels[order.price].is_empty():
                    del self.ask_levels[order.price]
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        
        # Update market data
        self._update_market_data()
        
        return True
    
    def modify_order(self, order_id: str, new_quantity: Optional[float] = None, 
                    new_price: Optional[float] = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: ID of order to modify
            new_quantity: New quantity (if provided)
            new_price: New price (if provided)
            
        Returns:
            True if modification successful
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
            return False
        
        # For price changes, we need to cancel and re-add
        if new_price is not None and new_price != order.price:
            # Remove from current price level
            self.cancel_order(order_id)
            
            # Create new order with updated price
            new_order = Order(
                order_id=str(uuid.uuid4()),
                side=order.side,
                order_type=order.order_type,
                quantity=new_quantity if new_quantity else order.remaining_quantity,
                price=new_price,
                timestamp=datetime.now(),
                trader_id=order.trader_id
            )
            
            return len(self.add_order(new_order)) >= 0
        
        # For quantity-only changes
        if new_quantity is not None:
            if new_quantity > order.remaining_quantity:
                # Increasing quantity
                order.quantity = order.filled_quantity + new_quantity
                order.remaining_quantity = new_quantity
            else:
                # Decreasing quantity
                order.quantity = order.filled_quantity + new_quantity
                order.remaining_quantity = new_quantity
        
        return True
    
    def _add_limit_order(self, order: Order) -> List[Trade]:
        """Add limit order and try to match against existing orders"""
        trades = []
        
        if order.is_buy:
            # Try to match against ask orders
            while order.remaining_quantity > 0 and self.ask_prices:
                best_ask = self.ask_prices[0]
                
                if order.price >= best_ask:
                    # Match possible
                    ask_level = self.ask_levels[best_ask]
                    opposing_order = ask_level.get_first_order()
                    
                    if opposing_order:
                        trade = self._execute_trade(order, opposing_order, best_ask)
                        trades.append(trade)
                        
                        # Update order quantities
                        fill_quantity = trade.quantity
                        order.filled_quantity += fill_quantity
                        order.remaining_quantity -= fill_quantity
                        opposing_order.filled_quantity += fill_quantity
                        opposing_order.remaining_quantity -= fill_quantity
                        
                        # Update order statuses
                        if order.remaining_quantity == 0:
                            order.status = OrderStatus.FILLED
                        elif order.filled_quantity > 0:
                            order.status = OrderStatus.PARTIAL_FILLED
                            
                        if opposing_order.remaining_quantity == 0:
                            opposing_order.status = OrderStatus.FILLED
                            ask_level.orders.popleft()
                            ask_level.total_quantity -= fill_quantity
                        elif opposing_order.filled_quantity > 0:
                            opposing_order.status = OrderStatus.PARTIAL_FILLED
                            ask_level.total_quantity -= fill_quantity
                        
                        # Remove empty price level
                        if ask_level.is_empty():
                            del self.ask_levels[best_ask]
                            heapq.heappop(self.ask_prices)
                    else:
                        break
                else:
                    break
            
            # Add remaining quantity to book
            if order.remaining_quantity > 0:
                self._add_to_bid_book(order)
        
        else:  # Sell order
            # Try to match against bid orders
            while order.remaining_quantity > 0 and self.bid_prices:
                best_bid = -self.bid_prices[0]  # Convert back from negative
                
                if order.price <= best_bid:
                    # Match possible
                    bid_level = self.bid_levels[best_bid]
                    opposing_order = bid_level.get_first_order()
                    
                    if opposing_order:
                        trade = self._execute_trade(opposing_order, order, best_bid)
                        trades.append(trade)
                        
                        # Update order quantities
                        fill_quantity = trade.quantity
                        order.filled_quantity += fill_quantity
                        order.remaining_quantity -= fill_quantity
                        opposing_order.filled_quantity += fill_quantity
                        opposing_order.remaining_quantity -= fill_quantity
                        
                        # Update order statuses
                        if order.remaining_quantity == 0:
                            order.status = OrderStatus.FILLED
                        elif order.filled_quantity > 0:
                            order.status = OrderStatus.PARTIAL_FILLED
                            
                        if opposing_order.remaining_quantity == 0:
                            opposing_order.status = OrderStatus.FILLED
                            bid_level.orders.popleft()
                            bid_level.total_quantity -= fill_quantity
                        elif opposing_order.filled_quantity > 0:
                            opposing_order.status = OrderStatus.PARTIAL_FILLED
                            bid_level.total_quantity -= fill_quantity
                        
                        # Remove empty price level
                        if bid_level.is_empty():
                            del self.bid_levels[best_bid]
                            heapq.heappop(self.bid_prices)
                    else:
                        break
                else:
                    break
            
            # Add remaining quantity to book
            if order.remaining_quantity > 0:
                self._add_to_ask_book(order)
        
        return trades
    
    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute market order against best available prices"""
        trades = []
        
        if order.is_buy:
            # Match against ask orders
            while order.remaining_quantity > 0 and self.ask_prices:
                best_ask = self.ask_prices[0]
                ask_level = self.ask_levels[best_ask]
                opposing_order = ask_level.get_first_order()
                
                if opposing_order:
                    trade = self._execute_trade(order, opposing_order, best_ask)
                    trades.append(trade)
                    
                    # Update quantities and status (similar to limit order logic)
                    fill_quantity = trade.quantity
                    order.filled_quantity += fill_quantity
                    order.remaining_quantity -= fill_quantity
                    opposing_order.filled_quantity += fill_quantity
                    opposing_order.remaining_quantity -= fill_quantity
                    
                    if order.remaining_quantity == 0:
                        order.status = OrderStatus.FILLED
                    elif order.filled_quantity > 0:
                        order.status = OrderStatus.PARTIAL_FILLED
                        
                    if opposing_order.remaining_quantity == 0:
                        opposing_order.status = OrderStatus.FILLED
                        ask_level.orders.popleft()
                        ask_level.total_quantity -= fill_quantity
                        
                        if ask_level.is_empty():
                            del self.ask_levels[best_ask]
                            heapq.heappop(self.ask_prices)
                    elif opposing_order.filled_quantity > 0:
                        opposing_order.status = OrderStatus.PARTIAL_FILLED
                        ask_level.total_quantity -= fill_quantity
                else:
                    break
        else:
            # Match against bid orders
            while order.remaining_quantity > 0 and self.bid_prices:
                best_bid = -self.bid_prices[0]
                bid_level = self.bid_levels[best_bid]
                opposing_order = bid_level.get_first_order()
                
                if opposing_order:
                    trade = self._execute_trade(opposing_order, order, best_bid)
                    trades.append(trade)
                    
                    # Update quantities and status
                    fill_quantity = trade.quantity
                    order.filled_quantity += fill_quantity
                    order.remaining_quantity -= fill_quantity
                    opposing_order.filled_quantity += fill_quantity
                    opposing_order.remaining_quantity -= fill_quantity
                    
                    if order.remaining_quantity == 0:
                        order.status = OrderStatus.FILLED
                    elif order.filled_quantity > 0:
                        order.status = OrderStatus.PARTIAL_FILLED
                        
                    if opposing_order.remaining_quantity == 0:
                        opposing_order.status = OrderStatus.FILLED
                        bid_level.orders.popleft()
                        bid_level.total_quantity -= fill_quantity
                        
                        if bid_level.is_empty():
                            del self.bid_levels[best_bid]
                            heapq.heappop(self.bid_prices)
                    elif opposing_order.filled_quantity > 0:
                        opposing_order.status = OrderStatus.PARTIAL_FILLED
                        bid_level.total_quantity -= fill_quantity
                else:
                    break
        
        # If market order is not fully filled, reject remaining quantity
        if order.remaining_quantity > 0:
            order.status = OrderStatus.PARTIAL_FILLED if order.filled_quantity > 0 else OrderStatus.REJECTED
        
        return trades
    
    def _add_to_bid_book(self, order: Order):
        """Add order to bid side of book"""
        if order.price not in self.bid_levels:
            self.bid_levels[order.price] = PriceLevel(order.price)
            heapq.heappush(self.bid_prices, -order.price)  # Negative for max heap
        
        self.bid_levels[order.price].add_order(order)
    
    def _add_to_ask_book(self, order: Order):
        """Add order to ask side of book"""
        if order.price not in self.ask_levels:
            self.ask_levels[order.price] = PriceLevel(order.price)
            heapq.heappush(self.ask_prices, order.price)  # Min heap
        
        self.ask_levels[order.price].add_order(order)
    
    def _execute_trade(self, buy_order: Order, sell_order: Order, price: float) -> Trade:
        """Execute trade between two orders"""
        quantity = min(buy_order.remaining_quantity, sell_order.remaining_quantity)
        
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            aggressor_side=buy_order.side if buy_order.timestamp > sell_order.timestamp else sell_order.side
        )
        
        self.trades.append(trade)
        self.total_volume += quantity
        self.trade_count += 1
        self.last_price = price
        
        return trade
    
    def _update_market_data(self):
        """Update best bid/ask and derived metrics"""
        # Update best bid
        if self.bid_prices:
            self.best_bid = -self.bid_prices[0]  # Convert from negative
        else:
            self.best_bid = None
        
        # Update best ask
        if self.ask_prices:
            self.best_ask = self.ask_prices[0]
        else:
            self.best_ask = None
        
        # Update mid price and spread
        if self.best_bid is not None and self.best_ask is not None:
            self.mid_price = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid
        else:
            self.mid_price = None
            self.spread = None
    
    def get_order_book_snapshot(self, levels: int = 10) -> Dict[str, Any]:
        """
        Get current order book snapshot
        
        Args:
            levels: Number of price levels to include on each side
            
        Returns:
            Dictionary with order book data
        """
        # Get top bid levels
        bid_prices_sorted = sorted(self.bid_levels.keys(), reverse=True)[:levels]
        bids = []
        for price in bid_prices_sorted:
            level = self.bid_levels[price]
            bids.append({
                'price': price,
                'quantity': level.total_quantity,
                'orders': len(level.orders)
            })
        
        # Get top ask levels
        ask_prices_sorted = sorted(self.ask_levels.keys())[:levels]
        asks = []
        for price in ask_prices_sorted:
            level = self.ask_levels[price]
            asks.append({
                'price': price,
                'quantity': level.total_quantity,
                'orders': len(level.orders)
            })
        
        return {
            'timestamp': datetime.now(),
            'bids': bids,
            'asks': asks,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'last_price': self.last_price,
            'total_volume': self.total_volume,
            'trade_count': self.trade_count
        }
    
    def get_market_depth(self, max_price_range: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get market depth data for visualization
        
        Args:
            max_price_range: Maximum price range from mid price
            
        Returns:
            Tuple of (bid_prices, bid_quantities, ask_prices, ask_quantities)
        """
        if self.mid_price is None:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Bid side
        bid_prices = []
        bid_quantities = []
        for price in sorted(self.bid_levels.keys(), reverse=True):
            if self.mid_price - price <= max_price_range:
                bid_prices.append(price)
                bid_quantities.append(self.bid_levels[price].total_quantity)
        
        # Ask side
        ask_prices = []
        ask_quantities = []
        for price in sorted(self.ask_levels.keys()):
            if price - self.mid_price <= max_price_range:
                ask_prices.append(price)
                ask_quantities.append(self.ask_levels[price].total_quantity)
        
        return (np.array(bid_prices), np.array(bid_quantities), 
                np.array(ask_prices), np.array(ask_quantities))

def plot_order_book(order_book: LimitOrderBook, max_levels: int = 20):
    """
    Visualize the order book
    
    Args:
        order_book: LimitOrderBook instance
        max_levels: Maximum number of levels to show
    """
    snapshot = order_book.get_order_book_snapshot(levels=max_levels)
    
    if not snapshot['bids'] and not snapshot['asks']:
        print("Order book is empty")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Bid side
    if snapshot['bids']:
        bid_prices = [level['price'] for level in snapshot['bids']]
        bid_quantities = [level['quantity'] for level in snapshot['bids']]
        
        ax1.barh(bid_prices, bid_quantities, color='green', alpha=0.7, label='Bids')
        ax1.set_xlabel('Quantity')
        ax1.set_ylabel('Price')
        ax1.set_title('Bid Side')
        ax1.grid(True, alpha=0.3)
    
    # Ask side
    if snapshot['asks']:
        ask_prices = [level['price'] for level in snapshot['asks']]
        ask_quantities = [level['quantity'] for level in snapshot['asks']]
        
        ax2.barh(ask_prices, ask_quantities, color='red', alpha=0.7, label='Asks')
        ax2.set_xlabel('Quantity')
        ax2.set_ylabel('Price')
        ax2.set_title('Ask Side')
        ax2.grid(True, alpha=0.3)
    
    # Add market data text
    info_text = f"""
    Best Bid: {snapshot['best_bid']:.2f if snapshot['best_bid'] else 'N/A'}
    Best Ask: {snapshot['best_ask']:.2f if snapshot['best_ask'] else 'N/A'}
    Mid Price: {snapshot['mid_price']:.2f if snapshot['mid_price'] else 'N/A'}
    Spread: {snapshot['spread']:.2f if snapshot['spread'] else 'N/A'}
    Last Price: {snapshot['last_price']:.2f if snapshot['last_price'] else 'N/A'}
    Total Volume: {snapshot['total_volume']:.0f}
    Trade Count: {snapshot['trade_count']}
    """
    
    fig.suptitle('Limit Order Book Visualization')
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("=== Limit Order Book Example ===")
    
    # Create order book
    lob = LimitOrderBook(tick_size=0.01)
    
    # Add some orders
    orders_to_add = [
        Order("buy1", OrderSide.BUY, OrderType.LIMIT, 100, 99.50, trader_id="trader1"),
        Order("buy2", OrderSide.BUY, OrderType.LIMIT, 200, 99.40, trader_id="trader2"),
        Order("buy3", OrderSide.BUY, OrderType.LIMIT, 150, 99.45, trader_id="trader3"),
        Order("sell1", OrderSide.SELL, OrderType.LIMIT, 100, 100.50, trader_id="trader4"),
        Order("sell2", OrderSide.SELL, OrderType.LIMIT, 250, 100.60, trader_id="trader5"),
        Order("sell3", OrderSide.SELL, OrderType.LIMIT, 180, 100.55, trader_id="trader6"),
    ]
    
    print("Adding initial orders...")
    for order in orders_to_add:
        trades = lob.add_order(order)
        if trades:
            print(f"  Order {order.order_id} generated {len(trades)} trades")
        else:
            print(f"  Order {order.order_id} added to book")
    
    # Show order book
    snapshot = lob.get_order_book_snapshot()
    print(f"\nOrder Book State:")
    print(f"  Best Bid: {snapshot['best_bid']}")
    print(f"  Best Ask: {snapshot['best_ask']}")
    print(f"  Spread: {snapshot['spread']}")
    print(f"  Mid Price: {snapshot['mid_price']}")
    
    # Add market order that will execute
    print(f"\nAdding market buy order for 120 shares...")
    market_order = Order("market1", OrderSide.BUY, OrderType.MARKET, 120, trader_id="aggressive_trader")
    trades = lob.add_order(market_order)
    
    print(f"Market order generated {len(trades)} trades:")
    for trade in trades:
        print(f"  Trade: {trade.quantity} @ {trade.price}")
    
    # Show updated order book
    snapshot = lob.get_order_book_snapshot()
    print(f"\nUpdated Order Book State:")
    print(f"  Best Bid: {snapshot['best_bid']}")
    print(f"  Best Ask: {snapshot['best_ask']}")
    print(f"  Spread: {snapshot['spread']}")
    print(f"  Last Price: {snapshot['last_price']}")
    print(f"  Total Volume: {snapshot['total_volume']}")
    
    # Visualize order book
    plot_order_book(lob)