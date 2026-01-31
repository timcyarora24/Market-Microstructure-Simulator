# Minimal Limited Order Book (LOB) 
# If I execute a large order into an existing book, how much does it cost me under Poisson vs Hawkes arrivals
# My LOB starts full and only consumes liquidity

class LimitOrderBook:
    def __init__(self, asks):
        """
        Initialize the Limit Order Book with a list of ask orders.
        
        Parameters:
        asks (list of tuples): Each tuple contains (price, quantity).
        """
        # Sort asks by price in ascending order
        self.asks = sorted(asks, key=lambda x: x[0])
    
    def execute_market_buy(self, qty):
        """
        Execute a market buy order of a given quantity.
        
        Parameters:
        qty (int): The quantity to buy.
        
        Returns:
        float: The total cost of the executed buy order.
        """
        total_cost = 0.0
        remaining_qty = qty
        
        for i in range(len(self.asks)):
            price, available_qty = self.asks[i]
            if remaining_qty <= 0:
                break
            
            if available_qty <= remaining_qty:
                # Buy all available quantity at this price
                total_cost += price * available_qty
                remaining_qty -= available_qty
                self.asks[i] = (price, 0)  # All quantity consumed
            else:
                # Buy only the remaining quantity needed
                total_cost += price * remaining_qty
                self.asks[i] = (price, available_qty - remaining_qty)
                remaining_qty = 0
        
        if remaining_qty > 0:
            raise ValueError("Not enough liquidity in the order book to fulfill the order.")
        
        return total_cost