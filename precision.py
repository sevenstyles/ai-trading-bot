from decimal import Decimal, ROUND_DOWN

class PrecisionHandler:
    def __init__(self, symbol_info):
        self.symbol_info = symbol_info
        self.quantity_precision = self._get_quantity_precision()
        self.price_precision = self._get_price_precision()
        self.tick_size = self._get_tick_size()
        self.step_size = self._get_step_size()
        self.min_notional = self._get_min_notional()

    def _get_filter(self, filter_type):
        return next(
            (f for f in self.symbol_info['filters'] if f['filterType'] == filter_type),
            None
        )

    def _get_quantity_precision(self):
        lot_size = self._get_filter('LOT_SIZE')
        return str(lot_size['stepSize']).rstrip('0').find('.')

    def _get_price_precision(self):
        price_filter = self._get_filter('PRICE_FILTER')
        return str(price_filter['tickSize']).rstrip('0').find('.')

    def _get_tick_size(self):
        price_filter = self._get_filter('PRICE_FILTER')
        return float(price_filter['tickSize'])

    def _get_step_size(self):
        lot_size = self._get_filter('LOT_SIZE')
        return float(lot_size['stepSize'])

    def _get_min_notional(self):
        notional = self._get_filter('MIN_NOTIONAL')
        return float(notional['notional']) if notional else 100.0

    def normalize_quantity(self, quantity):
        """Normalize quantity according to symbol's quantity precision"""
        step_size = Decimal(str(self.step_size))
        quantity = Decimal(str(quantity))
        normalized = float(
            (quantity / step_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_size
        )
        return normalized

    def normalize_price(self, price):
        """Normalize price according to symbol's price precision"""
        tick_size = Decimal(str(self.tick_size))
        price = Decimal(str(price))
        normalized = float(
            (price / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
        )
        return normalized

    def check_notional(self, quantity, price):
        """Check if order meets minimum notional value requirement"""
        notional = quantity * price
        return notional >= self.min_notional 