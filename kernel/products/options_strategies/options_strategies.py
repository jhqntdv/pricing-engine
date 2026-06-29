import numpy as np
from .abstract_option_strategy import AbstractOptionStrategy
from ..options.vanilla_options import EuropeanCallOption, EuropeanPutOption

class Straddle(AbstractOptionStrategy):
    """Represents a straddle strategy.
    """
    def __init__(self, maturity: float, strike: float, 
                 position_call: bool = True, position_put: bool = True):
        """Initializes a straddle strategy.

        Args:
            maturity (float): Maturity of options.
            strike (float): Common strike price for the call and the put.
            position_call (bool): Position on the call (True for long, False for short).
            position_put (bool): Position on the put (True for long, False for short).
        """
        self.call = EuropeanCallOption(maturity, strike)
        self.put = EuropeanPutOption(maturity, strike)
        super().__init__([(self.call, position_call), (self.put, position_put)])

class Strangle(AbstractOptionStrategy):
    """Represents a strangle strategy.
    """
    def __init__(self, maturity: float, strike_put: float, strike_call: float, 
                 position_call: bool = True, position_put: bool = True):
        """Initializes a strangle policy.

        Args:
            maturity (float): Maturity of options.
            strike_call (float): Strike price for the call.
            strike_put (float): Strike price for the put.
            position_call (bool): Position on the call (True for long, False for short).
            position_put (bool): Position on the put (True for long, False for short).
        """
        call = EuropeanCallOption(maturity, strike_call)
        put = EuropeanPutOption(maturity, strike_put)
        super().__init__([(call, position_call), (put, position_put)])

class BullSpread(AbstractOptionStrategy):
    """Represents a bull spread strategy.
    """
    def __init__(self, maturity: float, strike_low: float, strike_high: float, 
                 position_low: bool = True, position_high: bool = False):
        """Initializes a bull spread strategy.

        Args:
            maturity (float): Maturity of options.
            strike_low (float): Strike price of the option purchased (long).
            strike_high (float): Strike price of the option sold (short).
            position_low (bool): Position on the option with strike_low (True for long, False for short).
            position_high (bool): Position on the option with strike_high (True for long, False for short).
        """
        self.call_low = EuropeanCallOption(maturity, strike_low)
        self.call_high = EuropeanCallOption(maturity, strike_high)
        super().__init__([(self.call_low, position_low), (self.call_high, position_high)])

class BearSpread(AbstractOptionStrategy):
    """Represents a bear spread strategy.
    """
    def __init__(self, maturity: float, strike_low: float, strike_high: float, 
                 position_low: bool = False, position_high: bool = True):
        """Initializes a bear spread strategy.

        Args:
            maturity (float): Maturity of options.
            strike_low (float): Strike price of the option purchased (long).
            strike_high (float): Strike price of the option sold (short).
            position_low (bool): Position on the option with strike_low (True for long, False for short).
            position_high (bool): Position on the option with strike_high (True for long, False for short).
        """
        self.put_low = EuropeanPutOption(maturity, strike_low)
        self.put_high = EuropeanPutOption(maturity, strike_high)
        super().__init__([(self.put_low, position_low), (self.put_high, position_high)])

class ButterflySpread(AbstractOptionStrategy):
    """Represents a butterfly spread strategy.
    """
    def __init__(self, maturity: float, strike_low: float, strike_mid: float, strike_high: float, 
                 position_low: bool = True, position_mid: bool = False, position_high: bool = True):
        """Initializes a butterfly spread strategy.

        Args:
            maturity (float): Maturity of options.
            strike_low (float): Strike price of the option purchased (long).
            strike_mid (float): Strike price of options sold (short).
            strike_high (float): Strike price of the option purchased (long).
            position_low (bool): Position on the option with strike_low (True for long, False for short).
            position_mid (bool): Position on options with strike_mid (True for long, False for short).
            position_high (bool): Position on the option with strike_high (True for long, False for short).
        """
        self.call_low = EuropeanCallOption(maturity, strike_low)
        self.call_mid1 = EuropeanCallOption(maturity, strike_mid)
        self.call_mid2 = EuropeanCallOption(maturity, strike_mid)
        self.call_high = EuropeanCallOption(maturity, strike_high)
        super().__init__([
            (self.call_low, position_low),
            (self.call_mid1, position_mid),
            (self.call_mid2, position_mid),
            (self.call_high, position_high)
        ])

class CondorSpread(AbstractOptionStrategy):
    """Represents a condor spread strategy.
    """
    def __init__(self, maturity: float, strike_low: float, strike_mid1: float, strike_mid2: float, strike_high: float, 
                 position_low: bool = True, position_mid1: bool = False, position_mid2: bool = False, position_high: bool = True):
        """Initializes a condor spread strategy.

        Args:
            maturity (float): Maturity of options.
            strike_low (float): Strike price of the option purchased (long).
            strike_mid1 (float): Strike price of the first option sold (short).
            strike_mid2 (float): Strike price of the second option sold (short).
            strike_high (float): Strike price of the option purchased (long).
            position_low (bool): Position on the option with strike_low (True for long, False for short).
            position_mid1 (bool): Position on the option with strike_mid1 (True for long, False for short).
            position_mid2 (bool): Position on the option with strike_mid2 (True for long, False for short).
            position_high (bool): Position on the option with strike_high (True for long, False for short).
        """
        self.call_low = EuropeanCallOption(maturity, strike_low)
        self.call_mid1 = EuropeanCallOption(maturity, strike_mid1)
        self.call_mid2 = EuropeanCallOption(maturity, strike_mid2)
        self.call_high = EuropeanCallOption(maturity, strike_high)
        super().__init__([
            (self.call_low, position_low),
            (self.call_mid1, position_mid1),
            (self.call_mid2, position_mid2),
            (self.call_high, position_high)
        ])

class CalendarSpread(AbstractOptionStrategy):
    """Represents a calendar spread strategy.
    """
    def __init__(self, strike: float, maturity_near: float, maturity_far: float, 
                 position_near: bool = False, position_far: bool = True):
        """Initializes a calendar spread strategy.

        Args:
            strike (float): Common strike price for options.
            maturity_near (float): Maturity of the option sold (short).
            maturity_far (float): Maturity of the purchased option (long).
            position_near (bool): Position on the option with near expiry (True for long, False for short).
            position_far (bool): Position on the option with distant expiry (True for long, False for short).
        """
        self.call_near = EuropeanCallOption(maturity_near, strike)
        self.call_far = EuropeanCallOption(maturity_far, strike)
        super().__init__([(self.call_near, position_near), (self.call_far, position_far)])


class Collar(AbstractOptionStrategy):
    """Represents a collar strategy.
    """
    def __init__(self, maturity: float, strike_put: float, strike_call: float, 
                 position_call: bool = False, position_put: bool = True):
        """Initializes a collar strategy.

        Args:
            maturity (float): Maturity of options.
            strike_call (float): Strike price of the sold call option (short).
            strike_put (float): Strike price of the put option purchased (long).
            position_call (bool): Position on the call option (True for long, False for short).
            position_put (bool): Position on the put option (True for long, False for short).
        """
        self.call = EuropeanCallOption(maturity, strike_call)
        self.put = EuropeanPutOption(maturity, strike_put)
        super().__init__([(self.call, position_call), (self.put, position_put)])

class Strip(AbstractOptionStrategy):
    """Represents a strip strategy.
    Consisting of 1 call and 2 puts (all long or all short).
    """
    def __init__(self, maturity: float, strike: float, 
                 position_call: bool = True, position_put: bool = True):
        """Initializes a strip strategy.

        Args:
            maturity (float): Maturity of options.
            strike (float): Common strike price of options.
            position_call (bool): Position on the call option (True for long, False for short).
            position_put (bool): Position on put options (True for long, False for short).
        """
        call = EuropeanCallOption(maturity, strike)
        put1 = EuropeanPutOption(maturity, strike)
        put2 = EuropeanPutOption(maturity, strike)
        super().__init__([
            (call, position_call),
            (put1, position_put),
            (put2, position_put)
        ])

class Strap(AbstractOptionStrategy):
    """Represents a strap strategy.
    Composed of 2 calls and 1 put (all long or all short).
    """
    def __init__(self, maturity: float, strike: float, 
                 position_call: bool = True, position_put: bool = True):
        """Initializes a strap strategy.

        Args:
            maturity (float): Maturity of options.
            strike (float): Common strike price of options.
            position_call (bool): Position on call options (True for long, False for short).
            position_put (bool): Position on the put option (True for long, False for short).
        """
        call1 = EuropeanCallOption(maturity, strike)
        call2 = EuropeanCallOption(maturity, strike)
        put = EuropeanPutOption(maturity, strike)
        super().__init__([
            (call1, position_call),
            (call2, position_call),
            (put, position_put)
        ])

