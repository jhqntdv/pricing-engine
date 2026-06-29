import pytest
from kernel.tools import ObservationFrequency
from kernel.products.structured_products.autocall_products import Phoenix, Eagle
from kernel.products.structured_products.participation_products import TwinWin, Airbag

def test_structured_products_instantiation():
    """
    Verify that all structured products can be successfully instantiated.
    This acts as a regression test to ensure that no subclass fails to implement
    the @abstractmethod `description()` defined in AbstractStructuredProduct.
    """
    try:
        # Phoenix
        p1 = Phoenix(
            maturity=5.0, 
            observation_frequency=ObservationFrequency.ANNUAL,
            capital_barrier=60.0, 
            autocall_barrier=100.0, 
            coupon_rate=5.0, 
            coupon_barrier=80.0
        )
        
        # Eagle
        p2 = Eagle(
            maturity=3.0,
            observation_frequency=ObservationFrequency.SEMIANNUAL,
            capital_barrier=70.0,
            autocall_barrier=105.0,
            coupon_rate=8.0
        )
        
        # TwinWin
        p3 = TwinWin(
            maturity=2.0,
            upper_barrier=120.0,
            lower_barrier=80.0,
            rebate=5.0,
            leverage=2.0
        )
        
        # Airbag
        p4 = Airbag(
            maturity=2.0,
            upper_barrier=120.0,
            lower_barrier=80.0,
            rebate=5.0,
            leverage=2.0
        )
        
        # Ensure description is callable and returns a string
        assert isinstance(p1.description(), str)
        assert isinstance(p2.description(), str)
        assert isinstance(p3.description(), str)
        assert isinstance(p4.description(), str)

    except TypeError as e:
        pytest.fail(f"Failed to instantiate structured product, possibly missing abstract method: {e}")
