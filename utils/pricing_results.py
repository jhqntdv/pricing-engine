from dataclasses import dataclass, field
from typing import Dict, Optional,List

@dataclass
class PricingResults:
    """Container for the output of a pricing engine calculation."""

    price: Optional[float] = None
    greeks: Dict[str, float] = field(default_factory=dict)
    coupon_callable: Optional[float] = None
    rate: Optional[float] = None
    std_dev: Optional[float] = None
    confidence_level: float = 0.95

    @property
    def lower_bound(self) -> Optional[float]:
        """Calculate the lower bound of the 95% confidence interval.

        Returns:
            The lower bound price if standard deviation is available, otherwise None.
        """
        if self.price is not None and self.std_dev is not None:
            return self.price - 1.96 * self.std_dev
        return None

    @property
    def upper_bound(self) -> Optional[float]:
        """Calculate the upper bound of the 95% confidence interval.

        Returns:
            The upper bound price if standard deviation is available, otherwise None.
        """
        if self.price is not None and self.std_dev is not None:
            return self.price + 1.96 * self.std_dev
        return None

    def set_greek(self, name: str, value: float):
        """Set a Greek value in the results dictionary.

        Args:
            name: The name of the Greek (e.g., 'delta', 'gamma').
            value: The computed value of the Greek.
        """
        self.greeks[name] = value

    def __str__(self):
        """Format the pricing results as a readable string.

        Returns:
            String representation of price, standard deviation, confidence interval, and Greeks.
        """
        bounds = (
            f"[{self.lower_bound:.4f}, {self.upper_bound:.4f}]"
            if self.lower_bound is not None else "N/A"
        )
        return (
            f"Price: {self.price if self.price is not None else 'N/A'}\n"
            f"Std Dev: {self.std_dev if self.std_dev is not None else 'N/A'} "
            f"({self.confidence_level:.0%} CI: {bounds})\n"
            f"Greeks: {self.greeks if self.greeks else 'N/A'}"
        )

    @staticmethod
    def get_aggregated_results(results: List["PricingResults"]) -> "PricingResults":
        """Aggregate multiple PricingResults into a single result.

        Sums the prices and the Greeks across all provided results.

        Args:
            results: A list of PricingResults objects to aggregate.

        Returns:
            A new PricingResults object containing the aggregated values.
        """
        aggregated = PricingResults()
        aggregated.price = sum(r.price for r in results if r.price is not None)

        # Aggregation of the Greeks 
        all_greeks = {}
        for r in results:
            for k, v in r.greeks.items():
                all_greeks[k] = all_greeks.get(k, 0) + v
        aggregated.greeks = all_greeks

        return aggregated

