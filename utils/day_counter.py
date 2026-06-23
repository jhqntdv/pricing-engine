from kernel.tools import CalendarConvention 
from datetime import date,datetime
class DayCounter:
    def __init__(self, convention: CalendarConvention):
        self.convention = convention

    def get_year_fraction(self, start_date: date = datetime.now(), end_date: date=None) -> float:
        if start_date > end_date:
            raise ValueError("start_date must be before end_date")

        delta_days = (end_date - start_date).days

        if self.convention == CalendarConvention.ACT_360.value:
            return delta_days / 360.0
        elif self.convention == CalendarConvention.ACT_365.value:
            return delta_days / 365.0
        elif self.convention == CalendarConvention.ACT_ACT.value:
            return self._actual_actual(start_date, end_date)
        elif self.convention == CalendarConvention.THIRTY_360.value:
            return self._thirty_360(start_date, end_date)
        else:
            raise NotImplementedError(f"Convention {self.convention} not implemented")

    def _actual_actual(self, start_date: date, end_date: date) -> float:
        year1 = start_date.year
        year2 = end_date.year

        if year1 == year2:
            days_in_year = 366 if self._is_leap_year(year1) else 365
            return (end_date - start_date).days / days_in_year
        else:
            # Fraction in first year
            start_of_next_year = date(year1 + 1, 1, 1)
            days_in_year1 = 366 if self._is_leap_year(year1) else 365
            first_fraction = (start_of_next_year - start_date).days / days_in_year1

            # Fraction in last year
            start_of_year2 = date(year2, 1, 1)
            days_in_year2 = 366 if self._is_leap_year(year2) else 365
            last_fraction = (end_date - start_of_year2).days / days_in_year2

            # Whole years in between
            full_years = year2 - year1 - 1

            return first_fraction + full_years + last_fraction

    def _thirty_360(self, start_date: date, end_date: date) -> float:
        d1 = min(start_date.day, 30)
        d2 = min(end_date.day, 30) if start_date.day == 30 or start_date.day == 31 else end_date.day
        days_360 = 360 * (end_date.year - start_date.year) + 30 * (end_date.month - start_date.month) + (d2 - d1)
        return days_360 / 360.0

    def _is_leap_year(self, year: int) -> bool:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)