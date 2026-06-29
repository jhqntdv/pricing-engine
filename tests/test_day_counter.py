import pytest
from datetime import date, datetime
from unittest.mock import patch
import kernel
from kernel.tools import CalendarConvention
from utils.day_counter import DayCounter

def test_day_counter_default_start_date_evaluates_dynamically():
    """
    Test that the default start_date is evaluated at call-time (dynamically) 
    and not at import-time.
    """
    counter = DayCounter(CalendarConvention.ACT_360.value)
    
    # Mock datetime to return 2026-01-01 on first call
    with patch("utils.day_counter.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        
        end_date = date(2026, 12, 31)
        
        # Call without start_date, it should use 2026-01-01
        frac1 = counter.get_year_fraction(end_date=end_date)
        
        # Mock datetime to return 2026-06-01 on second call
        mock_datetime.now.return_value = datetime(2026, 6, 1)
        
        # Call again without start_date, it should use 2026-06-01
        frac2 = counter.get_year_fraction(end_date=end_date)
        
        # If the default was evaluated at import time, frac1 would equal frac2
        assert frac1 != frac2, "Default start_date is not evaluated dynamically!"

def test_day_counter_explicit_start_date():
    """
    Test that providing an explicit start_date ignores the default now() behavior.
    """
    counter = DayCounter(CalendarConvention.ACT_360.value)
    
    with patch("utils.day_counter.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 6, 1) # Should be ignored
        
        start = date(2025, 1, 1)
        end = date(2026, 1, 1)
        
        frac = counter.get_year_fraction(start_date=start, end_date=end)
        
        assert round(frac, 4) == 1.0139  # 365 / 360 = 1.01388...
