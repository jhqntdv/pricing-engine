import os
import pandas as pd

base_dir = r"c:\AppPy\structured-products-engine-only\data"

# 1. Convert underlying_data.xlsx to underlying_data.csv
ud_path = os.path.join(base_dir, "underlying_data.xlsx")
if os.path.exists(ud_path):
    df_ud = pd.read_excel(ud_path)
    df_ud.to_csv(os.path.join(base_dir, "underlying_data.csv"), index=False)
    print("Converted underlying_data.xlsx to .csv")

# 2. Convert RateCurve_temp.xlsx to RateCurve_temp.csv
rc_path = os.path.join(base_dir, "yield_curves", "RateCurve_temp.xlsx")
if os.path.exists(rc_path):
    df_rc = pd.read_excel(rc_path)
    df_rc.to_csv(os.path.join(base_dir, "yield_curves", "RateCurve_temp.csv"), index=False)
    print("Converted RateCurve_temp.xlsx to .csv")

# 3. Convert option_data_SPX.xlsx (wide) to options_SPX.csv (flat)
od_path = os.path.join(base_dir, "option_data", "option_data_SPX.xlsx")
if os.path.exists(od_path):
    df_od = pd.read_excel(od_path)
    # df_od is in wide format: Maturity | strike1 | strike2 ...
    # We want it in flat format to match AAPL: expiration, strike, implied_volatility
    # Let's melt it
    df_od_flat = pd.melt(df_od, id_vars=['Maturity'], var_name='strike', value_name='implied_volatility')
    # Rename Maturity to expiration to somewhat match standard format
    df_od_flat.rename(columns={'Maturity': 'expiration'}, inplace=True)
    # The original AAPL uses column separator ';'
    # Let's just use standard ',' for CSV
    out_path = os.path.join(base_dir, "option_data", "options_SPX.csv")
    df_od_flat.to_csv(out_path, index=False)
    print(f"Converted option_data_SPX.xlsx to flat {out_path}")
