"""Run once to generate the reference dataset committed to the repo."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_data, save_data

df = load_data(inject_drift=False)
save_data(df, "data/reference_data.csv")
print("Reference data saved to data/reference_data.csv")
