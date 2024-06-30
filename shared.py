from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
data = pd.read_csv(app_dir / "test.csv")

def ratio(column, value):
    nd1 = data[data["stroke"] == "Yes"]
    nd1 = nd1[nd1[column] == value]
    nd2 = data[data[column] == value]
    # print(len(nd1), len(nd2))
    return round(len(nd1) / len(nd2) * 100, 2)