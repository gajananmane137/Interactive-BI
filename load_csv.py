"""Load sales_data.csv into SQLite so the BI agent can query it.
Usage:  python load_csv_to_sqlite.py
Then in bi_agent.py set:  DB_PATH = "sales.db"
(and skip the Chinook download by commenting out ensure_db())
"""
import sqlite3
import pandas as pd

df = pd.read_csv("sales_data.csv", parse_dates=["order_date"])
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
with sqlite3.connect("sales.db") as con:
    df.to_sql("sales", con, if_exists="replace", index=False)
print(f"Loaded {len(df)} rows into sales.db (table: sales)")