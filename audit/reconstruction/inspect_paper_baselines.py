from pathlib import Path
import pandas as pd

folder = Path(r"D:\SummerResearch\audit\reconstruction")
for name in ("April29_tidy.csv", "Mar23_tidy.csv", "April02_tidy.csv"):
    data = pd.read_csv(folder / name)
    rows = data[data.method.isin(["ctgan", "tvae"]) & (data.train == "synthetic")]
    print(name)
    print(rows[["dataset", "method", "metric", "value"]].to_string(index=False))
book = pd.read_excel(r"D:\SummerResearch\final_results.xlsx")
print("Workbook columns:", list(book.columns))
print(book.to_string(index=False))
