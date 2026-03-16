import urllib.request
import pandas as pd

url = "https://raw.githubusercontent.com/sidhantagar/Kaggle-House-Prices/master/train.csv"
print("Downloading train.csv from GitHub...")
urllib.request.urlretrieve(url, "train.csv")

df = pd.read_csv("train.csv")
print(f"Shape: {df.shape}")
print(f"Columns (first 5): {df.columns[:5].tolist()}")
print(f"Has SalePrice: {'SalePrice' in df.columns}")
print(f"Has Id: {'Id' in df.columns}")
print("Download complete!")
