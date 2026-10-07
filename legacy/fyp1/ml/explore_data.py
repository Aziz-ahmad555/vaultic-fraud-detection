import pandas as pd
from paths import RAW_DIR

# Load the transaction data
df = pd.read_csv(RAW_DIR / "train_transaction.csv")
# 1. How big is the dataset?
print("Shape (rows, columns):", df.shape)

# 2. What columns exist?
print("\nFirst 10 column names:", list(df.columns[:10]))

# 3. Peek at the first few rows
print("\nFirst 5 rows:")
print(df.head())

# 4. Most important number: how many are fraud vs not fraud?
print("\nFraud vs Not Fraud counts:")
print(df["isFraud"].value_counts())

# 5. What percentage is fraud?
fraud_percent = df["isFraud"].mean() * 100
print(f"\nPercentage of fraud transactions: {fraud_percent:.2f}%")