import pandas as pd
from sklearn.model_selection import train_test_split
from paths import RAW_DIR, DATA_DIR

# NOTE: this FYP-1 split is random and imputes with full-dataset medians, so it
# leaks future information. Kept as-is to reproduce FYP-1 (baseline B6 / E2);
# Vaultic research uses the time-based split defined in src/vaultic/data/.

# 1. Load the data
df = pd.read_csv(RAW_DIR / "train_transaction.csv")

# 2. Pick a smaller, solid set of useful columns for our BASELINE model
#    (You can add more later — this is enough for a first working model)
selected_columns = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3",
    "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "C1", "C2", "C13", "isFraud"
]
df = df[selected_columns]

# 3. Check how much is missing in each column
print("Missing values per column:")
print(df.isnull().sum())

# 4. Fill missing NUMBER columns with the median (a safe default)
number_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in number_cols:
    df[col] = df[col].fillna(df[col].median())

# 5. Fill missing TEXT columns with "unknown"
text_cols = df.select_dtypes(include=["object"]).columns
for col in text_cols:
    df[col] = df[col].fillna("unknown")

# 6. Convert text columns into numbers (encoding)
df = pd.get_dummies(df, columns=text_cols)

# 7. Split into features (X) and target (y)
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# 8. Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# 9. Save these cleaned pieces so we can reuse them in the next step
DATA_DIR.mkdir(exist_ok=True)
X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

print(f"\n✅ Preprocessing done! Cleaned files saved in {DATA_DIR}")