import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from paths import DATA_DIR, MODEL_DIR

# 1. Load the cleaned data from Step 3
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

print("Loaded data:")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)

# 2. Create the XGBoost model
#    scale_pos_weight helps the model pay more attention to the rare fraud cases
fraud_count = sum(y_train == 1)
normal_count = sum(y_train == 0)
scale_pos_weight = normal_count / fraud_count

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# 3. Train it
print("\nTraining model... (this may take 1-3 minutes)")
model.fit(X_train, y_train)
print("Training done!")

# 4. Make predictions on the TEST set (data the model never saw)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 5. Calculate the important scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n===== MODEL RESULTS =====")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFull Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model so we can reuse it later
model.save_model(MODEL_DIR / "xgboost_baseline.json")
print("\n✅ Model saved to models_store/xgboost_baseline.json")