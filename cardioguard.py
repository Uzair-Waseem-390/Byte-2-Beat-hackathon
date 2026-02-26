import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # Fix backend issue

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# load dataset 

df = pd.read_csv("C:\\AI Agent\\hackathon\\cardiac_failure_processed.csv")

df.drop(columns=['Unnamed: 0', 'id'], inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# feature target 

y = df['cardio']
X = df.drop('cardio', axis=1)

# train test split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# models and pipelines 

models = {

    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": Pipeline([
        ('model', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ('model', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ))
    ])
}

# cross validation AUC 

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_auc = 0
best_model = None
best_name = ""

print("\nModel Comparison (Cross-Validation AUC):\n")

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_auc = scores.mean()
    print(f"{name}: {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model
        best_name = name

print("\nBest Model:", best_name)
print("Best CV AUC:", best_auc)

# train best model 

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# evaluation 

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC curve 

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')  # diagonal line
plt.title(f"ROC Curve - {best_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig("roc_curve.png")   # Save image
plt.show()
plt.close()

print("Test AUC Score:", roc_auc)

# feature importance 

if best_name != "Logistic Regression":
    model = best_model.named_steps['model']
    importance = model.feature_importances_
else:
  model = best_model.named_steps['model']
    importance = model.coef_[0]

# Sort importance
indices = np.argsort(importance)

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Feature Importance")
plt.xlabel("Importance Value")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig("feature_importance.png")   # Save image
plt.show()
plt.close()

print("\nGraphs saved as:")
print(" - roc_curve.png")
print(" - feature_importance.png")
