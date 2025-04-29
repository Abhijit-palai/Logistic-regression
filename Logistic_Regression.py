import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

# Load your data
file_path = 'breast-cancer.csv'  # Replace with the actual path to your file
data = pd.read_csv(file_path)

# Drop the 'Unnamed: 32' column if it exists (contains no data)
if 'Unnamed: 32' in data.columns:
    data = data.drop('Unnamed: 32', axis=1)

# Separate features (X) and target (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Missing Values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Fit Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

#  Evaluate the Model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Malignant)

# Calculate Evaluation Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)
print(f"Precision: {classification_rep['weighted avg']['precision']:.2f}")
print(f"Recall: {classification_rep['weighted avg']['recall']:.2f}")
print(f"F1-Score: {classification_rep['weighted avg']['f1-score']:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#  Tune the Threshold (Optional)
threshold = 0.3  # Example threshold, you can experiment with different values
y_pred_adjusted = (y_prob > threshold).astype(int)

# Re-evaluate metrics with adjusted threshold
precision_adjusted = classification_report(y_test, y_pred_adjusted, output_dict=True)['weighted avg']['precision']
recall_adjusted = classification_report(y_test, y_pred_adjusted, output_dict=True)['weighted avg']['recall']
f1_score_adjusted = classification_report(y_test, y_pred_adjusted, output_dict=True)['weighted avg']['f1-score']

print("\nMetrics with Adjusted Threshold (0.3):")
print(f"Adjusted Precision: {precision_adjusted:.2f}")
print(f"Adjusted Recall: {recall_adjusted:.2f}")
print(f"Adjusted F1-Score: {f1_score_adjusted:.2f}")

#  Visualize the Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid_values)
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.grid(True)
plt.show()