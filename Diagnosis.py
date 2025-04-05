import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the data
data = pd.read_excel("app_data.xlsx")

# Define the target variable
target_column = "DiagnosisByCriteria"
data[target_column] = data[target_column].map({"appendicitis": 1, "noAppendicitis": 0})

# Define categorical features
categorical_cols = ["Sex", "AppendixOnSono", "MigratoryPain", "LowerAbdominalPainRight", "ReboundTenderness",
                    "CoughingPain", "PsoasSign", "Nausea", "AppetiteLoss", "Dysuria", "Stool", "Peritonitis",
                    "FreeFluids", "AppendixWallLayers", "Kokarde", "TissuePerfusion", "SurroundingTissueReaction",
                    "PathLymphNodes", "MesentricLymphadenitis", "BowelWallThick", "Ileus", "FecalImpaction",
                    "Meteorism", "Enteritis", "TreatmentGroupBinar", "AppendicitisComplications"]

# Handle missing values in categorical columns
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Convert all data to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Identify columns with all NaN values
nan_columns = X.columns[X.isna().all()]
print(f"Columns with all NaN values that will be dropped: {list(nan_columns)}")

# Drop columns with all NaN values
X = X.drop(columns=nan_columns)

# Impute remaining NaN values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
}

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    scores = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, y_pred_fold))

    cv_scores[name] = np.mean(scores)
    print(f"\n{name} - Average CV Accuracy: {cv_scores[name]:.4f}")
    print("=" * 40)

# Train models on full training set and evaluate
plt.figure(figsize=(10, 6))
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        print(f"\n{name} Results:")
        print(f"AUPR: {pr_auc:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 40)

# Plot ROC curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(10, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{name} (AP = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="best")
plt.show()

# Add after model training
def plot_feature_importance(model_name, model):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.show()

# Call for applicable models
plot_feature_importance("RandomForest", models["RandomForest"])

# Save models after training
for name, model in models.items():
    joblib.dump(model, f"{name}_model.pkl")
    
# Function to load models
def load_models():
    loaded_models = {}
    for name in ["RandomForest", "GradientBoostingClassifier", "KNN", "LogisticRegression"]:
        loaded_models[name] = joblib.load(f"{name}_model.pkl")
    return loaded_models

# Function to predict appendicitis for new patients
def predict_appendicitis(patient_data):
    # Create DataFrame from input
    patient_df = pd.DataFrame([patient_data])

    # Convert all values to strings for categorical columns
    for col in categorical_cols:
        if col in patient_df.columns:
            patient_df[col] = patient_df[col].astype(str)

    # Handle categorical variables
    for col in categorical_cols:
        if col in patient_df.columns and col in label_encoders:
            unique_classes = list(label_encoders[col].classes_)
            if patient_df[col].iloc[0] not in unique_classes:
                print(f"Warning: Unknown value '{patient_df[col].iloc[0]}' in column {col}. Mapping to default.")
                patient_df[col] = unique_classes[0]
            else:
                patient_df[col] = label_encoders[col].transform([patient_df[col].iloc[0]])[0]

    # Ensure all features are present and numeric
    for col in X.columns:
        if col not in patient_df:
            patient_df[col] = 0  # Default value for missing features
    patient_df = patient_df[X.columns].apply(pd.to_numeric, errors='coerce')

    # Impute any remaining NaN values
    patient_imputed = imputer.transform(patient_df)
    patient_df = pd.DataFrame(patient_imputed, columns=X.columns)

    # Scale the features
    patient_scaled = scaler.transform(patient_df)
    patient_scaled = np.nan_to_num(patient_scaled)  # Final check

    predictions = {}
    for name, model in models.items():
        try:
            prediction = model.predict(patient_scaled)[0]
            predictions[name] = {
                'prediction': 'Appendicitis' if prediction == 1 else 'No Appendicitis',
                'confidence': float(model.predict_proba(patient_scaled)[0][1]) if hasattr(model, "predict_proba") else None
            }
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            predictions[name] = {
                'prediction': 'Error',
                'confidence': None
            }
    return predictions

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Appendicitis', 'Appendicitis'],
               yticklabels=['No Appendicitis', 'Appendicitis'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
    
# Add for each model after predictions
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}")

def get_patient_input():
    patient = {}
    patient["Age"] = float(input("Enter patient age: "))
    patient["Sex"] = input(f"Enter patient sex {list(label_encoders['Sex'].classes_)}: ")
    
    # Add more input prompts for other important features
    
    return patient

if __name__ == "__main__":
    # Print expected values for categorical columns
    print("Expected values for categorical columns:")
    for col in categorical_cols:
        if col in label_encoders:
            print(f"{col}: {list(label_encoders[col].classes_)}")

    # Example patient with corrected values
    example_patient = {
        "Age": 10,
        "Sex": list(label_encoders['Sex'].classes_)[0],
        "AppendixOnSono": list(label_encoders['AppendixOnSono'].classes_)[0],
        "MigratoryPain": list(label_encoders['MigratoryPain'].classes_)[0],
        "LowerAbdominalPainRight": list(label_encoders['LowerAbdominalPainRight'].classes_)[0]
    }

    # Add default values for any columns that were dropped due to all NaN values
    for col in nan_columns:
        example_patient[col] = 0

    print("\nExample patient data:", example_patient)

    predictions = predict_appendicitis(example_patient)
    print("\nPredictions for example patient:")
    for model_name, result in predictions.items():
        print(f"\n{model_name}:")
        print(f"Prediction: {result['prediction']}")
        if result['confidence'] is not None:
            print(f"Confidence: {result['confidence']:.2%}")


