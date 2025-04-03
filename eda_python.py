# This script performs some basic exploratory data analysis

########################## Import utilities
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
import gower  # For Gower distance calculation

# Import utility functions from setup_script.py
from setup_script import compute_area, make_transparent, summary_stats, rf_var_selection

########################## Load data
# In Python we'd use a different approach to load data
# Assuming the data is in a pickle or CSV format
# Replace with actual path to your data

# Set working directory to script location (similar to setwd in R)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Uncomment if needed

# Load data (replace with your actual loading method)
# In R, the load function loads an .Rda file which contains R objects
# In Python, we typically use pickle or CSV files
try:
    app_data = pd.read_pickle("app_data_clean.pkl")  # If saved as pickle
except FileNotFoundError:
    try:
        app_data = pd.read_csv("app_data.csv")  # If saved as CSV
    except FileNotFoundError:
        print("Data file not found. Please ensure data is available.")

########################## Summary statistics
print(app_data.info())  # Similar to str(app.data)
print(app_data.describe())  # Similar to summary(app.data)

########################## Summary statistics for the paper
# Full set of variables used in analysis
vars_incl = ["DiagnosisByCriteria", "TreatmentGroupBinar", "AppendicitisComplications",
            "Age", "BMI", "Sex", "Height", "Weight", 
            "AlvaradoScore", "PediatricAppendicitisScore",
            "AppendixOnSono", "AppendixDiameter", "MigratoryPain", "LowerAbdominalPainRight", 
            "ReboundTenderness", "CoughingPain", "PsoasSign",
            "Nausea", "AppetiteLoss", "BodyTemp", "WBCCount", "NeutrophilPerc", 
            "KetonesInUrine", "ErythrocytesInUrine", "WBCInUrine", "CRPEntry",
            "Dysuria", "Stool", "Peritonitis", "FreeFluids", 
            "AppendixWallLayers", "Kokarde", "TissuePerfusion",
            "SurroundingTissueReaction", "PathLymphNodes",
            "MesentricLymphadenitis", "BowelWallThick", "Ileus", "FecalImpaction",
            "Meteorism", "Enteritis"]

# Filter data where DiagnosisByCriteria is not NA
app_data_full = app_data.loc[~app_data["DiagnosisByCriteria"].isna(), vars_incl]

# Disease analysis
app_data_dis = app_data_full.drop(columns=["TreatmentGroupBinar", "AppendicitisComplications"])
print("Summary for appendicitis group:")
print(app_data_dis[app_data_dis["DiagnosisByCriteria"] == "appendicitis"].describe())
print("\nSummary for non-appendicitis group:")
print(app_data_dis[app_data_dis["DiagnosisByCriteria"] == "noAppendicitis"].describe())

# Calculate statistical summary using our imported function
s_dis = summary_stats(
    data=app_data_dis.drop(columns=["DiagnosisByCriteria"]), 
    y=app_data_dis["DiagnosisByCriteria"],
    seed=1799
)
print("P-values for disease comparison:")
print(s_dis['pvals'])

# Treatment analysis
app_data_trt = app_data_full.drop(columns=["DiagnosisByCriteria", "AppendicitisComplications"])
print("\nSummary for surgical treatment group:")
print(app_data_trt[app_data_trt["TreatmentGroupBinar"] == "surgical"].describe())
print("\nSummary for conservative treatment group:")
print(app_data_trt[app_data_trt["TreatmentGroupBinar"] == "conservative"].describe())

# Calculate statistical summary
s_trt = summary_stats(
    data=app_data_trt.drop(columns=["TreatmentGroupBinar"]), 
    y=app_data_trt["TreatmentGroupBinar"],
    seed=1799
)
print("P-values for treatment comparison:")
print(s_trt['pvals'])

# Complications analysis
app_data_comp = app_data_full.drop(columns=["DiagnosisByCriteria", "TreatmentGroupBinar"])
print("\nSummary for patients with complications:")
print(app_data_comp[app_data_comp["AppendicitisComplications"] == "yes"].describe())
print("\nSummary for patients without complications:")
print(app_data_comp[app_data_comp["AppendicitisComplications"] == "no"].describe())

# Calculate statistical summary
s_comp = summary_stats(
    data=app_data_comp.drop(columns=["AppendicitisComplications"]), 
    y=app_data_comp["AppendicitisComplications"],
    seed=1799
)
print("P-values for complications comparison:")
print(s_comp['pvals'])

########################## Percentages of missing values
# Calculate percentage of missing values for each column
na_percs = app_data.isna().mean() * 100

# Plot missing values percentages
plt.figure(figsize=(15, 8))
ax = na_percs.plot(kind='bar')
plt.xticks(rotation=90)
plt.ylim(0, 100)
plt.ylabel("Percentage of Missing Values, %")
plt.axhline(y=100, linestyle='--', color='red')
plt.tight_layout()
plt.savefig("missing_values.png")
plt.close()

########################## Response variables

# Treatment group: conservative therapy vs surgical
plt.figure(figsize=(10, 6))
app_data["TreatmentGroupBinar"].value_counts().plot(kind='bar')
plt.title("Treatment Groups")
plt.tight_layout()
plt.savefig("treatment_groups.png")
plt.close()

# Create a binarized version of treatment group
# Assuming TreatmentGroup is already in app_data
app_data["TreatmentGroupBinar"] = app_data["TreatmentGroupBinar"].astype(str)
app_data.loc[app_data["TreatmentGroupBinar"].isin(["primarySurgical", "secondarySurgical"]), "TreatmentGroupBinar"] = "surgical"
app_data["TreatmentGroupBinar"] = app_data["TreatmentGroupBinar"].astype('category')

plt.figure(figsize=(8, 6))
app_data["TreatmentGroupBinar"].value_counts().plot(kind='bar')
plt.title("Treatment (Binarized)")
plt.xlabel("Treatment")
plt.tight_layout()
plt.savefig("treatment_binarized.png")
plt.close()

print(app_data["TreatmentGroupBinar"].value_counts())

# Appendicitis classification: with or without complications?
plt.figure(figsize=(8, 6))
app_data["AppendicitisComplications"].value_counts().plot(kind='bar')
plt.title("Appendicitis Complications")
plt.xlabel("Complications?")
plt.tight_layout()
plt.savefig("complications.png")
plt.close()

print(app_data["AppendicitisComplications"].value_counts())

# Diagnosis: appendicitis vs. no appendicitis?
# Convert to human-readable format similar to R's factor
diagnosis_counts = app_data["DiagnosisByCriteria"].map({
    1: "appendicitis", 
    2: "no appendicitis",
    "appendicitis": "appendicitis", 
    "noAppendicitis": "no appendicitis"
}).value_counts()

plt.figure(figsize=(8, 6))
diagnosis_counts.plot(kind='bar')
plt.title("Diagnosis (According to Criteria)")
plt.xlabel("Diagnosis")
plt.tight_layout()
plt.savefig("diagnosis.png")
plt.close()

print(app_data["DiagnosisByCriteria"].value_counts())

########################## Dimensionality reduction
# Include only relevant variables for dimensionality reduction
vars_incl = ["Age", "BMI", "Sex", "Height", "Weight", 
            "AlvaradoScore", "PediatricAppendicitisScore", 
            "AppendixOnSono", "AppendixDiameter", "MigratoryPain", "LowerAbdominalPainRight", 
            "ReboundTenderness", "CoughingPain", "PsoasSign",
            "Nausea", "AppetiteLoss", "BodyTemp", "WBCCount", "NeutrophilPerc", 
            "KetonesInUrine", "ErythrocytesInUrine", "WBCInUrine", "CRPEntry",
            "Dysuria", "Stool", "Peritonitis", "FreeFluids", "FreeFluidsLoc", 
            "AppendixWallLayers", "Kokarde", "TissuePerfusion",
            "SurroundingTissueReaction", "PerityphliticAbscess", "PathLymphNodes",
            "MesentricLymphadenitis", "BowelWallThick", "Ileus", "FecalImpaction",
            "Meteorism", "Enteritis", "AppendicitisComplications"]

# Filter data and handle missing values
app_data_imputed = app_data[~app_data["DiagnosisByCriteria"].isna()].copy()

# Impute missing values using KNN
# First, prepare data for imputation
numeric_cols = app_data_imputed.select_dtypes(include=['number']).columns
categorical_cols = app_data_imputed.select_dtypes(exclude=['number']).columns

# Handle numeric columns with KNN imputation
imputer = KNNImputer(n_neighbors=5)
if len(numeric_cols) > 0:
    app_data_imputed[numeric_cols] = imputer.fit_transform(app_data_imputed[numeric_cols])

# Handle categorical columns with mode imputation
for col in categorical_cols:
    mode_val = app_data_imputed[col].mode()[0]
    app_data_imputed[col].fillna(mode_val, inplace=True)

# Compute Gower's distances (equivalent to daisy in R)
# First encode categorical variables
data_for_gower = app_data_imputed.copy()
for col in data_for_gower.select_dtypes(include=['category', 'object']).columns:
    data_for_gower[col] = LabelEncoder().fit_transform(data_for_gower[col])

# Calculate Gower distance matrix
gower_dist_matrix = gower.gower_matrix(data_for_gower)

# Perform MDS (equivalent to cmdscale in R)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1799)
mds_fit = mds.fit_transform(gower_dist_matrix)

# Plot by diagnosis
plt.figure(figsize=(10, 8))
diagnosis_values = app_data["DiagnosisByCriteria"][~app_data["DiagnosisByCriteria"].isna()]

# Create a mask for each diagnosis group
mask_appendicitis = np.array(diagnosis_values == "appendicitis")
mask_no_appendicitis = np.array(diagnosis_values == "noAppendicitis")

# Plot points by diagnosis
plt.scatter(mds_fit[mask_appendicitis, 0], mds_fit[mask_appendicitis, 1], 
           color='orange', s=30, alpha=0.75, label='Appendicitis')
plt.scatter(mds_fit[mask_no_appendicitis, 0], mds_fit[mask_no_appendicitis, 1], 
           color='blue', s=30, alpha=0.75, label='No appendicitis')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Diagnosis according to criteria:')
plt.title('MDS Plot by Diagnosis')
plt.tight_layout()
plt.savefig("mds_diagnosis.png")
plt.close()

# Plot by treatment among appendicitis patients
plt.figure(figsize=(10, 8))
treatment_values = app_data["TreatmentGroupBinar"][~app_data["DiagnosisByCriteria"].isna()]

# Create a mask for each treatment group
mask_conservative = np.array(treatment_values == "conservative")
mask_surgical = np.array(treatment_values == "surgical")

# Plot points by treatment
plt.scatter(mds_fit[mask_conservative, 0], mds_fit[mask_conservative, 1], 
           color='orange', s=30, alpha=0.75, label='Conservative')
plt.scatter(mds_fit[mask_surgical, 0], mds_fit[mask_surgical, 1], 
           color='blue', s=30, alpha=0.75, label='Surgical')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Treatment:')
plt.title('MDS Plot by Treatment')
plt.tight_layout()
plt.savefig("mds_treatment.png")
plt.close()

# Plot by complications among appendicitis patients
plt.figure(figsize=(10, 8))
complication_values = app_data["AppendicitisComplications"][~app_data["DiagnosisByCriteria"].isna()]

# Create a mask for each complication group
mask_no_complications = np.array(complication_values == "no")
mask_yes_complications = np.array(complication_values == "yes")

# Plot points by complications
plt.scatter(mds_fit[mask_no_complications, 0], mds_fit[mask_no_complications, 1], 
           color='orange', s=30, alpha=0.75, label='No')
plt.scatter(mds_fit[mask_yes_complications, 0], mds_fit[mask_yes_complications, 1], 
           color='blue', s=30, alpha=0.75, label='Yes')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Complicated appendicitis:')
plt.title('MDS Plot by Complications')
plt.tight_layout()
plt.savefig("mds_complications.png")
plt.close()

print("Analysis complete. Plots saved as PNG files.")
