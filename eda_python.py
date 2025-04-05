# This script performs some basic exploratory data analysis

########################## Import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
import gower

# Import utility functions from setup_script.py
from setup_script import summary_stats

########################## Load data

app_data = pd.read_csv("app_data.csv")
    
########################## Summary statistics

print(app_data.info())  
print(app_data.describe())  

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
app_data_full = app_data.dropna(subset=['DiagnosisByCriteria'])[vars_incl]

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

# Combine p-values from all analyses into a DataFrame

print("P-values for targets:")
p_values_df = pd.DataFrame({
    'diagnosis': s_dis['pvals'],
    'treatment': s_trt['pvals'],
    'complications': s_comp['pvals']
}).round(4)

print("\nP-values for all comparisons:")
print(p_values_df)

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
plt.savefig("./Eda_plots/missing_values.png")
plt.close()

########################## Target Variables

# Treatment group: conservative therapy vs surgical

plt.figure(figsize=(8, 6))
app_data["TreatmentGroupBinar"].value_counts().plot(kind='bar')
plt.title("Treatment (Binarized)")
plt.xlabel("Treatment")
plt.tight_layout()
plt.savefig("./Eda_plots/treatment_binarized.png")
plt.close()

print(app_data["TreatmentGroupBinar"].value_counts())

# Appendicitis classification: with or without complications?
plt.figure(figsize=(8, 6))
app_data["AppendicitisComplications"].value_counts().plot(kind='bar')
plt.title("Appendicitis Complications")
plt.xlabel("Complications?")
plt.tight_layout()
plt.savefig("./Eda_plots/complications.png")
plt.close()

print(app_data["AppendicitisComplications"].value_counts())

# Diagnosis: appendicitis vs. no appendicitis?

plt.figure(figsize=(8, 6))
app_data["DiagnosisByCriteria"].value_counts().plot(kind='bar')
plt.title("Diagnosis")
plt.xlabel("Diagnosis")
plt.tight_layout()
plt.savefig("./Eda_plots/diagnosis.png")
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

# Filter data and handle missing values using kNN
app_data_imputed = app_data[~app_data["DiagnosisByCriteria"].isna()].copy()

numeric_cols = app_data_imputed.select_dtypes(include=['number']).columns
categorical_cols = app_data_imputed.select_dtypes(exclude=['number']).columns

# For Numerical cols
imputer = KNNImputer(n_neighbors=5)
app_data_imputed[numeric_cols] = imputer.fit_transform(app_data_imputed[numeric_cols])

# For categorical col
# Categorical -> Numerical using Label Encoder
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    non_null_mask = ~app_data_imputed[col].isna()
    le_dict[col] = le
    app_data_imputed.loc[non_null_mask, col] = le.fit_transform(app_data_imputed.loc[non_null_mask, col])
    app_data_imputed[col] = app_data_imputed[col].astype(float)

# kNN on Numerical data
imputer = KNNImputer(n_neighbors=5)
app_data_imputed[categorical_cols] = imputer.fit_transform(app_data_imputed[categorical_cols])

# Numerical -> Categorical
for col in categorical_cols:
    app_data_imputed[col] = np.round(app_data_imputed[col]).astype(int)
    app_data_imputed[col] = le_dict[col].inverse_transform(app_data_imputed[col])

# Compute Gower's distance
# First encode categorical variables
data_for_gower = app_data_imputed.copy()
for col in data_for_gower.select_dtypes(exclude=['number']).columns:
    data_for_gower[col] = LabelEncoder().fit_transform(data_for_gower[col])

# Calculate Gower distance matrix
gower_dist_matrix = gower.gower_matrix(data_for_gower)

# Perform MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1799)
mds_fit = mds.fit_transform(gower_dist_matrix)

# Plot by diagnosis
plt.figure(figsize=(10, 8))
diagnosis_values = app_data_imputed["DiagnosisByCriteria"]

# Create a mask for each diagnosis group
mask_appendicitis = np.array(diagnosis_values == "appendicitis")
mask_no_appendicitis = np.array(diagnosis_values == "noAppendicitis")

plt.scatter(mds_fit[mask_appendicitis, 0], mds_fit[mask_appendicitis, 1], 
           color='orange', s=30, alpha=0.75, label='Appendicitis')
plt.scatter(mds_fit[mask_no_appendicitis, 0], mds_fit[mask_no_appendicitis, 1], 
           color='blue', s=30, alpha=0.75, label='No appendicitis')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Diagnosis according to criteria:')
plt.title('MDS Plot by Diagnosis')
plt.tight_layout()
plt.savefig("./Eda_plots/mds_diagnosis.png")
plt.close()

# Plot by treatment among appendicitis patients
plt.figure(figsize=(10, 8))
treatment_values = app_data_imputed["TreatmentGroupBinar"]

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
plt.savefig("./Eda_plots/mds_treatment.png")
plt.close()

# Plot by complications among appendicitis patients
plt.figure(figsize=(10, 8))
complication_values = app_data_imputed["AppendicitisComplications"]

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
plt.savefig("./Eda_plots/mds_complications.png")
plt.close()

print("Analysis complete. Plots saved as PNG files.")
