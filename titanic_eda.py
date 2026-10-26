"""
=============================================================================
TITANIC DATASET: EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
Course: Exploratory Data Analysis Assignment
Topic: Passenger Survival Analysis on the Titanic
Libraries Used: Python, Pandas, NumPy, Matplotlib

Description:
This script performs a complete end-to-end exploratory data analysis on the
famous Titanic dataset. It covers data loading from a public URL, initial
inspection, missing value treatment, duplicate detection, grouping and
aggregation, data visualization using Matplotlib, and key analytical takeaways.
=============================================================================
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
LOCAL_CSV_PATH = os.path.join(DATA_DIR, "titanic.csv")

# Ensure required folders exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def print_section(title: str):
    """Utility function to print visual section dividers in the console."""
    border = "=" * 78
    print(f"\n{border}")
    print(f" {title.upper()}")
    print(f"{border}\n")


# ===========================================================================
# STEP 1: LOAD THE DATASET
# ===========================================================================
print_section("Step 1: Loading Dataset")

print(f"Attempting to load Titanic dataset from public URL:\n  {DATA_URL}")

try:
    # Attempt loading directly from URL
    df = pd.read_csv(DATA_URL)
    print(" Successfully loaded dataset directly from the web URL!")
    # Save a local cached copy for future offline access
    df.to_csv(LOCAL_CSV_PATH, index=False)
    print(f" Local backup copy saved to: {LOCAL_CSV_PATH}")
except Exception as e:
    print(f"[!] Warning: Could not download from web ({e}).")
    if os.path.exists(LOCAL_CSV_PATH):
        print(" Loading from local backup CSV instead...")
        df = pd.read_csv(LOCAL_CSV_PATH)
        print(" Successfully loaded dataset from local cache!")
    else:
        raise RuntimeError("Failed to load dataset from URL and no local copy exists.")


# ===========================================================================
# STEP 2: DEMONSTRATE BASIC DATAFRAME OPERATIONS
# ===========================================================================
print_section("Step 2: Basic DataFrame Operations")

# 2.1 DataFrame Shape
print(f"1. Dataset Shape (Rows, Columns): {df.shape}")
print(f"   -> Total Rows (Passengers): {df.shape[0]}")
print(f"   -> Total Columns (Features): {df.shape[1]}")

# 2.2 Column Names
print(f"\n2. Column Names ({len(df.columns)} total):")
for idx, col in enumerate(df.columns, start=1):
    print(f"   {idx:2d}. {col:<12} (Type: {df[col].dtype})")

# 2.3 First 5 Rows (head())
print("\n3. First 5 Rows of the Dataset (df.head()):")
print("-" * 78)
print(df.head().to_string())
print("-" * 78)

# 2.4 DataFrame Information (info())
print("\n4. DataFrame Info (df.info()):")
print("-" * 78)
df.info()
print("-" * 78)

# 2.5 Summary Statistics for Numerical Columns (describe())
print("\n5. Statistical Summary of Numerical Columns (df.describe()):")
print("-" * 78)
print(df.describe().to_string())
print("-" * 78)


# ===========================================================================
# STEP 3: IDENTIFY AND HANDLE MISSING VALUES
# ===========================================================================
print_section("Step 3: Handling Missing Values")

print("Checking for missing (NaN / Null) values in each column:")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Count": missing_counts,
    "Percentage (%)": missing_percent.round(2)
})
print(missing_df[missing_df["Missing Count"] > 0])

# Strategy for handling missing values:
# 1. 'Age': Impute with median age (median is robust against outliers/skewness)
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)
print(f"\n-> 'Age' column: Imputed 177 missing values with median age = {median_age:.1f} years.")

# 2. 'Embarked': Impute with the most frequent port (mode)
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)
print(f"-> 'Embarked' column: Imputed 2 missing values with mode = '{mode_embarked}'.")

# 3. 'Cabin': Over 77% missing data. Dropping or filling as 'Unknown'.
# Here we fill 'Unknown' so we retain the row data without fabricating cabin numbers.
df["Cabin"] = df["Cabin"].fillna("Unknown")
print(f"-> 'Cabin' column: 687 values (~77.1%) were missing; imputed with label 'Unknown'.")

# Verify no critical missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ===========================================================================
# STEP 4: IDENTIFY AND REMOVE DUPLICATE RECORDS
# ===========================================================================
print_section("Step 4: Duplicate Record Check")

duplicates_count = df.duplicated().sum()
print(f"Number of duplicate rows found: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"-> Successfully dropped {duplicates_count} duplicate row(s).")
else:
    print("-> No duplicate rows detected in this dataset. DataFrame is clean!")


# ===========================================================================
# STEP 5: GROUPING AND AGGREGATION
# ===========================================================================
print_section("Step 5: Grouping and Aggregation")

# 5.1 Overall Survival Rate
overall_survival_rate = df["Survived"].mean() * 100
print(f"Overall Survival Rate: {overall_survival_rate:.2f}%\n")

# 5.2 Survival Rate by Gender (Sex)
print("1. Survival Rate by Gender:")
gender_group = df.groupby("Sex")["Survived"].agg(
    Total_Passengers="count",
    Survived_Count="sum",
    Survival_Rate="mean"
)
gender_group["Survival_Rate_%"] = (gender_group["Survival_Rate"] * 100).round(2)
print(gender_group[["Total_Passengers", "Survived_Count", "Survival_Rate_%"]])

# 5.3 Survival Rate by Passenger Class (Pclass)
print("\n2. Survival Rate by Passenger Class (Pclass):")
pclass_group = df.groupby("Pclass")["Survived"].agg(
    Total_Passengers="count",
    Survived_Count="sum",
    Survival_Rate="mean"
)
pclass_group["Survival_Rate_%"] = (pclass_group["Survival_Rate"] * 100).round(2)
print(pclass_group[["Total_Passengers", "Survived_Count", "Survival_Rate_%"]])

# 5.4 Survival Rate by Gender and Passenger Class
print("\n3. Survival Rate by Gender AND Passenger Class:")
gender_pclass_group = df.groupby(["Sex", "Pclass"])["Survived"].agg(
    Total_Passengers="count",
    Survived_Count="sum",
    Survival_Rate="mean"
)
gender_pclass_group["Survival_Rate_%"] = (gender_pclass_group["Survival_Rate"] * 100).round(2)
print(gender_pclass_group[["Total_Passengers", "Survived_Count", "Survival_Rate_%"]])

# 5.5 Average Age and Average Fare Analysis
print("\n4. Age and Fare Analysis:")
print(f"   -> Overall Average Age : {df['Age'].mean():.2f} years (Median: {df['Age'].median():.2f})")
print(f"   -> Overall Average Fare: ${df['Fare'].mean():.2f} (Median: ${df['Fare'].median():.2f})")

avg_metrics_class = df.groupby("Pclass").agg(
    Average_Age=("Age", "mean"),
    Average_Fare=("Fare", "mean")
).round(2)
print("\n   Average Age and Fare by Passenger Class:")
print(avg_metrics_class)

avg_metrics_survival = df.groupby("Survived").agg(
    Average_Age=("Age", "mean"),
    Average_Fare=("Fare", "mean")
).round(2)
avg_metrics_survival.index = ["Died (0)", "Survived (1)"]
print("\n   Average Age and Fare by Survival Status:")
print(avg_metrics_survival)


# ===========================================================================
# STEP 6: DATA VISUALIZATION WITH MATPLOTLIB
# ===========================================================================
print_section("Step 6: Generating Charts")

# Set consistent Matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.size"] = 11

# --- Chart 1: Survival Rate by Gender ---
fig, ax = plt.subplots(figsize=(6, 5))
genders = [g.capitalize() for g in gender_group.index]
survival_by_gender = gender_group["Survival_Rate_%"]
colors_gender = ["#e75480", "#4682b4"]  # Pink for Female, SteelBlue for Male

bars = ax.bar(genders, survival_by_gender, color=colors_gender, width=0.5, edgecolor="black", linewidth=1.2)
ax.set_title("Survival Rate by Gender", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 100)

# Add text percentage on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
chart1_path = os.path.join(CHARTS_DIR, "survival_by_gender.png")
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f" Saved: {chart1_path}")

# --- Chart 2: Survival Rate by Passenger Class ---
fig, ax = plt.subplots(figsize=(7, 5))
classes = [f"Class {c}" for c in pclass_group.index]
survival_by_class = pclass_group["Survival_Rate_%"]
colors_class = ["#2ca02c", "#ff7f0e", "#d62728"]  # Green (1st), Orange (2nd), Red (3rd)

bars = ax.bar(classes, survival_by_class, color=colors_class, width=0.5, edgecolor="black", linewidth=1.2)
ax.set_title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
chart2_path = os.path.join(CHARTS_DIR, "survival_by_class.png")
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f" Saved: {chart2_path}")

# --- Chart 3: Survival Rate by Gender and Passenger Class ---
fig, ax = plt.subplots(figsize=(8, 5.5))
classes_idx = np.array([1, 2, 3])
bar_width = 0.35

female_rates = [
    gender_pclass_group.loc[("female", c), "Survival_Rate_%"] for c in classes_idx
]
male_rates = [
    gender_pclass_group.loc[("male", c), "Survival_Rate_%"] for c in classes_idx
]

bars1 = ax.bar(classes_idx - bar_width/2, female_rates, width=bar_width,
               label="Female", color="#e75480", edgecolor="black")
bars2 = ax.bar(classes_idx + bar_width/2, male_rates, width=bar_width,
               label="Male", color="#4682b4", edgecolor="black")

ax.set_title("Survival Rate by Class and Gender", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Passenger Ticket Class", fontsize=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_xticks(classes_idx)
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
ax.set_ylim(0, 110)
ax.legend(title="Gender", frameon=True)

# Add annotations
for bar in bars1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

for bar in bars2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

plt.tight_layout()
chart3_path = os.path.join(CHARTS_DIR, "survival_by_gender_class.png")
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f" Saved: {chart3_path}")

# --- Chart 4: Comprehensive 4-Panel EDA Summary Dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot (0,0): Survival by Gender
axes[0, 0].bar(genders, survival_by_gender, color=colors_gender, width=0.45, edgecolor="black")
axes[0, 0].set_title("(A) Survival Rate by Gender", fontweight="bold")
axes[0, 0].set_ylabel("Survival Rate (%)")
axes[0, 0].set_ylim(0, 100)
for b in axes[0, 0].patches:
    axes[0, 0].annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

# Subplot (0,1): Survival by Class
axes[0, 1].bar(classes, survival_by_class, color=colors_class, width=0.45, edgecolor="black")
axes[0, 1].set_title("(B) Survival Rate by Passenger Class", fontweight="bold")
axes[0, 1].set_ylabel("Survival Rate (%)")
axes[0, 1].set_ylim(0, 100)
for b in axes[0, 1].patches:
    axes[0, 1].annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

# Subplot (1,0): Average Fare by Class
avg_fares = avg_metrics_class["Average_Fare"]
axes[1, 0].bar(classes, avg_fares, color=["#1f77b4", "#aec7e8", "#c7c7c7"], width=0.45, edgecolor="black")
axes[1, 0].set_title("(C) Average Fare Paid by Passenger Class", fontweight="bold")
axes[1, 0].set_ylabel("Average Fare ($)")
for b in axes[1, 0].patches:
    axes[1, 0].annotate(f"${b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

# Subplot (1,1): Age Distribution (Survivors vs Deceased)
axes[1, 1].hist(df[df["Survived"] == 1]["Age"], bins=20, alpha=0.6, label="Survived", color="#2ca02c", edgecolor="black")
axes[1, 1].hist(df[df["Survived"] == 0]["Age"], bins=20, alpha=0.5, label="Did Not Survive", color="#d62728", edgecolor="black")
axes[1, 1].set_title("(D) Age Distribution by Survival Status", fontweight="bold")
axes[1, 1].set_xlabel("Age (Years)")
axes[1, 1].set_ylabel("Number of Passengers")
axes[1, 1].legend()

plt.suptitle("Titanic Exploratory Data Analysis (EDA) Summary Dashboard", fontsize=16, fontweight="bold", y=0.99)
plt.tight_layout()
dashboard_path = os.path.join(CHARTS_DIR, "titanic_eda_summary_dashboard.png")
plt.savefig(dashboard_path, dpi=300)
plt.close()
print(f" Saved: {dashboard_path}")


# ===========================================================================
# STEP 7: OBSERVATIONS & KEY INSIGHTS (CALCULATED FROM ACTUAL DATA)
# ===========================================================================
print_section("Step 7: Key Observations & Insights")

print(f"""
[1] GENDER IMPACT:
    - Female survival rate was {gender_group.loc['female', 'Survival_Rate_%']:.2f}% (233 out of 314 survived).
    - Male survival rate was {gender_group.loc['male', 'Survival_Rate_%']:.2f}% (109 out of 577 survived).
    - Observation: Females were nearly 4 times more likely to survive than males,
      strongly confirming the 'women and children first' maritime protocol.

[2] SOCIO-ECONOMIC STATUS (TICKET CLASS):
    - 1st Class survival rate: {pclass_group.loc[1, 'Survival_Rate_%']:.2f}%
    - 2nd Class survival rate: {pclass_group.loc[2, 'Survival_Rate_%']:.2f}%
    - 3rd Class survival rate: {pclass_group.loc[3, 'Survival_Rate_%']:.2f}%
    - Observation: Passenger class had a direct linear correlation with survival.
      First-class passengers enjoyed top-deck accommodation proximity to lifeboats
      and prioritized evacuation.

[3] INTERSECTION OF GENDER AND CLASS:
    - 1st Class Females had a {gender_pclass_group.loc[('female', 1), 'Survival_Rate_%']:.2f}% survival rate (91 of 94 survived).
    - 2nd Class Females had a {gender_pclass_group.loc[('female', 2), 'Survival_Rate_%']:.2f}% survival rate (70 of 76 survived).
    - 3rd Class Females had a {gender_pclass_group.loc[('female', 3), 'Survival_Rate_%']:.2f}% survival rate (50.00%).
    - In contrast, 3rd Class Males had a survival rate of only {gender_pclass_group.loc[('male', 3), 'Survival_Rate_%']:.2f}% (47 of 347 survived).
    - Observation: Upper-class females experienced near-total survival, whereas
      third-class men bore the heaviest casualty rates.

[4] FARE AND AGE PATTERNS:
    - The average ticket fare was ${df['Fare'].mean():.2f}.
    - Passengers who survived paid an average fare of ${df[df['Survived']==1]['Fare'].mean():.2f},
      compared to only ${df[df['Survived']==0]['Fare'].mean():.2f} for those who perished.
    - The average passenger age was {df['Age'].mean():.2f} years. 1st class passengers
      were older on average ({avg_metrics_class.loc[1, 'Average_Age']:.1f} yrs) compared to 3rd class passengers ({avg_metrics_class.loc[3, 'Average_Age']:.1f} yrs).
""")


# ===========================================================================
# STEP 8: CONCLUSION
# ===========================================================================
print_section("Step 8: Conclusion")

print("""
Summary Conclusion:
The exploratory data analysis reveals that survival on the Titanic was not random.
Instead, survival chances were strongly determined by gender and socio-economic
status (passenger class and fare paid). The historical evacuation guideline of
'women and children first' gave female passengers an overwhelming survival advantage.
Simultaneously, first-class passengers who paid significantly higher fares had faster
access to lifeboat decks and privileged boarding, leading to far higher survival
rates compared to third-class passengers.
""")

print("=" * 78)
print(" EDA Execution Completed Successfully!")
print(f" All 4 charts have been saved into the folder:\n  '{CHARTS_DIR}'")
print("=" * 78)
