# Titanic Exploratory Data Analysis (EDA) Project

A complete, beginner-friendly Python data science project analyzing passenger survival patterns from the historic Titanic disaster.

---

## 📌 Project Overview

This project was built for an **Exploratory Data Analysis (EDA)** college assignment. It uses Python and standard data science libraries (**Pandas**, **NumPy**, and **Matplotlib**) to load real-world data, inspect its structure, clean missing and duplicate values, perform statistical grouping and aggregations, and generate clear visualizations.

All calculations and observations in this project are derived directly from the actual Titanic dataset (no fabricated numbers).

---

## 📂 Project Directory Structure

```text
Titanic_EDA/
│
├── data/
│   └── titanic.csv                    # Cached dataset (automatically downloaded)
│
├── charts/
│   ├── survival_by_gender.png         # Bar chart: Female vs. Male survival rate
│   ├── survival_by_class.png          # Bar chart: Survival rate across ticket classes
│   ├── survival_by_gender_class.png   # Grouped bar chart: Gender × Class survival rate
│   └── titanic_eda_summary_dashboard.png # 4-panel visual summary dashboard
│
├── titanic_eda.py                     # Main Python script with complete EDA pipeline
├── requirements.txt                   # Required Python libraries
└── README.md                          # Project documentation and student guide
```

---

## 📊 Dataset Details

- **Dataset Name:** Titanic: Machine Learning from Disaster
- **Source:** [Data Science Dojo Public Repository (GitHub)](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Shape:** 891 rows (passengers) × 12 columns (features)
- **Key Columns:**
  - `PassengerId`: Unique identifier for each passenger
  - `Survived`: Survival status (`0 = No`, `1 = Yes`)
  - `Pclass`: Ticket class (`1 = 1st`, `2 = 2nd`, `3 = 3rd`)
  - `Name`: Full name of passenger
  - `Sex`: Gender (`female`, `male`)
  - `Age`: Age in years
  - `SibSp`: Number of siblings / spouses aboard
  - `Parch`: Number of parents / children aboard
  - `Ticket`: Ticket number
  - `Fare`: Passenger fare in British pounds
  - `Cabin`: Cabin number
  - `Embarked`: Port of embarkation (`C = Cherbourg`, `Q = Queenstown`, `S = Southampton`)

---

## ⚙️ Installation & Requirements

### 1. Prerequisites
- Python 3.8 or higher installed on your computer.

### 2. Install Required Dependencies
Open your terminal or command prompt inside the `Titanic_EDA` folder and run:

```bash
pip install -r requirements.txt
```

*The required packages are:*
- `pandas>=2.0.0` (Data manipulation and analysis)
- `numpy>=1.24.0` (Numerical computing and arrays)
- `matplotlib>=3.7.0` (Data visualization and plotting)

---

## 🚀 How to Run the Project

1. Navigate to the project directory:
   ```bash
   cd Titanic_EDA
   ```
2. Run the main script:
   ```bash
   python titanic_eda.py
   ```
3. Once completed:
   - All statistical tables and insights will be printed directly in your terminal.
   - All generated charts will be automatically saved into the `charts/` folder.
   - A copy of the dataset will be saved into the `data/` folder for offline use.

---

## 📖 Step-by-Step EDA Workflow

The script is organized into 8 easy-to-follow steps:

### Step 1: Loading Data
Loads `titanic.csv` directly from a verified public URL into a Pandas `DataFrame`. If an internet connection is unavailable, it automatically falls back to a locally cached copy.

### Step 2: Basic DataFrame Operations
Demonstrates fundamental Pandas operations:
- `df.shape`: Returns dimensions (891 rows, 12 columns).
- `df.columns`: Lists all feature names and data types.
- `df.head()`: Displays the first 5 records.
- `df.info()`: Displays non-null counts and memory usage.
- `df.describe()`: Provides descriptive statistics (mean, std, min, 25%, median, 75%, max) for numerical variables.

### Step 3: Missing Value Treatment
- **Identified:**
  - `Age`: 177 missing values (19.87%)
  - `Cabin`: 687 missing values (77.10%)
  - `Embarked`: 2 missing values (0.22%)
- **Handling Strategy:**
  - Imputed `Age` with the **median age (28.0 years)** because the median is resistant to extreme outliers.
  - Imputed `Embarked` with the **mode ('S')**, representing the most frequent embarkation port.
  - Filled missing `Cabin` entries with the string `"Unknown"` because over 77% was absent, making value imputation unreliable.

### Step 4: Duplicate Record Handling
- Executed `df.duplicated().sum()` to check for duplicate records.
- Confirmed **0 duplicate rows** exist in the dataset; included `df.drop_duplicates()` to demonstrate good cleaning practices.

### Step 5: Grouping and Aggregations
Calculated key survival rates and averages:
- Overall survival rate
- Survival rate grouped by `Sex`
- Survival rate grouped by `Pclass`
- Survival rate grouped across both `Sex` and `Pclass`
- Average Age and Average Fare comparisons

### Step 6: Data Visualization
Matplotlib produces 4 high-resolution (300 DPI) publication-quality charts:
1. `survival_by_gender.png`: Bar chart contrasting female vs. male survival rates.
2. `survival_by_class.png`: Bar chart comparing survival across classes 1, 2, and 3.
3. `survival_by_gender_class.png`: Multi-bar chart highlighting intersectional survival rates.
4. `titanic_eda_summary_dashboard.png`: 4-in-1 consolidated dashboard.

### Step 7: Key Observations & Insights
Synthesizes factual takeaways from the computed data.

### Step 8: Academic Conclusion
Wraps up the findings in clear, academic language suitable for submission.

---

## 📈 Real Calculated Results & Findings

### 1. Survival Rate by Gender
| Gender | Total Passengers | Survived Count | Survival Rate |
| :--- | :---: | :---: | :---: |
| **Female** | 314 | 233 | **74.20%** |
| **Male** | 577 | 109 | **18.89%** |

> **Key Finding:** Females were nearly **4 times more likely to survive** than males.

---

### 2. Survival Rate by Passenger Class
| Class | Total Passengers | Survived Count | Survival Rate |
| :--- | :---: | :---: | :---: |
| **1st Class** | 216 | 136 | **62.96%** |
| **2nd Class** | 184 | 87 | **47.28%** |
| **3rd Class** | 491 | 119 | **24.24%** |

> **Key Finding:** A strong positive correlation exists between socio-economic status and survival. First-class passengers were over **2.5 times more likely to survive** than third-class passengers.

---

### 3. Survival Rate by Gender and Class
| Gender | Class | Total Passengers | Survived Count | Survival Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Female** | 1st Class | 94 | 91 | **96.81%** |
| **Female** | 2nd Class | 76 | 70 | **92.11%** |
| **Female** | 3rd Class | 144 | 72 | **50.00%** |
| **Male** | 1st Class | 122 | 45 | **36.89%** |
| **Male** | 2nd Class | 108 | 17 | **15.74%** |
| **Male** | 3rd Class | 347 | 47 | **13.54%** |

> **Key Finding:** Upper-class females experienced near-total survival (96.81%), whereas third-class males suffered the heaviest mortality (only 13.54% survived).

---

### 4. Fare & Age Averages
- **Overall Average Age:** 29.70 years (imputed median: 28.00)
- **Overall Average Fare:** \$32.20
- **Average Fare by Class:**
  - 1st Class: **\$84.15**
  - 2nd Class: **\$20.66**
  - 3rd Class: **\$13.68**
- **Average Fare by Survival Status:**
  - Survived: **\$48.40**
  - Perished: **\$22.12**

---

## 🎯 Conclusion

The exploratory data analysis reveals that survival on the Titanic was heavily non-random:
1. **Gender Protocol:** The historical protocol of *"women and children first"* gave female passengers a massive survival advantage (74.20% vs 18.89%).
2. **Class Privilege:** Proximity to upper lifeboat decks and social privilege conferred by first-class tickets resulted in significantly higher survival odds.
3. **Compounded Vulnerability:** Third-class male passengers faced the combined disadvantages of lower deck location, delay in lifeboat access, and low evacuation priority.

---

## 📝 Assignment Submission Note

This repository contains all your code, cached data, visualizations, and documentation. You can review all files, inspect the output in your terminal, and submit them according to your university/college guidelines.
