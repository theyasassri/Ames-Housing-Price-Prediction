# Ames-Housing-Price-Prediction
yasassri Ekanayake undergratuate in SLIIT, Sri lanka(2nd year)
My 2nd Project that I used python and google colab"Predicting house prices using Linear Regression and Feature Engineering."
# 🏡 Ames Housing Price Prediction: A Machine Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

##  Project Overview
The goal of this project was to move beyond descriptive analysis and build a **Predictive Machine Learning Model** capable of estimating residential house prices in Ames, Iowa. By analyzing physical property features, I developed a Linear Regression model that predicts sale prices with **~90% accuracy**.

**Key Question:** Can we predict a house's value using only its size, quality, and age?

---

##  Technical Approach

### 1. Data Cleaning & Strategic Imputation
To ensure the integrity of the predictive model, I performed a two-step imputation process to handle missing values without sacrificing valuable data:
* **Categorical Missingness:** Addressed features like `PoolQC` and `FireplaceQu` by filling null values with **'None'**. This correctly identified that a missing entry represented the *absence of a feature* rather than a recording error.
* **Numerical Missingness:** Utilized **Median Imputation** for `LotFrontage` to fill gaps with the typical street-connection value, preventing the loss of nearly 90% of observations.

### 2. Feature Engineering
I improved model performance by creating new features that better represent value:
* **Total Square Footage (`TotalSF`):** Combined basement, first floor, and second floor areas into a single metric. This new feature showed a correlation of **0.78** with price, outperforming individual floor measurements.

### 3. Exploratory Data Analysis (EDA)
* **Correlation Heatmap:** Identified `OverallQual` (0.8) and `GrLivArea` (0.7) as the strongest drivers of price.
* **Outlier Detection:** Visualized "High-Size, Low-Price" outliers using regression plots and excluded them to prevent model skew.

---

##  Model Performance

I trained a **Linear Regression** model using an 80/20 train-test split. The final evaluation demonstrates strong predictive capabilities.

* **Mean Absolute Error (MAE):** **$24,810.15**
* **Model Accuracy:** ~90%

### Actual vs. Predicted Analysis
The table below shows the model's performance on five random test cases. The model is highly accurate for standard homes but tends to be more conservative on luxury properties.

| Property ID | Actual Sale Price | Predicted Price | Variance (Error) |
| :--- | :--- | :--- | :--- |
| **892** | $154,500 | $147,122 | -$7,378 |
| **1105** | $325,000 | $287,519 | -$37,481 |
| **413** | $115,000 | $130,063 | +$15,063 |
| **522** | $159,000 | $180,077 | +$21,077 |
| **1036** | $315,500 | $295,836 | -$19,664 |

---

## 💡 Key Insights
1.  **Size is Secondary to Quality:** A house with lower square footage but higher material quality often sells for more than a larger, lower-quality home.
2.  **The Luxury Variance:** As property quality increases (Level 10), price variance increases significantly compared to standard homes (Level 5).
3.  **Feature Impact:** Adding `YearBuilt` and `GarageCars` reduced the error margin by nearly $1,000, confirming that age and utility are critical secondary value drivers.

---

##  Technologies Used
* **Python:** The core programming language.
* **Pandas:** For data manipulation and aggregation.
* **Seaborn/Matplotlib:** For correlation heatmaps and regression plotting.
* **Scikit-Learn:** For Linear Regression model building and evaluation metrics.

##  How to Run This Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/theyasassri/Ames-Housing-Price-Prediction.git](https://github.com/theyasassri/Ames-Housing-Price-Prediction.git)
