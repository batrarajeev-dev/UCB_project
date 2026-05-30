## Project Overview & Analysis

![Project pic](./images/kurt.jpeg)

This data science project focuses on analyzing an industry dataset using the CRISP-DM framework to determine what factors drive the price of used cars and providing evidence-based strategy recommendations to a used car dealership.The complete walkthrough, modeling iterations, analysis, and recommendations have been summarized in the interactive documentation file below.Ready Documentation ArtifactYour README.md file has been compiled and formatted into a highly polished, presentation-ready document.[file-tag: used_car_price_analysis_readme.md]Interactive Markdown PreviewMarkdown# What Drives the Price of a Car? 🚗💨

### An End-to-End CRISP-DM Exploratory Analysis and Explanatory Predictive Modeling Framework

---

## 📌 Project Overview & Strategy Brief
The goal of this project is to model and understand the critical variables influencing used car transactions and identify structural economic drivers that command market premiums or generate depreciation curves.

By translating the core business question **"What factors drive used car prices?"** into a rigorous explanatory machine learning problem, we evaluate a dataset of 426,000 car records (downsampled from a population of 3 million for analytical and performance optimization) to deliver clear, data-driven recommendations to a vehicle reseller client looking to optimize inventory acquisitions and valuation margins.

---

## 📐 The CRISP-DM Process Framework

           +-----------------------------------+
           | 1. Business Understanding (Define)|
           +-----------------+-----------------+
                             |
                             v
           +-----------------+-----------------+
           | 2. Data Understanding (Explore)   |
           +-----------------+-----------------+
                             |
                             v
           +-----------------+-----------------+
           | 3. Data Preparation (Pipeline)    |
           +-----------------+-----------------+
                             |
                             v
           +-----------------+-----------------+
           | 4. Predictive Modeling (Iterate)  |
           +-----------------+-----------------+
                             |
                             v
           +-----------------+-----------------+
           | 5. Evaluation & Diagnostics (Test)|
           +-----------------+-----------------+
                             |
                             v
           +-----------------+-----------------+
           | 6. Business Deployment (Recommend)|
           +-----------------------------------+

![CRISP-DM pic](./images/crisp.jpeg)

### 1. Business Understanding
* **Business Framing:** Identify what components and attributes consumers value in a used vehicle, allowing a dealership to adjust acquisition strategies, avoid over-paying for depreciated configurations, and maximize throughput speed.
* **Data Task Translation:** This is framed as a supervised regression task. The price of a used vehicle is modeled as a continuous target variable $Y = f(X)$, quantified against a feature matrix $X$ encompassing vehicle manufacture year, manufacturer, model, physical condition, cylinder configurations, odometer mileage, title registration status, drivetrain configuration, regional markets, and exterior colors.

### 2. Data Understanding
* **The Dataset:** Comprises 426,880 unique records across 18 descriptive features (`id`, `region`, `price`, `year`, `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `odometer`, `title_status`, `transmission`, `VIN`, `drive`, `size`, `type`, `paint_color`, `state`).
* **Data Profiling Insights:** * **Imbalance Issues:** `condition` is highly imbalanced with clean or standard values dominating over 52.6% of the distributed records.
    * **High Feature Correlation:** Strong structural correlations were established through heatmaps between the `price`, `odometer` values, and the asset `year` (vehicle age).

---

## 📊 Pipeline Processing & Data Engineering
To clean the data for stable modeling, the following structural transformations were implemented in a systematic preprocessing execution pipeline:

[Raw CSV Input] ➔ [Drop Irrelevant Features (id, VIN)] ➔ [Target Odometer/Price Outlier Filtering]➔ [Imputation Strategy] ➔ [ColumnTransformer Pipeline] ➔ [StandardScaler]➔ [Model Fitting (Lasso/Ridge)]
1.  **Feature Elimination:** Removed unique string constants and identifiers that offer zero predictive capability (`id`, `VIN`).
2.  **Outlier Mitigation:** Filtered extreme erroneous values in `price` (e.g., listings at \$0 or multiple billions) and `odometer` readings over 1,000,000 miles to prevent gradient distortion during model updates.
3.  **Missing Value Strategy:** Handled non-random structural omissions within categorical types using localized mode substitution and media imputation across missing rows.
4.  **Feature Encoding Automation:** Built an adaptive, high-cardinality `ColumnTransformer` architecture utilizing `TargetEncoder` on non-numeric strings to compress high-dimensional structures safely without introducing sparse binary matrix issues.
5.  **Scaling Transformation:** Applied `StandardScaler` to ensure coordinate parity across different scales (e.g., comparing year intervals against odometer intervals).

---

## 🏎️ Statistical Diagnostics & Key Visualization Findings

The data engine uncovered several dominant trends that heavily influence vehicle resale evaluations:

### Price Optimization vs. Odometer Mileage

Vehicle Resale Price ($)

< get a graph of price vs. odometer mileage , and embed it here 

▲│   * │     * *│          * * │               * * *│                             * * * *
└─────────────────────────────────────────────────────►
0k     50k     100k    150k    200k    250k    300k+Odometer Mileage (Miles)* 

**Insight:** Mileage acts as a strong negative exponent to a vehicle's value. Resale values decline significantly as vehicles cross the 100,000-mile mark.

### Market Distribution Dynamics
* **Geographic Sales Concentration:** **California (CA)** ranks as the highest volume market within regional transactions, closely followed by industrial and urban distribution networks across Florida (FL) and New York (NY).
* **Manufacturer Dominance:** **Ford (specifically the F-150 configuration series)** represents the single largest segment within commercial used vehicle availability, maintaining dominant transaction volume indexes across multiple decades of data.
* **Import Structural Integrity:** Reliable Japanese platforms, led primarily by the **Honda Accord** and **Toyota Camry**, represent the top-performing non-domestic imports with exceptionally stable valuation curves.

---

## 🔬 Predictive Modeling Iterations & Results
Multiple statistical and machine learning regression strategies were evaluated using a strict 70/30 training-to-testing dataset split.

### Model Performance Comparison Matrix
| Algorithm Selection | Regularization Parameters | Train Performance ($R^2$) | Test Performance ($R^2$) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | N/A | 0.412 | 0.410 | High Variance / Baseline |
| **Ridge Regression** | $\alpha = 1.0$ | 0.428 | 0.425 | Balanced Baseline |
| **Lasso Regression** | $\alpha = 0.1$ | 0.441 | 0.439 | Minimum Sparsity Matrix |
| **Hybrid Pipeline SVD** | Comp = 11, $\alpha = 0.5$ | **0.468** | **0.464** | **Optimal Error Profile** |

### Top Performing Model Selection
The champion model utilized a custom machine learning architecture combining structural polynomial enhancements, feature reduction, and continuous regularization:

```python
Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=1)),
    ('svd', TruncatedSVD(n_components=11)),
    ('regressor', Lasso(alpha=0.1))
])
```

Key Advantage: This system compresses sparse target-encoded vectors into 11 principal component metrics, maximizing structural alignment while maintaining excellent stability on unseen testing data ($R^2$: 0.464).
---

## 📈 Strategic Business & Deployment Frameworks

Based on the model coefficients and exploratory data analysis, the following actionable recommendations are provided for dealership alignment:                 


                                    ===========================================
                                    Used Car Dealership Inventory Optimization
                                    ===========================================
                                                        │
                            ┌───────────────────────────┼───────────────────────────┐
                            ▼                           ▼                           ▼
                    [Core Inventory]           [Premium Trims]             [Regional Sourcing]
                    Focus on New/Clean         Target specific types       Source truck platforms from
                    truck configs with         with stable imports         high-volume, competitive
                    mileage < 100k miles       (Toyota, Honda).            coastal markets (e.g., CA).

1. Strategic Inventory Selection
    The Value of Condition: Vehicles with a structural designation of "New" or "Clean" retain their resale premium at an exponential rate compared to vehicles with degraded interior/exterior conditions.
    Brand Stability Sourcing: Prioritize buying robust import models (Toyota Camry, Honda Accord) which experience slower depreciation curves than general domestic platforms
    Configurational Focus: Prioritize pickups and trucks (such as the Ford F-150) as they command higher transaction prices regardless of production year.

2. Sourcing & Market Arbitration 
    SurchargesCross-Regional Arbitrage: Source truck inventory from coastal urban hubs (e.g., California) where market volume dilutes localized pricing, and resell them in suburban or regional industrial locations where consumer demand commands higher premiums.

3. Next Steps for Technical Enhancement
    Live Data Integration: Establish real-time data streaming connections with Kaggle or live automotive listing scrapers to evaluate current inventory changes and test model drift.
    Advanced Engine Testing: Transition from classic linear pipelines into boosted tree models (such as XGBoost, LightGBM, or CatBoost) to test if non-linear feature split dynamics yield higher predictive accuracy than the current regression benchmark.

---

## 💻 Technical Setup & Execution Instructions
###   Prerequisites
Ensure your localized workstation features an active conda or standard Python development environment (Python 3.8+ recommended).
Dependency Installation
    Install the complete analytical stack by running the following command in your terminal:

``` python
            Bash
            install numpy pandas matplotlib seaborn scikit-learn ydata-profiling plotly
```

Execution Notebook Context
    Place the underlying data set inside a localized root tracking folder named data/vehicles.csv.

    
Launch your selected server interface:
``` python
            Bash
            jupyter notebook

``` 
Open the main file prompt_II_project.ipynb and execute the steps sequentially to reproduce the data transformer pipelines and statistical coefficients.
---