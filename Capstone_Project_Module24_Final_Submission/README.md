### Project Title
#### Intelligent Fleet Recommendation and  Optimization System 
(for European Commercial Fleet Vehicle Classification - Make_n_Model)

**Author**
Rajeev Batra

#### Executive summary

 Globally over 70% of the freight moves on road and the fleet management is estimated to be a 30 billion $ business with 18% CAGR.  However, fleet management is a very thrifty business as it relies on razor thin margins and seasonality. A fleet is almost always made up of vehicles from different manufacturers, models and configurations. This ads up to the problem of selecting the right vehicle for a given mission considering multiple of parameters such as load, distance, destination etc. 
 The aim of this project is to build a vehicle recommendation and assignment model for commercial vehicle fleets to ensure that correct vehicle is assigned to correct mission as it can help the fleet owner save between 2~3% of the operational cost and this is a direct addition to the bottom line. 

 This capstone project (`Capstone_Project_M24_final.ipynb`) develops and evaluates supervised multiclass machine learning pipelines to classify commercial fleet vehicles into **21 unique `Make_n_Model` categories** based on trip-level telematics, payload weight ratings, and specialized body/cargo requirements (`special_need`). Using a synthetic fleet dataset of **400,000 trip records** (`391,204` cleaned records after filtering), nine classification pipelines—spanning four baseline algorithms and five hyperparameter-tuned models via 3-fold `RandomizedSearchCV`—were benchmarked on accuracy, weighted Precision, weighted F1-score, One-vs-Rest (OvR) ROC-AUC, and computational training time.

 Across all evaluated models, test accuracy converged to a narrow ceiling of **71.50% to 71.81%** (weighted F1-score of **0.7081 to 0.7124**). Detailed class-level confusion matrix analysis revealed that **12 of the 21 vehicle classes are classified with 1.00 F1-score**, whereas **7 European heavy-duty 44-ton GCW / 26-ton GVW truck models** (`DAF_XG+`, `Iveco_S-Way`, `MAN_TGX`, `Mercedes_Actros`, `Renault_T High`, `Scania_R-Series`, and `Volvo_FH`) share identical static weight limits and overlapping cargo profiles, creating an irreducible feature-overlap ceiling under the current 8-feature input set.

#### Rationale
Why should anyone care about this question?

Commercial logistics and freight operators face severe financial and environmental losses from suboptimal vehicle dispatching:

-   **Widespread Payload Under-Utilization ("Ghost Loads")**: Dispatching oversized heavy-duty trucks for light loads wastes fuel, inflates per-kilometer operating costs, and increases carbon emissions.
-   **Severe Overload and  Safety Risks**: Conversely, assigning an undersized vehicle whose rated capacity falls below the actual cargo load ($C_{\text{pred}} < C_{\text{actual}}$) causes accelerated mechanical wear, braking hazards, regulatory fines, and unscheduled maintenance downtime.
-   **Operational Complexity at Scale**: Dispatchers must manually balance physical cargo weight, route distance, regional terrain, and specialized cargo constraints (such as refrigerated reefers or hazardous material containment) across dozens of competing vehicle models. Automating this decision with a cost-sensitive machine learning recommender enables consistent fleet right-sizing, reduces fuel burn, and protects fleet assets.

#### Research Question
What are you trying to answer?

 **How can we accurately recommend the optimal commercial vehicle (`Make_n_Model`) for a given logistics trip—using pre-dispatch route, payload weight, and specialized cargo attributes—to maximize payload utilization and minimize excess fuel consumption without risking vehicle overload? and to minimize under-utilization and reduce trip operating costs by 2–3% and this is big savings given the razor thin  margins in fleet management**

Specifically, this study investigates:
1. Which supervised machine learning architecture (regularized linear models, distance-based neighborhoods, kernel SVMs, or tree-based ensembles) best maps trip and cargo characteristics to the 21 available vehicle configurations?
2. How does evaluating models with an **asymmetric domain cost-benefit loss function (`custom_fleet_loss`)** compare against standard **Weighted F1-Score (`f1_weighted`)** when multiple competing manufacturers offer functionally identical trucks within the same tonnage tier?
3. Which trip and cargo attributes drive optimal vehicle dispatch decisions across regional logistics networks?

#### Data Sources
What data will you use to answer you question?

Because commercial carriers treat multi-year fleet telematics as proprietary, gathering bulk historical records across multiple fleets over a six-year window was not feasible due to data confidentiality. Instead,  based on the discussion with the learning facilitator I combined real-world fleet seed data with a rule-constrained LLM generation pipeline to build a realistic dataset of 400,000 trips across 10,000 unique vehicles spanning 2018–2024. To ensure the synthetic data reflects actual commercial operations, the generation process enforces four domain rules:

Immutable Vehicle Registry: A master registry locks physical specifications—Gross Vehicle Weight (GVW), Gross Combination Weight (GCW), and payload capacity—to unique serial numbers across 10 manufacturers (Fuso, Isuzu, Hino, Renault, MAN, DAF, Volvo, IVECO, Scania, and Mercedes-Benz), ranging from 1.5-tonne panel vans (Renault Master) to 40-tonne heavy haulers (Volvo FH16, Scania S-Series V8).

Physics and Regulatory Constraints: Trip distances, speeds, and durations follow vehicle segment physics and EU road laws—limiting Light Commercial Vehicles (LCVs) to short urban deliveries (5–150 km) while capping heavy trucks (200–2,500 km) at 85–90 km/h and automatically inserting EU-mandated 45-minute breaks every 4.5 driving hours and 11-hour rest intervals every 9 driving hours.

Six-Year Timeline Alignment: Trip start times are distributed across a 2018–2024 window, with end timestamps calculated to the exact second from driving time plus mandatory rest offsets.

Chassis-Matched Vocations: Special mission needs are paired with appropriate chassis configurations, mapping urban box and refrigerated units to LCVs, tippers and refuse bodies to medium rigid trucks (such as DAF LF and Volvo FL), and cold-chain, container, timber, and heavy haulage to high-tonnage tractors.

This study uses a large-scale commercial fleet telemetry and dispatch dataset (`synthetic_fleet_data_400k_splneeds.csv`) capturing vehicle specifications, route telemetry, and cargo characteristics across multi-regional logistics operations.

### Dataset Summary and  Schema

| Attribute | Description |
| :--- | :--- |
| **Source File** | `synthetic_fleet_data_400k_splneeds.csv` |
| **Initial Volume** | **400,000** synthetic commercial trip logs , generated from a actual seed data set and criterias. |
| **Cleaned Modeling Dataset** | **391,204 records** after filtering out non-freight `Ambulance` emergency dispatch records and removing unique trip identifiers (`trip_id`) |
| **Target Variable (`Make_n_Model`)** | **21 unique vehicle classes** (combining `manufacturer` and `Model`) spanning **9 rated payload capacity tiers** (`1.5t` to `40t`) |
| **Retained Numerical Predictors (4)** | `trip_duration` (minutes), `total_distance` (km), `total_fuel` (liters), `axle_weight_average` (kg) |
| **Retained Categorical Predictors (3)** | `special_need` (e.g., Refrigerated, Curtainsider, Hazmat, Standard) |
| **Excluded Leakage Columns** | `trip_id`, `Group`,`Serial Number	plant`,	`Region`,	`Country manufacturer`,	`wmi` , and raw trip timestamps (dropped prior to modeling due to deterministic 1:1 mapping with the target vehicle) |
| **Train / Test Partition** | **80% Train** (`3,12,963` records) / **20% Test** (`78,241` records), stratified by `Make_n_Model` |


#### Methodology
What methods are you using to answer the question?

Following the **CRISP-DM** framework, I began with Exploratory Data Analysis (EDA), feature encoding, and L1/L2 regularization to handle mixed vehicle and route attributes. I then trained and compared classification and recommendation algorithms—including K-Nearest Neighbors (KNN), tree-based ensembles, SVM, and regularized linear models—evaluated on Accuracy, Precision, Recall, and Weighted F1-Score, paired with a custom cost-benefit loss function that explicitly penalizes payload under-utilization and excess fuel burn.

### 1. Exploratory Data Analysis (EDA) and  Domain Feature Engineering

To quantify fleet efficiency prior to modeling, we conducted 

####  Correlation Analysis and Multicollinearity Structure**
####  Multi-Modal Trip Duration and 3-Tier Staircase Pairplot 
The data reveals two structural signatures of commercial European fleet operations:

1. **Five Multi-Modal Trip Duration Peaks**:
   - `trip_duration` separates into **5 discrete operational regimes** separated by mandatory EU tachograph daily rest periods (`11 hours = 39,600 seconds`) between driving shifts on multi-day trips:

![Total Distance vs Trip Duration Scatter](code/images/05_distance_vs_duration_scatter.png)

![Trip Duration Distribution Histogram](code/images/06_trip_duration_distribution.png)

2. **Three-Tier Vehicle Staircase in the Corner Pairplot**:
   - Inspecting the pairplot scatter subplots of `axle_weight_average`, `total_distance`, and `total_fuel` uncovers **3 exact operational vehicle tiers** built into the dataset:
     - **Tier 1 — Light Commercial Vans (`GCW 6–11t, Cap 1.5–4t`)**: Distance `5–150 km`, Fuel Rate `8–15 L/100km`, Axle Weight `1,000–3,000 kg`.
     - **Tier 2 — Medium-Duty Rigid Trucks (`GCW 24–32t, Cap 6–10t`)**: Distance `50–450 km`, Fuel Rate `15–25 L/100km`, Axle Weight `3,000–7,000 kg`.
     - **Tier 3 — Heavy-Duty Articulated Trucks (`GCW 44–60t, Cap 24–40t`)**: Distance `200–2,500 km`, Fuel Rate `25–35 L/100km`, Axle Weight `6,000–11,500 kg`.

![Numerical Features Corner Pairplot](code/images/11_numerical_features_pairplot.png)

#### Special Need Category Distribution and Boxplots

Next we compare `trip_duration`, `total_fuel`, and `total_distance` across all 10 `special_need` cargo categories, and later we rank the category trip counts:

![Special Need Boxplots (Duration, Fuel, Distance)](code/images/03_special_need_boxplots.png)

### 2. Leakage-Free Preprocessing Pipeline

To prevent data leakage across cross-validation folds, all transformations were encapsulated inside a Scikit-Learn `ColumnTransformer`:
* **Numerical Pipeline**: Median imputation (`SimpleImputer(strategy='median')`) followed by zero-mean, unit-variance standardization (`StandardScaler()`).
* **Categorical Pipeline**: Mode imputation (`SimpleImputer(strategy='most_frequent')`) followed by dense one-hot encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
* **

### 3. Models Trained and  Hyperparameter Tuning Approach

Five distinct supervised classification families were trained and tuned using 3-fold stratified cross-validation (`cv=3`) optimizing **Weighted F1-Score (`scoring='f1_weighted'`)**:

| Model Family | Baseline Configuration | Hyperparameter Tuning Strategy (`cv=3`, `scoring='f1_weighted'`) | Optimal Hyperparameters Found |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `max_iter=500`, `C=1.0`, `penalty='l2'` | `RandomizedSearchCV` (`n_iter=5`) over `C` $\in \text{loguniform}(10^{-3}, 10^2)$ and `penalty` $\in \{\text{L1}, \text{L2}\}$ with `saga` solver | `C=1.9812`, `penalty='l1'`, `solver='saga'` |
| **Random Forest** | `n_estimators=100`, `max_depth=15` | `RandomizedSearchCV` (`n_iter=5`) over `n_estimators` (50–150), `max_depth` (10, 20, None), `min_samples_split` (2–10), `min_samples_leaf` (1–4) | `n_estimators=113`, `max_depth=10`, `min_samples_leaf=4`, `min_samples_split=6` |
| **Support Vector Classifier (SVC)** | `kernel='rbf'`, `C=1.0`, `max_iter=1000` | `RandomizedSearchCV` (`n_iter=5`) over `C` $\in \text{loguniform}(0.1, 10)$, `kernel` $\in \{\text{rbf}, \text{linear}\}$, `gamma` $\in \{\text{scale}, \text{auto}\}$ | `C=0.2561`, `gamma='scale'`, `kernel='rbf'` |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=5`, `weights='uniform'` | `GridSearchCV` over `n_neighbors` $\in \{5, 9, 15\}$, `weights` $\in \{\text{uniform}, \text{distance}\}$, `p` $\in \{1, 2\}$ | `n_neighbors=15`, `p=1` (Manhattan), `weights='uniform'` |
| **Decision Tree** | `max_depth=15`, `random_state=42` | Baseline structural benchmark | `max_depth=15` |


#### Results
What did your research find?

### Baseline vs. Hyperparameter-Tuned Model Performance

Four baseline classifiers and five tuned classifiers using 3-fold `RandomizedSearchCV` (`n_iter=30`, `scoring='f1_weighted'`, `random_state=42`, were evaluated on the `78,241`-row test set:

| Rank (by F1) | Model | Train Time (s) | Train Accuracy | Test Accuracy | Weighted Precision | Weighted F1 | Weighted ROC-AUC | Optimal Hyperparameters (`Best Params`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Random Forest Tuned** | 826.37 | 0.8242 | 0.7150 | 0.7466 | **0.7124** | 0.9613 | `n_estimators=99`, `max_depth=20`, `max_features='log2'`, `min_samples_split=5`, `min_samples_leaf=1`, `criterion='gini'` |
| **2** | **Decision Tree Tuned** | 27.88 | 0.7347 | 0.7163 | 0.7393 | 0.7121 | 0.9555 | `max_depth=20`, `min_samples_leaf=3`, `min_samples_split=2`, `criterion='entropy'` |
| **3** | **SVM Tuned** | 64,684.36 | 0.7202 | **0.7181** | 0.7482 | 0.7113 | 0.9609 | `kernel='rbf'`, `C=0.3807`, `gamma=0.5089` |
| **4** | **Logistic Regression Tuned** | 178.97 | 0.7184 | 0.7176 | 0.7313 | 0.7088 | 0.9617 | `C=73.9227`, `penalty='l2'`, `solver='lbfgs'` |
| **5** | **KNN Tuned** | 972.58 | 1.0000 | 0.7078 | 0.7100 | 0.7081 | 0.9454 | `n_neighbors=12`, `metric='manhattan'`, `weights='distance'` |
| **6** | **Logistic Regression (Baseline)** | 36.95 | 0.7190 | 0.7178 | **0.7494** | 0.7060 | **0.9617** | Default (`max_iter=1000`) |
| **7** | **Decision Tree (Baseline)** | 2.96 | 1.0000 | 0.7050 | 0.7052 | 0.7051 | 0.8447 | Default (unconstrained depth) |
| **8** | **SVM (Baseline)** | 111.14 | 0.7188 | 0.7178 | 0.7289 | 0.6994 | NaN | `kernel='linear'`, `probability=False` |
| **9** | **K-Nearest Neighbors (Baseline)** | 0.35 | 0.7871 | 0.6819 | 0.6848 | 0.6809 | 0.9095 | Default (`n_neighbors=5`) |

* **Best Overall F1-Score**: `Random Forest Tuned` achieved the highest weighted F1-score (`0.712359`) and strong generalization (`ROC-AUC = 0.9613`) in `826.37s`.
* **Best Efficiency-to-Performance Ratio**: `Decision Tree Tuned` achieved virtually identical weighted F1 (`0.712064`) and higher test accuracy (`71.63%`) in just **27.88 seconds**, while constraining `max_depth=20` and `min_samples_leaf=3` eliminated the severe overfitting of the baseline tree (`100.00%` train accuracy vs. `70.50%` test accuracy) and improved ROC-AUC from `0.8447` to `0.9555`.
* **Diminishing Returns of Kernel SVM**: `SVM Tuned` (RBF kernel) required **64,684.36 seconds (~17.97 hours)** for 90 CV fits—more than `2,300×` slower than `Decision Tree Tuned`—while yielding a lower weighted F1 (`0.711283`).

### Class-Level Performance Breakdown Across Models

Inspecting the test-set classification reports and training confusion matrix reveals three distinct performance regimes across the 21 classes:

| Performance Tier | Vehicle Classes (`Make_n_Model`) | Test F1 Range (Across Tuned Models) | Root Cause in Dataset |
| :--- | :--- | :---: | :--- |
| **Tier 1: Perfectly Separable (12 Classes)** | `DAF_LF`, `Hino_500 Series`, `Iveco_Daily`, `Iveco_Eurocargo`, `MAN_TGE`, `MAN_TGL`, `Mercedes_Atego`, `Mercedes_Sprinter`, `Renault_Master`, `Scania_S-Series V8`, `Volvo_FH16`, `Volvo_FL` | `0.98 – 1.00` | Unique combinations of static weight limits (`GVW`, `GCW`, `Capacity`) and/or dedicated `special_need` assignments (e.g., `Refuse` → `Volvo_FL`, `Timber` → `Scania_S-Series V8`). |
| **Tier 2: Moderately Confused Light/Medium Trucks (2 Classes)** | `Fuso_Canter`, `Isuzu_N-Series` | `0.66 – 0.80` | Overlapping light-truck payload ratings; in the training confusion matrix , `7,371` of `14,475` `Isuzu_N-Series` trips are mispredicted as `Fuso_Canter`. |
| **Tier 3: Heavily Overlapping 44t Heavy-Duty Trucks (7 Classes)** | `Iveco_S-Way` (`0.42–0.54`), `MAN_TGX` (`0.40–0.46`), `Mercedes_Actros` (`0.20–0.27`), `Volvo_FH` (`0.19–0.26`), `DAF_XG+` (`0.15–0.22`), `Renault_T High` (`0.00–0.17`), `Scania_R-Series` (`0.00–0.17`) | `0.00 – 0.54` | Identical static weight specs (`GCW = 44.0`, `GVW = 26.0`, `Capacity = 24.0`) and shared long-haul `special_need` categories (`Box`, `Refrigerated`, `Curtainsider`, `Container`). Linear/RBF models collapse `Scania_R-Series` and `Renault_T High` predictions into `Volvo_FH`, `Mercedes_Actros`, `DAF_XG+`, and `Iveco_S-Way`. |

### Feature Importance Analysis

Permutation importance on the test set (`5,000`-row subsample, `n_repeats=10`, `scoring='f1_weighted'`) ranked the `8` input features by mean F1 drop when shuffled:

| Rank | Original Feature | Permutation Importance (Mean F1 Drop) | Standard Deviation (`std`) |
| :---: | :--- | :---: | :---: |
| **1** | `total_distance` | 0.460367 | 0.008543 |
| **2** | `GVW` | 0.425812 | 0.004104 |
| **3** | `total_fuel` | 0.395061 | 0.007641 |
| **4** | `GCW` | 0.377863 | 0.005576 |
| **5** | `trip_duration` | 0.283885 | 0.005894 |
| **6** | `special_need` | 0.242254 | 0.004924 |
| **7** | `Capacity` | 0.238453 | 0.005588 |
| **8** | `axle_weight_average` | 0.089916 | 0.002271 |

![ Feature Importance plot ](code/images/Feature_Importance.png)

#### Next steps
What suggestions do you have for next steps?

1. **Incorporate Non-Leaking Geographic and Temporal Features**:
   * Add `Country` (one-hot or target encoded) and temporal features extracted from `trip_start_time` and `trip_end_time` (e.g., trip day-of-week, month/seasonality, and multi-day trip indicator) into `ColumnTransformer` to test whether regional fleet allocations help separate the 7 overlapping European 44-ton heavy-duty truck models (`DAF_XG+`, `Iveco_S-Way`, `MAN_TGX`, `Mercedes_Actros`, `Renault_T High`, `Scania_R-Series`, `Volvo_FH`) without leaking direct vehicle identifiers (`Group`, `wmi`, `plant`, `Serial Number`).
2. **Engineer Domain-Specific Load and  Efficiency Interaction Features**:
   * Introduce payload utilization and specific fuel consumption ratios inside `add_derived_features(X)` to capture mechanical differences among same-tonnage vehicles:
     * `load_utilization_ratio = axle_weight_average / Capacity`
     * `axle_to_gvw_ratio = axle_weight_average / GVW`
     * `specific_fuel_per_ton_km = fuel_per_km / axle_weight_average`
   * Prune collinear raw variables (`total_fuel`, `trip_duration`, `GCW`) for linear and distance-based models to mitigate multicollinearity ($r > 0.97$).
3. **Implement a Two-Stage Hierarchical Classifier for the 44-Ton Heavy-Duty Cluster**:
   * Because 12 classes already reach `1.00` F1-score and errors are concentrated in the 7-class heavy-duty cluster (`GCW = 44.0`, `GVW = 26.0`, `Capacity = 24.0`) plus the `Fuso_Canter` / `Isuzu_N-Series` pair, route records falling into the 44-ton segment to a dedicated second-stage gradient boosted classifier (`HistGradientBoostingClassifier`, `XGBoost`, or `LightGBM`) tuned specifically on subtle fuel-burn, speed, and axle-load distributions.


#### Outline of project

- [Link to notebook](code/Capstone_Project_M24_final.ipynb)
- [Link to extracted best model package](code/best_model_package.joblib)
- [Link to sample test data notebook](code/test_model.ipynb)
- [Link to datafile](code/data/synthetic_fleet_data_400k_splneeds.csv)

##### Contact and Further Information

**Rajeev Batra***

**Email**: batra.rajeev@gmail.com 

[LinkedIn](www.linkedin.com/in/rajeevbatra22)