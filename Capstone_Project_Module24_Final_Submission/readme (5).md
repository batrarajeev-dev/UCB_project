# Fleet Vehicle Make and Model Classification from Telematics Trip Data

## Executive Summary & Project Objective

This capstone project (`Capstone_Project_M24_final.ipynb`) develops and evaluates supervised multiclass machine learning pipelines to classify commercial fleet vehicles into **21 unique `Make_n_Model` categories** based on trip-level telematics, payload weight ratings, and specialized body/cargo requirements (`special_need`). Using a synthetic fleet dataset of **400,000 trip records** (`391,204` cleaned records after filtering), nine classification pipelines—spanning four baseline algorithms and five hyperparameter-tuned models via 3-fold `RandomizedSearchCV`—were benchmarked on accuracy, weighted Precision, weighted F1-score, One-vs-Rest (OvR) ROC-AUC, and computational training time.

Across all evaluated models, test accuracy converged to a narrow ceiling of **71.50% to 71.81%** (weighted F1-score of **0.7081 to 0.7124**). Detailed class-level confusion matrix analysis revealed that **12 of the 21 vehicle classes are classified with 1.00 F1-score**, whereas **7 European heavy-duty 44-ton GCW / 26-ton GVW truck models** (`DAF_XG+`, `Iveco_S-Way`, `MAN_TGX`, `Mercedes_Actros`, `Renault_T High`, `Scania_R-Series`, and `Volvo_FH`) share identical static weight limits and overlapping cargo profiles, creating an irreducible feature-overlap ceiling under the current 8-feature input set.

## Dataset Architecture & Preprocessing Pipeline

### Source Data Schema

The raw dataset (`data/synthetic_fleet_data_400k_splneeds.csv`) comprises `400,000` rows and `19` columns (`5` `float64`, `3` `int64`, and `11` `object` columns) with zero null entries:

| Column Name | Data Type | Role in Pipeline | Description |
| :--- | :---: | :---: | :--- |
| `trip_id` | `int64` | Dropped (Cell 8) | Unique identifier for each trip record |
| `trip_start_time` | `object` (ISO Date) | Converted to Date (Cell 9), Excluded | Timestamp when the trip begins |
| `trip_end_time` | `object` (ISO Date) | Converted to Date (Cell 9), Excluded | Timestamp when the trip ends |
| `trip_duration` | `int64` | Raw Numeric Feature | Total trip duration in seconds |
| `total_distance` | `float64` | Raw Numeric Feature | Total distance traveled in kilometers |
| `total_fuel` | `float64` | Raw Numeric Feature | Total fuel consumed during the trip in liters |
| `axle_weight_average` | `int64` | Raw Numeric Feature | Average load per axle during the trip (kg) |
| `Capacity` | `float64` | Raw Numeric Feature | Maximum cargo payload capacity (`1.5`, `2.5`, `4.0`, `6.0`, `8.0`, `9.0`, `10.0`, `24.0`, `40.0` tons) |
| `GCW` | `float64` | Raw Numeric Feature | Gross Combination Weight (vehicle + trailer + load, in tons) |
| `GVW` | `float64` | Raw Numeric Feature | Gross Vehicle Weight (maximum allowable vehicle weight, in tons) |
| `special_need` | `object` | Categorical Feature | Special vehicle body/cargo requirement (`10` active classes after filtering) |
| `manufacturer` | `object` | Target Component | Vehicle OEM/brand (combined into `Make_n_Model`) |
| `Model` | `object` | Target Component | Specific vehicle model (combined into `Make_n_Model`) |
| `Group` | `object` | Excluded | Vehicle classification code (directly encodes OEM/Model prefix) |
| `Serial Number` | `object` | Excluded | Manufacturer-assigned unique vehicle identifier |
| `plant` | `object` | Excluded | Facility or plant code where the vehicle operates |
| `Region` | `object` | Excluded | Geographic operating region (`Europe`) |
| `Country` | `object` | Excluded | Country of operation (e.g., Netherlands, France, Italy, Sweden, Spain, Germany, Belgium) |
| `wmi` | `object` | Excluded | World Manufacturer Identifier (first 3 VIN characters directly tied to OEM) |

### Data Cleaning & Target Engineering

1. **Identifier & Category Filtering (Cell 8)**: Dropped `trip_id` and removed `8,796` rows where `special_need.str.lower() == 'ambulance'`, leaving **391,204 rows** across **10 `special_need` categories**.
2. **Target Construction (Cells 10–11)**: Created the 21-class target column `Make_n_Model = manufacturer + '_' + Model` to prevent model-name collisions across OEMs:
   * `DAF_LF`, `DAF_XG+`, `Fuso_Canter`, `Hino_500 Series`, `Isuzu_N-Series`, `Iveco_Daily`, `Iveco_Eurocargo`, `Iveco_S-Way`, `MAN_TGE`, `MAN_TGL`, `MAN_TGX`, `Mercedes_Actros`, `Mercedes_Atego`, `Mercedes_Sprinter`, `Renault_Master`, `Renault_T High`, `Scania_R-Series`, `Scania_S-Series V8`, `Volvo_FH`, `Volvo_FH16`, `Volvo_FL`.
3. **Stratified Train-Test Split (Cell 43)**: Partitioned `X` (`391,204 × 8`) and `y` (`391,204`) using `train_test_split(test_size=0.20, shuffle=True, stratify=y, random_state=42)`, producing **312,963 training samples** and **78,241 test samples**.

### Automated Feature Engineering & Transformation (`preprocessor`)

Implemented a scikit-learn `Pipeline` (Cells 40–47) that dynamically constructs derived ratios inside cross-validation folds before scaling and encoding:

* **`FunctionTransformer(add_derived_features)`**:
  * `trip_duration_hours = trip_duration / 3600`
  * `fuel_per_km = total_fuel / total_distance` (when `total_distance > 0`, else `NaN`)
  * `avg_speed_kmph = total_distance / trip_duration_hours` (when `trip_duration_hours > 0`, else `NaN`)
* **`ColumnTransformer`**:
  * `StandardScaler()` applied to all `10` numeric features (`7` raw + `3` derived).
  * `OneHotEncoder(handle_unknown='ignore')` applied to `special_need` (`10` categories), yielding a final transformed matrix of shape **`(312963, 20)`**.

## Exploratory Data Analysis Findings

### Collinearity Across Operational and Weight Features

The numerical correlation matrix (Cell 20) highlights two tightly coupled feature clusters:

| Feature | `trip_duration` | `total_distance` | `total_fuel` | `axle_weight_average` | `GCW` | `GVW` | `Capacity` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`trip_duration`** | 1.0000 | 0.9824 | 0.9736 | 0.6432 | 0.6604 | 0.6825 | 0.6892 |
| **`total_distance`** | 0.9824 | 1.0000 | 0.9908 | 0.6921 | 0.7132 | 0.7359 | 0.7331 |
| **`total_fuel`** | 0.9736 | 0.9908 | 1.0000 | 0.6948 | 0.7146 | 0.7380 | 0.7405 |
| **`axle_weight_average`** | 0.6432 | 0.6921 | 0.6948 | 1.0000 | 0.8695 | 0.8896 | 0.8210 |
| **`GCW`** | 0.6604 | 0.7132 | 0.7146 | 0.8695 | 1.0000 | 0.9749 | 0.9567 |
| **`GVW`** | 0.6825 | 0.7359 | 0.7380 | 0.8896 | 0.9749 | 1.0000 | 0.9254 |
| **`Capacity`** | 0.6892 | 0.7331 | 0.7405 | 0.8210 | 0.9567 | 0.9254 | 1.0000 |

* **Operational Collinearity**: `total_distance` and `total_fuel` ($r = 0.9908$) as well as `trip_duration` and `total_distance` ($r = 0.9824$) move almost linearly with trip length.
* **Weight Specification Collinearity**: `GCW` and `GVW` ($r = 0.9749$) and `GCW` and `Capacity` ($r = 0.9567$) reflect discrete vehicle tonnage tiers.

### Distribution of `special_need` Across Target Classes

Grouping `Make_n_Model` by `special_need` (Cell 32) explains why some vehicle classes are trivial to classify while others overlap heavily:

| `special_need` Category | Total Records | Unique `Make_n_Model` Classes | Most Frequent Class (`top`) | Top Class Frequency (`freq`) |
| :--- | :---: | :---: | :--- | :---: |
| **Box** | 91,111 | 10 | `Mercedes_Atego` | 10,482 |
| **Refrigerated** | 87,065 | 10 | `Mercedes_Actros` | 11,260 |
| **Curtainsider** | 79,073 | 9 | `Volvo_FH` | 10,447 |
| **Panel Van** | 29,305 | 3 | `Mercedes_Sprinter` | 10,526 |
| **Heavy Haulage** | 28,387 | 2 | `Volvo_FH16` | 20,667 |
| **Tipper** | 20,947 | 3 | `Iveco_Eurocargo` | 7,612 |
| **Flatbed** | 20,121 | 2 | `Hino_500 Series` | 11,238 |
| **Container** | 16,091 | 2 | `Iveco_S-Way` | 9,592 |
| **Refuse** | 9,738 | 1 | `Volvo_FL` | 9,738 |
| **Timber** | 9,366 | 1 | `Scania_S-Series V8` | 9,366 |

## Research Results: What the Analysis Found

### Baseline vs. Hyperparameter-Tuned Model Performance

Four baseline classifiers (Cells 49–52) and five tuned classifiers using 3-fold `RandomizedSearchCV` (`n_iter=30`, `scoring='f1_weighted'`, `random_state=42`, Cells 62–65) were evaluated on the `78,241`-row test set:

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

Inspecting the test-set classification reports (Cell 64) and training confusion matrix (Cell 67) reveals three distinct performance regimes across the 21 classes:

| Performance Tier | Vehicle Classes (`Make_n_Model`) | Test F1 Range (Across Tuned Models) | Root Cause in Dataset |
| :--- | :--- | :---: | :--- |
| **Tier 1: Perfectly Separable (12 Classes)** | `DAF_LF`, `Hino_500 Series`, `Iveco_Daily`, `Iveco_Eurocargo`, `MAN_TGE`, `MAN_TGL`, `Mercedes_Atego`, `Mercedes_Sprinter`, `Renault_Master`, `Scania_S-Series V8`, `Volvo_FH16`, `Volvo_FL` | `0.98 – 1.00` | Unique combinations of static weight limits (`GVW`, `GCW`, `Capacity`) and/or dedicated `special_need` assignments (e.g., `Refuse` → `Volvo_FL`, `Timber` → `Scania_S-Series V8`). |
| **Tier 2: Moderately Confused Light/Medium Trucks (2 Classes)** | `Fuso_Canter`, `Isuzu_N-Series` | `0.66 – 0.80` | Overlapping light-truck payload ratings; in the training confusion matrix (Cell 67), `7,371` of `14,475` `Isuzu_N-Series` trips are mispredicted as `Fuso_Canter`. |
| **Tier 3: Heavily Overlapping 44t Heavy-Duty Trucks (7 Classes)** | `Iveco_S-Way` (`0.42–0.54`), `MAN_TGX` (`0.40–0.46`), `Mercedes_Actros` (`0.20–0.27`), `Volvo_FH` (`0.19–0.26`), `DAF_XG+` (`0.15–0.22`), `Renault_T High` (`0.00–0.17`), `Scania_R-Series` (`0.00–0.17`) | `0.00 – 0.54` | Identical static weight specs (`GCW = 44.0`, `GVW = 26.0`, `Capacity = 24.0`) and shared long-haul `special_need` categories (`Box`, `Refrigerated`, `Curtainsider`, `Container`). Linear/RBF models collapse `Scania_R-Series` and `Renault_T High` predictions into `Volvo_FH`, `Mercedes_Actros`, `DAF_XG+`, and `Iveco_S-Way`. |

### Feature Importance Analysis

Permutation importance on the test set (`5,000`-row subsample, `n_repeats=10`, `scoring='f1_weighted'`, Cell 70) ranked the `8` input features by mean F1 drop when shuffled:

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

## Fixing the Best-Model Selection Bug in the Training Loop

### Root Cause of the Bug

Across **Cells 56, 64, 65, 67, 70, and 73**, three state-management issues caused the wrong model and wrong model name to be evaluated and exported to `best_model_package.joblib`:

1. **Cell 56** assigns `best_model_name = "Logistic Regression"` based only on the four untuned baseline models.
2. **Cell 64** overwrites `best_model = random_search.best_estimator_` inside every iteration of `for name, base_model in models.items():`. Because `'SVM Tuned'` is the last key in the `models` dictionary, `best_model` always ends up as `'SVM Tuned'` (`F1 = 0.7113`, `Train Time = 64,684.36s`) after the loop completes, rather than `'Random Forest Tuned'` (`F1 = 0.7124`).
3. **Cell 73** exports `{"model": best_model, "model_name": best_model_name, ...}`, pairing the `'SVM Tuned'` pipeline object with the stale string `"Logistic Regression"` from Cell 56.

### Exact Cell Modifications

#### 1. Update Cell 65 (Immediately After `RandomizedSearchCV` Finishes)

Replace the contents of **Cell 65** with the following code so `best_model_name`, `best_model`, and `roc_auc` are dynamically selected from the top-ranked model in `results_df` (using the `best_estimators` dictionary already populated in Cell 64):

```python
# Convert the results dictionary into a DataFrame sorted by highest Test F1 Score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

# Select the true best model and its metadata based on the highest F1 Score
best_model_name = results_df.iloc[0]["Model"]
best_model = (
    best_estimators[best_model_name]
    if best_model_name in best_estimators
    else pipelines[best_model_name]
)
roc_auc = results_df.iloc[0]["ROC-AUC"]

print(f"Selected Best Model: {best_model_name} (Test F1 = {results_df.iloc[0]['F1 Score']:.6f}, ROC-AUC = {roc_auc:.6f})")
results_df
```

#### 2. Verify Downstream Cells (Cells 67, 70, and 73)

Once **Cell 65** is updated as shown above:
* **Cell 67** (`y_train_pred = best_model.predict(X_train)`) will automatically evaluate `'Random Forest Tuned'` on the training set and log its true `roc_auc` (`0.961298`).
* **Cell 70** (`classifier = best_model.named_steps["classifier"]`) will automatically detect `hasattr(classifier, "feature_importances_") == True` on `RandomForestClassifier`, computing fast model-native Gini feature importances instead of falling back to slow permutation importance on `SVC`.
* **Cell 73** (`joblib.dump(model_package, "best_model_package.joblib")`) will serialize `'Random Forest Tuned'` alongside `model_name = "Random Forest Tuned"`.

## Next Steps & Recommendations

1. **Incorporate Non-Leaking Geographic and Temporal Features**:
   * Add `Country` (one-hot or target encoded) and temporal features extracted from `trip_start_time` and `trip_end_time` (e.g., trip day-of-week, month/seasonality, and multi-day trip indicator) into `ColumnTransformer` to test whether regional fleet allocations help separate the 7 overlapping European 44-ton heavy-duty truck models (`DAF_XG+`, `Iveco_S-Way`, `MAN_TGX`, `Mercedes_Actros`, `Renault_T High`, `Scania_R-Series`, `Volvo_FH`) without leaking direct vehicle identifiers (`Group`, `wmi`, `plant`, `Serial Number`).
2. **Engineer Domain-Specific Load & Efficiency Interaction Features**:
   * Introduce payload utilization and specific fuel consumption ratios inside `add_derived_features(X)` to capture mechanical differences among same-tonnage vehicles:
     * `load_utilization_ratio = axle_weight_average / Capacity`
     * `axle_to_gvw_ratio = axle_weight_average / GVW`
     * `specific_fuel_per_ton_km = fuel_per_km / axle_weight_average`
   * Prune collinear raw variables (`total_fuel`, `trip_duration`, `GCW`) for linear and distance-based models to mitigate multicollinearity ($r > 0.97$).
3. **Implement a Two-Stage Hierarchical Classifier for the 44-Ton Heavy-Duty Cluster**:
   * Because 12 classes already reach `1.00` F1-score and errors are concentrated in the 7-class heavy-duty cluster (`GCW = 44.0`, `GVW = 26.0`, `Capacity = 24.0`) plus the `Fuso_Canter` / `Isuzu_N-Series` pair, route records falling into the 44-ton segment to a dedicated second-stage gradient boosted classifier (`HistGradientBoostingClassifier`, `XGBoost`, or `LightGBM`) tuned specifically on subtle fuel-burn, speed, and axle-load distributions.
4. **Optimize Deployment Footprint and Inference Latency**:
   * Replace `SVM Tuned` (`64,684s` training time and $O(n_{\text{SV}} \cdot d)$ inference cost) in production with either `Decision Tree Tuned` (`F1 = 0.7121`, `27.88s` training time, sub-millisecond inference) or `Random Forest Tuned` (`F1 = 0.7124`, `826.37s` training time) after applying the Cell 65 selection fix.
