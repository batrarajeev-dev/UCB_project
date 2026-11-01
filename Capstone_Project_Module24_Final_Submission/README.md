### Project Title
#### Intelligent Fleet Recommendation and  Optimization System 
(for European Commercial Fleet Telemetry Analysis and Vehicle Classification)

**Author**
Rajeev Batra

#### Executive summary

 Globally over 70% of the freight moves on road and the fleet management is estimated to be a 30 billion $ business with 18% CAGR.  However, fleet management is a very thrifty business as it relies on razor thin margins and seasonality. A fleet is almost always made up of vehicles from different manufacturers, models and configurations. This ads up to the problem of selecting the right vehicle for a given mission considering multiple of parameters such as load, distance, destination etc. 
 The aim of this project is to build a vehicle recommendation and assignment model for commercial vehicle fleets to ensure that correct vehicle is assigned to correct mission as it can help the fleet owner save between 2~3% of the operational cost and this is a direct addition to the bottom line. 

#### Rationale
Why should anyone care about this question?

Commercial logistics and freight operators face severe financial and environmental losses from suboptimal vehicle dispatching:

-   **Widespread Payload Under-Utilization ("Ghost Loads")**: Dispatching oversized heavy-duty trucks for light loads wastes fuel, inflates per-kilometer operating costs, and increases carbon emissions.
-   **Severe Overload and  Safety Risks**: Conversely, assigning an undersized vehicle whose rated capacity falls below the actual cargo load ($C_{\text{pred}} < C_{\text{actual}}$) causes accelerated mechanical wear, braking hazards, regulatory fines, and unscheduled maintenance downtime.
-   **Operational Complexity at Scale**: Dispatchers must manually balance physical cargo weight, route distance, regional terrain, and specialized cargo constraints (such as refrigerated reefers or hazardous material containment) across dozens of competing vehicle models. Automating this decision with a cost-sensitive machine learning recommender enables consistent fleet right-sizing, reduces fuel burn, and protects fleet assets.

#### Research Question
What are you trying to answer?

 > **How can we accurately recommend the optimal commercial vehicle (`Make_n_Model`) for a given logistics trip—using pre-dispatch route, payload weight, and specialized cargo attributes—to maximize payload utilization and minimize excess fuel consumption without risking vehicle overload? and to minimize under-utilization and reduce trip operating costs by 2–3% and this is big savings given the razor thin  margins in fleet management**

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

To quantify fleet efficiency prior to modeling, six domain-specific operational metrics were engineered during EDA:
* **Fuel Consumption Rate (`fuel_per_100km`)**: $\frac{\text{total\_fuel}}{\text{total\_distance}} \times 100$
* **Average Trip Speed (`average_speed_kmh`)**: $\frac{\text{total\_distance}}{\text{trip\_duration} / 60}$
* **Load Factor Proxy (`load_factor_proxy`)**: $\frac{\text{axle\_weight\_average}}{\text{Capacity}}$
* **Fuel Economy (`Distance_per_Litre`)**: $\frac{\text{total\_distance}}{\text{total\_fuel}}$
* **Speed Efficiency (`Speed_Efficiency`)**: Ratio of actual trip speed to expected highway baseline
* **Payload Utilization Proxy (`Payload_Utilization_Proxy`)**: $\frac{\text{axle\_weight\_average}}{\text{GVW} \times 1000}$

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

#### Next steps
What suggestions do you have for next steps?

#### Outline of project

- [Link to notebook](code/Capstone_Project_M24_final.ipynb)
- [Link to extracted best model package](code/best_model_package.joblib)
- [Link to sample test data notebook](code/test_model.ipynb)
- [Link to datefile](code/data/synthetic_fleet_data_400k_splneeds.csv)

##### Contact and Further Information

**Rajeev Batra***

**Email**: batra.rajeev@gmail.com 

[LinkedIn](www.linkedin.com/in/rajeevbatra22)