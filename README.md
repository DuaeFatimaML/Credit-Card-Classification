💳 Credit Card Approval Prediction using Random Forest

Predicting credit card approval decisions using an advanced machine learning pipeline — featuring GridSearchCV hyperparameter tuning, SMOTE for class imbalance, engineered features, and a custom threshold for real-world cautious decision making.





🎯 Problem Statement
Credit card approval is a high-stakes decision affecting both financial institutions and individuals. Manual review processes are slow and inconsistent. This project automates approval prediction using personal financial and demographic data — with special attention to class imbalance, hyperparameter optimization, and cautious real-world decision making.

🚀 Live Demo
PlatformLink🔴 Streamlit App[Your Streamlit Link Here]🤗 Hugging Face Space[Your HF Link Here]

📊 Model Performance
Threshold: 0.35
ClassPrecisionRecallF1-ScoreSupport0 — Approved0.940.930.942751 — Rejected0.500.510.5135Accuracy0.89310Macro Avg0.720.720.72310Weighted Avg0.890.890.89310

🧠 Technical Implementation
Full Pipeline Architecture
Two CSV Files (features + labels)
        ↓
Merge on ind_id
        ↓
Outlier Removal (annual_income capped at 95th percentile)
        ↓
Feature Engineering
(age, years_employed, inc_per_family)
        ↓
Drop irrelevant columns
(mobile_phone, work_phone, phone, email_id, birthday_count, employed_days)
        ↓
Train/Test Split (80/20, stratified)
        ↓
ImbPipeline
   ├── ColumnTransformer
   │     ├── Numeric  → SimpleImputer(median) → StandardScaler
   │     └── Categorical → SimpleImputer(most_frequent) → OneHotEncoder
   ├── SMOTE (balances class imbalance on training folds only)
   └── RandomForestClassifier (class_weight='balanced')
        ↓
GridSearchCV (StratifiedKFold, 5 splits, scoring='f1')
        ↓
Best Model → Custom Threshold (0.35)
        ↓
Final Prediction

🔧 Feature Engineering
Raw columns were transformed into more meaningful features:
Engineered FeatureFormulaReasonageabs(birthday_count / 365)Convert days to readable ageyears_employedabs(employed_days / 365) if negativeNegative days = currently employedinc_per_familyannual_income / family_membersIncome per person — better financial indicator

🧹 Data Preprocessing
StepMethodReasonOutlier removalCap annual_income at 95th percentileExtreme incomes skew the modelNumeric imputationSimpleImputer(median)Robust to outliersCategorical imputationSimpleImputer(most_frequent)Fills missing categories naturallyNumeric scalingStandardScalerNormalizes feature rangesCategorical encodingOneHotEncoder(handle_unknown='ignore')Handles unseen categories safely

⚖️ Handling Class Imbalance
The dataset has severe class imbalance:
Approved (Class 0): 275 samples  — majority
Rejected (Class 1):  35 samples  — minority  (~11%)
Two strategies applied simultaneously:
1. SMOTE (Synthetic Minority Oversampling Technique)

Generates synthetic rejected cases during training
Applied inside the pipeline — only on training folds, never on test data
Prevents data leakage from oversampling

2. class_weight='balanced'

Automatically penalizes the model more for missing rejected cases
Works alongside SMOTE for double protection against bias


⚙️ Hyperparameter Tuning — GridSearchCV
Model was optimized using GridSearchCV with StratifiedKFold (5 splits):
ParameterValues Testedn_estimators100, 200max_depthNone, 10, 20min_samples_split2, 5

Scoring metric: F1 (best for imbalanced datasets — balances precision and recall)
Total combinations tested: 12 parameter combinations × 5 folds = 60 fits
Best parameters automatically selected and used for final prediction


🎯 Custom Threshold (0.35)
Default classification threshold is 0.50 — but this creates bias toward the majority class on imbalanced data.
Default threshold 0.50 → biased toward approving everyone
Custom threshold 0.35  → flags rejection if model is 35% confident
Why 0.35 specifically?
In credit card approval, missing a genuine rejection (False Negative) is more costly than a false alarm. A lower threshold makes the model more cautious — better aligned with how real financial institutions operate.

📋 Input Features
FeatureTypeProcessingannual_incomeNumericOutlier capped + StandardScalerageEngineeredFrom birthday_countyears_employedEngineeredFrom employed_daysinc_per_familyEngineeredincome / family_membersfamily_membersNumericStandardScalergenderCategoricalOneHotEncoderincome_typeCategoricalOneHotEncodereducation_typeCategoricalOneHotEncoderfamily_statusCategoricalOneHotEncoderhousing_typeCategoricalOneHotEncoderoccupation_typeCategoricalOneHotEncoderowned_carCategoricalOneHotEncoderowned_propertyCategoricalOneHotEncoder
Dropped columns: mobile_phone, work_phone, phone, email_id, birthday_count, employed_days — either redundant or replaced by engineered features.

📈 Visualization

✅ Actual vs Predicted comparison
✅ Feature importance chart — which factors influence approval most
✅ Class distribution before and after SMOTE


🛠️ Tech Stack
ToolPurposePython 3.11Core languagescikit-learnPipeline, RF, GridSearchCV, metricsimbalanced-learnSMOTE, ImbPipelinepandasData loading, merging, manipulationnumpyFeature engineeringjoblibModel & feature serializationStreamlitWeb app deployment

📂 Dataset

Source: Kaggle — Credit Card Approval Dataset
Files: Credit_card.csv + Credit_card_label.csv (merged on ind_id)
Class Distribution: ~88.7% Approved | ~11.3% Rejected


⚠️ Model Limitations

Class imbalance limits performance on minority class (rejected cases) despite SMOTE
Trained on Kaggle data — real bank approval criteria may differ significantly
Does not incorporate credit score or transaction history
Threshold of 0.35 was optimized for this dataset — may need adjustment for other distributions


🔮 Future Improvements

 Try XGBoost / LightGBM for comparison
 Add SHAP values for per-prediction explainability
 Collect more rejected case samples to reduce imbalance naturally
 Integrate credit score as a direct feature
 Build REST API with FastAPI for production deployment


👨‍💻 Author
[Your Name]
