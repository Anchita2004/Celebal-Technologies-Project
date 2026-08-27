## Credit Scoring with Random Forest

This project uses the German Credit Dataset (UCI Machine Learning Repository) to build a machine learning model that predicts an individual's creditworthiness using a Random Forest Classifier. It covers preprocessing, training, evaluation, and interpretability analysis.

📊 Problem Statement

Financial institutions need to assess the risk of loan applicants defaulting before extending credit. This project predicts whether an applicant is a good or bad credit risk using demographic and financial features, and evaluates the model with a focus on the cost-asymmetric nature of credit decisions (misclassifying a bad-risk applicant as good is more costly than the reverse).

🧠 Model: Random Forest

Random Forest is an ensemble learning method that builds multiple decision trees on bootstrapped samples and merges their predictions for more accurate and stable results than a single decision tree.

📁 Dataset
Source: UCI Machine Learning Repository — German Credit Data
Rows: 1,000 applicants
Features: 20 attributes (categorical and numerical) — e.g. credit history, loan purpose, credit amount, employment duration, age, housing status
Target:
1: Good credit risk (700 applicants)
2: Bad credit risk (300 applicants)
Note: the target is imbalanced (70/30 split), which directly affects model behavior — see Results and Limitations below.
🧹 Preprocessing
Checked for and confirmed no missing values across all 21 columns
Encoded categorical variables using LabelEncoder
Split data into features (X) and target (y)
Train/test split: 80/20 (random_state=42)
Trained RandomForestClassifier(n_estimators=100, random_state=42) — no class-balancing applied in this version (see Next Steps)
📊 Results

Holdout accuracy: 80.5% 5-fold cross-validation mean accuracy: 76%

Class	Precision	Recall	F1-score	Support
1 – Good	0.82	0.93	0.87	141
2 – Bad	0.75	0.51	0.61	59

Key takeaway: overall accuracy looks solid, but the model only catches 51% of actual bad-risk applicants (recall). In a lending context this is the metric that matters most — a missed bad-risk applicant (false negative) is typically far costlier than a good applicant flagged for extra review (false positive). The 70/30 class imbalance in the training data is the main driver of this gap: the model defaults toward predicting the majority class ("Good").

The gap between holdout accuracy (80.5%) and cross-validation mean (76%) also suggests the single 80/20 split is slightly optimistic — cross-validation is the more reliable estimate of real-world performance.

Limitations & Next Steps

This model is a solid baseline but has a clear, known limitation worth stating openly:

Class imbalance is unaddressed. With only 300 "Bad" examples vs. 700 "Good," the model under-predicts the minority class. Planned fixes:
Retrain with class_weight='balanced' in RandomForestClassifier
Try oversampling the minority class with SMOTE
Re-evaluate using recall/F1 on the "Bad" class or ROC-AUC as the primary metric instead of accuracy, since accuracy is misleading on imbalanced data
No hyperparameter tuning yet — a GridSearchCV or RandomizedSearchCV pass over n_estimators, max_depth, and min_samples_leaf would likely improve minority-class recall further
No probability threshold tuning — by default the model uses a 0.5 decision threshold; lowering it for the "Bad" class would trade some precision for meaningfully higher recall, which may be the right tradeoff for a lender
💼 Business Application

A lender could use this model's predicted probability (rather than the hard class label) to flag applicants below a chosen risk threshold for manual underwriting review, rather than fully automating approve/reject decisions — this hybrid approach limits automation risk while still reducing manual review volume for clearly low-risk applicants.

🖥️ How to Run
bash
pip install pandas scikit-learn matplotlib seaborn
python credit_scoring.py

Requires german.data (UCI German Credit Data, space-separated, no header) in the working directory.

🛠️ Tools

Python, pandas, scikit-learn (RandomForestClassifier, LabelEncoder, cross_val_score), matplotlib, seaborn
