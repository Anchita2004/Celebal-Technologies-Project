# Credit Scoring with Random Forest

Predicting loan applicant creditworthiness on the UCI German Credit dataset (1,000 applicants), with evaluation focused on the real-world cost of missed defaulters rather than accuracy alone.

**Tech stack:** Python · pandas · scikit-learn · matplotlib · seaborn

---

## Overview

Lenders need to judge whether an applicant is likely to default before approving credit. This project trains a Random Forest classifier to label each applicant as a **good** or **bad** credit risk using demographic and financial features, then evaluates it with the asymmetry of credit decisions in mind: approving a bad-risk applicant (false negative) usually costs far more than flagging a good applicant for extra review (false positive).

The pipeline covers preprocessing, model training, holdout and cross-validated evaluation, and a discussion of limitations and next steps.

## Key Results

| Metric | Value |
|---|---|
| Holdout accuracy (80/20 split) | **80.5%** |
| 5-fold cross-validation accuracy (mean) | **76%** |

**Per-class performance (test set, 200 applicants):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 1 – Good risk | 0.82 | 0.93 | 0.87 | 141 |
| 2 – Bad risk | 0.75 | 0.51 | 0.61 | 59 |

**Takeaways**

- Overall accuracy looks healthy, but the model catches only **51% of actual bad-risk applicants**. In lending, this recall on the bad class is the metric that matters most.
- The main driver is **class imbalance** (70% good / 30% bad). With no balancing applied, the model leans toward predicting the majority class.
- The gap between holdout accuracy (80.5%) and cross-validation accuracy (76%) suggests a single 80/20 split is slightly optimistic. Cross-validation is the more reliable estimate of real-world performance.

## Dataset

- **Source:** [UCI Machine Learning Repository, Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Size:** 1,000 applicants, 20 features plus 1 target (21 columns)
- **Features:** a mix of categorical and numerical attributes, such as credit history, loan purpose, credit amount, employment duration, age and housing status
- **Target:**
  - `1` = Good credit risk (700 applicants)
  - `2` = Bad credit risk (300 applicants)

> The 70/30 target split is imbalanced, which directly affects model behavior (see [Results](#key-results) and [Limitations & Next Steps](#limitations--next-steps)).

## Methodology

1. **Data check:** confirmed there are no missing values across all 21 columns.
2. **Encoding:** converted categorical variables to numeric using `LabelEncoder`.
3. **Feature/target split:** separated features (`X`) from the target (`y`).
4. **Train/test split:** 80/20 with `random_state=42`.
5. **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`, with no class balancing in this baseline version.
6. **Evaluation:** holdout accuracy, per-class precision/recall/F1, and 5-fold cross-validation (`cross_val_score`).

## Business Application

Rather than fully automating approve/reject decisions, a lender could use the model's **predicted probability** (not the hard label) to route applicants below a chosen risk threshold to manual underwriting. This hybrid approach limits automation risk while still reducing manual review volume for clearly low-risk applicants.

## Limitations & Next Steps

This is a solid baseline with a known, clearly stated weakness: **weak recall on bad-risk applicants**. Planned improvements:

- [ ] **Address class imbalance:** retrain with `class_weight='balanced'`, and try SMOTE oversampling of the minority class
- [ ] **Change the primary metric:** evaluate on bad-class recall/F1 or ROC-AUC instead of accuracy, which is misleading on imbalanced data
- [ ] **Tune the decision threshold:** lower the 0.5 default for the bad class to trade some precision for higher recall, which is likely the right tradeoff for a lender
- [ ] **Hyperparameter tuning:** `GridSearchCV` / `RandomizedSearchCV` over `n_estimators`, `max_depth` and `min_samples_leaf`
- [ ] **Revisit encoding:** `LabelEncoder` imposes an artificial order on nominal categories, so test one-hot encoding

## Getting Started

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/Anchita2004/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# 3. Add the dataset
# Download german.data from the UCI link above (space-separated, no header)
# and place it in the project root.

# 4. Run
python credit_scoring.py
```

## Author

**Anchita Majumdar** · [GitHub](https://github.com/Anchita2004)
