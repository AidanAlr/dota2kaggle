# Dota 2 Match Outcome Prediction

Predicting Dota 2 match winners from mid-game statistics using feature engineering and logistic regression. Built for a Kaggle competition.

## Results

| Metric | Score |
|---|---|
| Cross-Validation ROC AUC | **0.8389** |
| Hold-out Validation ROC AUC | 0.8287 |
| Features Used | 89 / 128 |

## Approach

### Feature Engineering

Raw per-player stats (kills, gold, XP, etc.) are transformed into **lead features** — the difference between each Radiant and Dire player at the same position. This reframes the problem from "what are the raw stats?" to "who has the advantage?".

Additional engineered features:
- **Team-level aggregates** — total gold/XP leads across all 5 players
- **Carry concentration** — what % of total gold the carry (position 1) holds
- **Hero win rates** — historical win rate for each hero, computed from training data only to avoid leakage

### Model Selection

Logistic Regression with L1 (Lasso) penalty was selected over Random Forest for its interpretability and comparable performance. Key hyperparameters found via `RandomizedSearchCV`:

```
C=0.1, penalty=l1, solver=liblinear, class_weight=balanced
```

### Feature Selection

1. Rank all 128 features by Random Forest Gini importance
2. Sweep feature counts (5–150) evaluating each with 3-fold CV
3. Select the count with the highest CV score (89 features)
4. Optimize hyperparameters only on the winning feature set

### Top Predictive Features

| Feature | Importance |
|---|---|
| `radiant_team_gold_lead` | 0.092 |
| `radiant_team_xp_lead` | 0.067 |
| `radiant_carries_gold_lead` | 0.027 |
| `radiant_carries_xp_lead` | 0.019 |
| Individual player gold leads | 0.013–0.016 |

Team-level economic advantages are far more predictive than individual hero stats.

## Project Structure

```
├── Functional.ipynb          # Full pipeline: preprocessing, feature engineering, training, evaluation
├── data/
│   ├── train_data.CSV        # 29,675 matches, 307 raw features
│   └── test_data.CSV         # 10,000 matches for submission
├── submissions/              # Kaggle submission CSVs
└── outputs/                  # Feature importance rankings per scaler
```

## Running

```bash
pip install pandas numpy scikit-learn
jupyter notebook Functional.ipynb
```

Place `train_data.CSV` and `test_data.CSV` in the `data/` directory before running.
