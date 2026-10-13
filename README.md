# Assessment 3 — Objective 2: Linear Regression (FIFA World Cup 2026)

Run `WorldCup2026_Linear_Regression.ipynb` top to bottom. It does everything in Python:

* **Part A, wrangling**: `Players.xlsx` + `team_info.csv` → `dataset_2_1_match_goal_difference.csv` (104 rows) and `dataset_2_2_team_goals.csv` (208 rows)
* **Part B, models**: OLS fit, 80/20 hold-out, repeated 5-fold CV, mean baseline, VIF, residual diagnostics, figures

The same code is also in `build_datasets.py` and `regression_models.py`.

| File | Contents |
|---|---|
| `Players.xlsx` | FBref player-by-match stats for all 104 matches (from Assessment 2) |
| `team_info.csv` | FIFA ranking (11 June 2026), World Cup appearances before 2026, host flag |
| `model_summary.csv`, `coefficients_2_*.csv` | Model outputs used in the report |
| `fig_*.png` | Report figures |

Requirements: pandas, numpy, matplotlib, scipy, statsmodels, scikit-learn, openpyxl.
