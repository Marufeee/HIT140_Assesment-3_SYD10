# %% [markdown]
# ## Part B: linear regression models
# Both models use ordinary least squares (statsmodels for inference, scikit-learn for out-of-sample evaluation).
#
# **Evaluation design** (identical for both models, so they can be compared):
# 1. Fit on all rows to read coefficients, p-values, R² and adjusted R².
# 2. Hold-out test: 80/20 random split (seed 42), fit on 80 %, score on the unseen 20 %.
# 3. Repeated 5-fold cross-validation (10 repeats = 50 fits), so results don't hinge on one lucky split.
# 4. Every score is compared with a **baseline that always predicts the training mean** — a model is only
#    useful if it beats this.
# 5. Diagnostics: variance inflation factors (multicollinearity), residual plots, Q-Q plot.

# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

d1 = pd.read_csv("dataset_2_1_match_goal_difference.csv")
d2 = pd.read_csv("dataset_2_2_team_goals.csv")
X1_COLS = ["rank_strength_diff", "host_diff", "rest_diff", "knockout",
           "form_gd_diff", "form_points_diff", "form_sot_diff", "wc_experience_diff"]
X2_COLS = ["own_rank_strength", "host", "rest_days", "knockout",
           "opp_rank_strength", "form_goals_for", "opp_form_goals_against", "squad_age"]
print("2.1:", d1.shape, " 2.2:", d2.shape)
print("explanatory variables:", len(X1_COLS), len(X2_COLS))
print("shared concepts: FIFA ranking strength, host nation, rest days, knockout stage  (4 = maximum allowed)")


# %%
def evaluate(df, xcols, ycol, name):
    X, y = df[xcols], df[ycol]
    out = {"model": name}

    # 1. full-data OLS for inference
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    out["ols"] = ols
    out["R2_full"], out["adjR2_full"] = ols.rsquared, ols.rsquared_adj

    # 2. hold-out test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    lr = LinearRegression().fit(Xtr, ytr)
    pred = lr.predict(Xte)
    base = np.full(len(yte), ytr.mean())
    out.update(test_R2=r2_score(yte, pred), test_RMSE=mean_squared_error(yte, pred) ** 0.5,
               test_MAE=mean_absolute_error(yte, pred), base_RMSE=mean_squared_error(yte, base) ** 0.5,
               base_MAE=mean_absolute_error(yte, base), n_train=len(ytr), n_test=len(yte))

    # 3. repeated 5-fold CV (model vs mean baseline on identical folds)
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    cv = cross_validate(LinearRegression(), X, y, cv=rkf,
                        scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
    base_rmse = [np.sqrt(np.mean((y.iloc[te] - y.iloc[tr].mean()) ** 2)) for tr, te in rkf.split(X)]
    out.update(cv_R2=cv["test_r2"].mean(), cv_R2_sd=cv["test_r2"].std(),
               cv_RMSE=-cv["test_neg_root_mean_squared_error"].mean(),
               cv_MAE=-cv["test_neg_mean_absolute_error"].mean(), cv_base_RMSE=np.mean(base_rmse))

    # 5. diagnostics
    Xc = sm.add_constant(X)
    out["vif"] = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=xcols)
    out["bp_p"] = het_breuschpagan(ols.resid, Xc)[1]
    out["shapiro_p"] = stats.shapiro(ols.resid)[1]
    out["fitted"], out["resid"], out["y"] = ols.fittedvalues, ols.resid, y
    return out


def coef_table(res, X):
    ols = res["ols"]
    t = pd.DataFrame({"coef": ols.params, "std_err": ols.bse, "t": ols.tvalues, "p": ols.pvalues})
    sd = X.std()
    t["std_coef"] = (ols.params.drop("const") * sd / res["y"].std()).reindex(t.index)   # standardised beta
    return t.round(3)


def summary_row(r):
    keys = ["R2_full", "adjR2_full", "test_R2", "test_RMSE", "base_RMSE", "test_MAE", "base_MAE",
            "cv_R2", "cv_R2_sd", "cv_RMSE", "cv_base_RMSE", "cv_MAE", "bp_p", "shapiro_p"]
    return pd.Series({k: r[k] for k in keys}, name=r["model"]).round(3)


def diagnostics_plot(res, title, fname):
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    ax[0].scatter(res["fitted"], res["y"], s=12, alpha=.6)
    lim = [min(res["fitted"].min(), res["y"].min()) - .5, max(res["fitted"].max(), res["y"].max()) + .5]
    ax[0].plot(lim, lim, "r--", lw=1); ax[0].set(xlabel="Predicted", ylabel="Actual", title="Actual vs predicted")
    ax[1].scatter(res["fitted"], res["resid"], s=12, alpha=.6); ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")
    stats.probplot(res["resid"], plot=ax[2]); ax[2].set_title("Normal Q-Q of residuals")
    ax[2].get_lines()[0].set_markersize(3)
    fig.suptitle(title, fontsize=10, y=1.02); fig.tight_layout(); fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()


# %% [markdown]
# ### Linear Regression 2.1 — goal difference (Team A − Team B), 104 matches
# **Explanatory variables (all known before kick-off):**
#
# | Variable | Meaning |
# |---|---|
# | `rank_strength_diff` | log(FIFA rank B) − log(FIFA rank A), ranking of 11 June 2026 |
# | `host_diff` | +1 if A is a host nation, −1 if B is, 0 otherwise |
# | `rest_diff` | rest days of A − rest days of B |
# | `knockout` | 1 if knockout match |
# | `form_gd_diff` | difference in goal difference per game in earlier 2026 matches |
# | `form_points_diff` | difference in points per game in earlier 2026 matches |
# | `form_sot_diff` | difference in shots on target per game in earlier 2026 matches |
# | `wc_experience_diff` | difference in World Cup appearances before 2026 |

# %%
print(d1[X1_COLS + ["goal_diff"]].describe().T.round(2))
print("\nCorrelation with goal_diff:")
print(d1[X1_COLS].corrwith(d1.goal_diff).round(3).sort_values(ascending=False))

# %%
r1 = evaluate(d1, X1_COLS, "goal_diff", "LR 2.1 goal difference")
print(r1["ols"].summary())
ct1 = coef_table(r1, d1[X1_COLS]); print(ct1)
print("\nVIF:\n", r1["vif"].round(2))

# %%
# Result-direction check: does the sign of the prediction match win / loss? (draws: |pred| < 0.5)
pred_res = np.select([r1["fitted"] > .5, r1["fitted"] < -.5], ["A win", "B win"], "Draw")
act_res = np.select([d1.goal_diff > 0, d1.goal_diff < 0], ["A win", "B win"], "Draw")
r1["outcome_acc"] = (pred_res == act_res).mean()
print("In-sample W/D/L accuracy:", round(r1["outcome_acc"], 3))
print(pd.crosstab(pd.Series(act_res, name="actual"), pd.Series(pred_res, name="predicted")))

# %%
diagnostics_plot(r1, "Model 2.1 — goal difference (Team A − Team B)", "fig_2_1_diagnostics.png")

# %% [markdown]
# ### Linear Regression 2.2 — goals scored by a team, 208 team-matches
#
# | Variable | Meaning | Shared with 2.1? |
# |---|---|---|
# | `own_rank_strength` | −log(team's FIFA rank) | yes (ranking) |
# | `host` | team is a host nation | yes |
# | `rest_days` | team's days since previous match | yes |
# | `knockout` | knockout match | yes |
# | `opp_rank_strength` | −log(opponent's FIFA rank) | no |
# | `form_goals_for` | team's goals scored per game in earlier 2026 matches | no |
# | `opp_form_goals_against` | opponent's goals conceded per game in earlier 2026 matches | no |
# | `squad_age` | mean age of the team's squad | no |

# %%
print(d2[X2_COLS + ["goals"]].describe().T.round(2))
print("\nCorrelation with goals:")
print(d2[X2_COLS].corrwith(d2.goals).round(3).sort_values(ascending=False))

# %%
r2 = evaluate(d2, X2_COLS, "goals", "LR 2.2 team goals")
print(r2["ols"].summary())
ct2 = coef_table(r2, d2[X2_COLS]); print(ct2)
print("\nVIF:\n", r2["vif"].round(2))
print("\nNote: the two rows of one match share the same game, so rows are not fully independent;"
      "\ncluster-robust standard errors by match are reported as a check:")
rob = sm.OLS(d2.goals, sm.add_constant(d2[X2_COLS])).fit(cov_type="cluster",
                                                         cov_kwds={"groups": pd.factorize(d2.match_id)[0]})
r2["cluster_p"] = rob.pvalues
print(rob.pvalues.round(3))

# %%
diagnostics_plot(r2, "Model 2.2 — goals scored by a team", "fig_2_2_diagnostics.png")

# %% [markdown]
# ### Comparison and coefficient figure

# %%
summary = pd.concat([summary_row(r1), summary_row(r2)], axis=1)
print(summary)
summary.to_csv("model_summary.csv")
ct1.to_csv("coefficients_2_1.csv"); ct2.to_csv("coefficients_2_2.csv")

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
for a, ct, title in [(ax[0], ct1, "2.1 goal difference"), (ax[1], ct2, "2.2 team goals")]:
    s = ct.drop("const").sort_values("std_coef")
    colors = ["tab:blue" if p < .05 else "lightgray" for p in s.p]
    a.barh(s.index, s.std_coef, color=colors); a.axvline(0, color="k", lw=.8)
    a.set(title=f"Standardised coefficients — {title}", xlabel="std. coefficient (blue: p < 0.05)")
fig.tight_layout(); fig.savefig("fig_coefficients.png", bbox_inches="tight", dpi=200); plt.show()

# %% [markdown]
# ### Robustness: is the extra complexity worth it?
# Compare each 8-variable model with a 1-variable model (ranking only) under the same repeated CV.

# %%
for df, cols, y, one in [(d1, X1_COLS, "goal_diff", ["rank_strength_diff"]),
                         (d2, X2_COLS, "goals", ["own_rank_strength", "opp_rank_strength"])]:
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    for label, c in [("all 8", cols), ("ranking only", one)]:
        s = cross_validate(LinearRegression(), df[c], df[y], cv=rkf, scoring="neg_root_mean_squared_error")
        print(f"{y:10s} {label:13s} CV RMSE = {-s['test_score'].mean():.3f}")
