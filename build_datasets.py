# %% [markdown]
# # FIFA World Cup 2026 — Linear Regression (Objective 2)
# ## Part A: data wrangling — building the 104-row and 208-row datasets
#
# **Raw inputs**
# * `Players.xlsx` — one row per player per match for all 104 matches (FBref match reports, collected in Assessment 2).
# * `team_info.csv` — static pre-tournament facts per team: FIFA/Coca-Cola Men's World Ranking of 11 June 2026
#   (published before the opening match), number of World Cup appearances before 2026, and host-nation flag.
#
# **Rule applied to every explanatory variable:** it must be known before kick-off of the match it describes.
# In-tournament "form" variables therefore use **only matches the team played earlier** in the tournament.

# %%
import numpy as np
import pandas as pd

p = pd.read_excel("Players.xlsx")
for c in ["game", "team", "game_id"]:
    p[c] = p[c].ffill()                     # labels only appear on the first row of each block
p["match_date"] = pd.to_datetime(p["game"].str[:10])
p["age_years"] = p["age"].str.split("-").str[0].astype(float) + p["age"].str.split("-").str[1].astype(float) / 365.25
print(p.shape, "matches:", p.game_id.nunique(), "teams:", p.team.nunique())

info = pd.read_csv("team_info.csv")

# %% [markdown]
# ### Step 1 — team-match table (208 rows)
# Goals for a team = goals by its own players (penalties included) + own goals conceded by the opponent.
# Penalty-shoot-out kicks are not goals and are excluded, so a shoot-out match counts as a draw.

# %%
tm = p.groupby(["game_id", "game", "match_date", "team"], as_index=False).agg(
    gls=("Performance_Gls", "sum"), og=("Performance_OG", "sum"),
    shots=("Performance_Sh", "sum"), sot=("Performance_SoT", "sum"),
    yellow=("Performance_CrdY", "sum"), red=("Performance_CrdR", "sum"))

opp = tm[["game_id", "team", "og", "gls", "sot"]].rename(columns={"team": "opponent", "og": "opp_og", "gls": "opp_gls", "sot": "opp_sot"})
tm = tm.merge(opp, on="game_id")
tm = tm[tm.team != tm.opponent].copy()
tm["goals_for"] = tm.gls + tm.opp_og
tm["goals_against"] = tm.opp_gls + tm.og
tm["points"] = np.select([tm.goals_for > tm.goals_against, tm.goals_for == tm.goals_against], [3, 1], 0)
assert len(tm) == 208

# Team A = first team named by FIFA in the fixture ("YYYY-MM-DD TeamA-TeamB")
fixture = (tm.game.str[11:].str.replace("USA-", "United States-", regex=False)
           .str.replace("Bosnia–Herz", "Bosnia and Herzegovina", regex=False))   # FBref short names
tm["is_team_a"] = [f.startswith(t + "-") for f, t in zip(fixture, tm.team)]
assert tm.groupby("game_id").is_team_a.sum().eq(1).all()

# %% [markdown]
# ### Step 2 — pre-match variables
# * **Rest days**: days since the team's previous match (capped at 7; the first match is set to 7 = fully rested).
# * **Knockout**: the team's 4th or later match. Every team plays exactly 3 group matches.
# * **Form variables** (goals for/against, goal difference, points, shots on target per game) use only
#   earlier matches. To handle matchday 1 (no history) and to stop one result dominating, each average is
#   shrunk towards a pre-tournament prior with weight of one match:
#   `form = (sum of earlier values + prior) / (earlier matches + 1)`.
#   The goals prior comes from the 2022 World Cup (172 goals in 64 matches = 1.34 goals per team per match),
#   so no 2026 information leaks in. In a given match both teams have always played the same number of
#   earlier matches, so in the Team A − Team B differences of model 2.1 the prior cancels out exactly
#   (matchday-1 differences are 0 = "no evidence either way"); the points/SoT priors are therefore placeholders.
# * **Squad age**: mean age of the players the team used (all registered before the tournament).

# %%
tm = tm.sort_values(["team", "match_date"]).reset_index(drop=True)
g = tm.groupby("team")
tm["match_no"] = g.cumcount() + 1
tm["knockout"] = (tm.match_no > 3).astype(int)
tm["rest_days"] = g.match_date.diff().dt.days.fillna(7).clip(upper=7)

PRIOR = {"goals_for": 1.34, "goals_against": 1.34, "points": 1.36, "sot": 4.0}
tm["gd"] = tm.goals_for - tm.goals_against
PRIOR["gd"] = 0.0
for col, prior in PRIOR.items():
    earlier_sum = g[col].cumsum() - tm[col]            # sum over earlier matches only
    tm[f"form_{col}"] = (earlier_sum + prior) / (tm.match_no - 1 + 1)

squad_age = p.groupby(["team", "player"]).age_years.min().groupby("team").mean().rename("squad_age")
tm = tm.merge(squad_age, on="team").merge(info, on="team")
tm["log_rank"] = np.log(tm.fifa_rank_jun2026)

# %% [markdown]
# ### Step 3 — attach the opponent's pre-match values

# %%
opp_cols = ["log_rank", "host", "rest_days", "form_goals_for", "form_goals_against", "form_gd",
            "form_points", "form_sot", "squad_age", "wc_appearances_before_2026"]
o = tm[["game_id", "team"] + opp_cols].rename(columns={"team": "opponent", **{c: "opp_" + c for c in opp_cols}})
tm = tm.merge(o, on=["game_id", "opponent"])
assert len(tm) == 208

# %% [markdown]
# ### Step 4 — Dataset 2.1: one row per match (104 rows), Team A minus Team B
# Rank enters as log(rank of B) − log(rank of A), so a positive value means Team A is the stronger side
# and the log reflects that the gap between 1st and 10th matters more than between 50th and 60th.

# %%
a = tm[tm.is_team_a].copy()
match = pd.DataFrame({
    "match_id": a.game_id, "match": a.game, "date": a.match_date.dt.date,
    "team_a": a.team, "team_b": a.opponent,
    "rank_strength_diff": a.opp_log_rank - a.log_rank,
    "host_diff": a.host - a.opp_host,
    "rest_diff": a.rest_days - a.opp_rest_days,
    "knockout": a.knockout,
    "form_gd_diff": a.form_gd - a.opp_form_gd,
    "form_points_diff": a.form_points - a.opp_form_points,
    "form_sot_diff": a.form_sot - a.opp_form_sot,
    "wc_experience_diff": a.wc_appearances_before_2026 - a.opp_wc_appearances_before_2026,
    "goal_diff": a.goals_for - a.goals_against,
}).sort_values("date").reset_index(drop=True)
assert len(match) == 104
match.to_csv("dataset_2_1_match_goal_difference.csv", index=False)

# %% [markdown]
# ### Step 5 — Dataset 2.2: one row per team per match (208 rows)

# %%
team = pd.DataFrame({
    "match_id": tm.game_id, "match": tm.game, "date": tm.match_date.dt.date,
    "team": tm.team, "opponent": tm.opponent,
    "own_rank_strength": -tm.log_rank,          # higher = stronger
    "host": tm.host,
    "rest_days": tm.rest_days,
    "knockout": tm.knockout,
    "opp_rank_strength": -tm.opp_log_rank,
    "form_goals_for": tm.form_goals_for,
    "opp_form_goals_against": tm.opp_form_goals_against,
    "squad_age": tm.squad_age,
    "goals": tm.goals_for,
}).sort_values(["date", "match_id", "team"]).reset_index(drop=True)
assert len(team) == 208
team.to_csv("dataset_2_2_team_goals.csv", index=False)

print(match.describe().T.round(2))
print(team.describe().T.round(2))
