import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


pitches_22 = pd.read_csv("updated_pitches_22.csv")
pitches_23 = pd.read_csv("updated_pitches_23.csv")

pitches_22["Year"] = 2022
pitches_23["Year"] = 2023
pitches_all = pd.concat([pitches_22, pitches_23], ignore_index=True)
pitch_counts = pitches_all.groupby("pitcher").size()

valid_pitchers = pitch_counts[pitch_counts > 200].index
filtered = pitches_all.set_index(["pitcher","Year"]).loc[valid_pitchers].reset_index()
keys = ["gameid", "ab", "pitcher", "Year"]

# Count the number of pitches and pitch types per at bat
pitch_ct   = filtered.groupby(keys)["pitchnum"].transform("count")
type_ct    = filtered.groupby(keys)["pitchname_desc"].transform("nunique")

# filter out the plays with only one pitch/ pitch type
mask = (pitch_ct > 1) & (type_ct > 1)
filtered_multiAB = filtered[mask].copy()

def atbat_mean_dist(group):
    pts = group[["initposx","initposz"]].values
    center = pts.mean(axis=0)
    return np.mean(np.linalg.norm(pts - center, axis=1))

def atbat_mean_dist_end(group):
    pts = group[["platelocside","platelocheight"]].values
    center = pts.mean(axis=0)
    return np.mean(np.linalg.norm(pts - center, axis=1))

start_df = (
    filtered_multiAB.groupby(keys)
            .apply(atbat_mean_dist)
            .rename("mean_init_atbat_dist")
            .reset_index()
)

end_df = (
    filtered_multiAB.groupby(keys)
            .apply(atbat_mean_dist_end)
            .rename("mean_end_atbat_dist")
            .reset_index()
)

atbat_df = start_df.merge(end_df, on=keys, how = 'inner')
print(atbat_df.head(10))

# Create the simple tunnel ratio per each at bat
atbat_df['tunnel_ratio'] = (
    atbat_df['mean_end_atbat_dist'] / (atbat_df['mean_init_atbat_dist'])
)

pitcher_season = (
    atbat_df.groupby(["pitcher","Year"])
      .agg(
          tunnel_ratio_median=("tunnel_ratio","median")
      )
      .reset_index()
)

innings_per_season = (
    filtered_multiAB[["pitcher","Year","gameid","inning"]]
      .drop_duplicates()
      .groupby(["pitcher","Year"])
      .size()
      .reset_index(name="n_innings")
)

eps = 1e-9

# AB length
ab_len = (
    filtered_multiAB.groupby(["gameid","ab","pitcher","Year"])
    .size().rename("ab_len").reset_index()
)

clean_ab = (
    atbat_df.merge(ab_len, on=["gameid","ab","pitcher","Year"], how="left")
            .assign(tunnel_ratio=lambda d: d["mean_end_atbat_dist"]/(d["mean_init_atbat_dist"]+eps),
                    log_tunnel_ratio=lambda d: np.log(d["tunnel_ratio"] + eps))
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["log_tunnel_ratio","pitcher","Year","ab_len"])
)

stable_pitchers = (clean_ab.groupby("pitcher").size()
                   .reset_index(name="ABs_total")
                   .query("ABs_total >= 20")["pitcher"])
clean_ab = clean_ab[clean_ab["pitcher"].isin(stable_pitchers)].copy()

# Mixed-effects: fixed = C(Year) + ab_len, random = pitcher
m = smf.mixedlm("log_tunnel_ratio ~ C(Year) + scale(ab_len)",
                data=clean_ab, groups=clean_ab["pitcher"])
r = m.fit(method="lbfgs")
print(r.summary())

# Overall (pooled) tunneling scores = random intercepts
re = r.random_effects
overall_scores = pd.DataFrame({
    "pitcher": list(re.keys()),
    "tunneling_score": [v[0] for v in re.values()]
})

# Per-year display (z within each season using the same pooled score)
pitcher_years = clean_ab[["pitcher","Year"]].drop_duplicates()
per_year = pitcher_years.merge(overall_scores, on="pitcher", how="left")
per_year["tunneling_score_z"] = (
    per_year.groupby("Year")["tunneling_score"]
            .transform(lambda s: (s - s.mean())/(s.std(ddof=0) + eps))
)
yearly_leaderboard = per_year.sort_values(["Year","tunneling_score_z"], ascending=[True, False])
print(yearly_leaderboard.head(15))

