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

# Only include pitchers that have pitched over 200 times in two seasons
valid_pitchers = pitch_counts[pitch_counts > 200].index
filtered = pitches_all.set_index(["pitcher","Year"]).loc[valid_pitchers].reset_index()

keys = ["gameid", "ab", "pitcher", "Year"]

pitch_ct   = filtered.groupby(keys)["pitchnum"].transform("count")
type_ct    = filtered.groupby(keys)["pitchname_desc"].transform("nunique")

mask = (pitch_ct > 1) & (type_ct > 1)
filtered_multiAB = filtered[mask].copy()

pitch_type_counts = filtered['pitchname_desc'].value_counts()
top_two_types = pitch_type_counts.head(2).index.tolist()
print(top_two_types)
top_two_types = set(top_two_types)

fb_slider_unique = (
    filtered_multiAB.groupby(keys)["pitchname_desc"]
    .transform(lambda s: set(s.unique()).issubset(top_two_types))        
)

fb_slider = (
    filtered_multiAB.groupby(keys)["pitchname_desc"]
    .transform(lambda s: top_two_types.issubset(set(s.unique())))          
)

mask_fb_sl_exact = fb_slider_unique & fb_slider

fb_sl_pitches = filtered_multiAB[
    mask_fb_sl_exact & filtered_multiAB["pitchname_desc"].isin(top_two_types)
].copy()


def atbat_mean_dist(group):
    pts = group[["initposx","initposz"]].values
    center = pts.mean(axis=0)
    return np.mean(np.linalg.norm(pts - center, axis=1))

def atbat_mean_dist_end(group):
    pts = group[["platelocside","platelocheight"]].values
    center = pts.mean(axis=0)
    return np.mean(np.linalg.norm(pts - center, axis=1))

start_df = (
    fb_sl_pitches.groupby(keys)
            .apply(atbat_mean_dist)
            .rename("mean_init_atbat_dist")
            .reset_index()
)

end_df = (
    fb_sl_pitches.groupby(keys)
            .apply(atbat_mean_dist_end)
            .rename("mean_end_atbat_dist")
            .reset_index()
)

atbat_df = start_df.merge(end_df, on=keys, how = 'inner')
print(atbat_df.head(10))

atbat_df['tunnel_ratio'] = (
    atbat_df['mean_end_atbat_dist'] / (atbat_df['mean_init_atbat_dist'])
)

pitcher_season = (
    atbat_df.groupby(["pitcher","Year"])
      .agg(
          fbsl_tunnel_ratio_median=("tunnel_ratio","median")
      )
      .reset_index()
)

innings_per_season = (
    fb_sl_pitches[["pitcher","Year","gameid","inning"]]
      .drop_duplicates()
      .groupby(["pitcher","Year"])
      .size()
      .reset_index(name="n_innings")
)

pitcher_season = pitcher_season.merge(innings_per_season, on=["pitcher","Year"], how="left")

# Stability filters: innings + minimum FB/SL ABs
pitcher_season = pitcher_season[
    (pitcher_season["n_innings"] >= 20)
].copy()

pitcher_season["fbsl_tunnel_ratio_median_z"] = (
    pitcher_season.groupby("Year")["fbsl_tunnel_ratio_median"]
    .transform(lambda s: (s - s.mean()) / (s.std(ddof=0)))
)

pitcher_season["proto_tunnel_score"] = pitcher_season["fbsl_tunnel_ratio_median_z"]

leaderboard = (
    pitcher_season.sort_values(["Year","proto_tunnel_score"], ascending=[True, False])
)
print(leaderboard.head(15))

results = []

clean_ab = atbat_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["tunnel_ratio","pitcher","Year"])

# Keep only pitcher-seasons that pass your innings threshold
ok_pairs = pitcher_season[["pitcher","Year"]].drop_duplicates()
clean_ab = clean_ab.merge(ok_pairs, on=["pitcher","Year"], how="inner")

for y, sub in clean_ab.groupby("Year"):
    sub = sub.copy()
    sub["log_tunnel_ratio"] = np.log(sub["tunnel_ratio"])

    # Random-intercept model
    m = smf.mixedlm("log_tunnel_ratio ~ 1", data=sub, groups=sub["pitcher"])
    r = m.fit(method="lbfgs")

    # BLUPs = pitcher tunneling scores for that season
    re = pd.DataFrame({
        "pitcher": list(r.random_effects.keys()),
        "tunneling_score": [v[0] for v in r.random_effects.values()],
        "Year": y
    })
    re["tunneling_score_z"] = (
        (re["tunneling_score"] - re["tunneling_score"].mean()) /
        (re["tunneling_score"].std(ddof=0))
    )
    results.append(re)

me_leaderboard = (
    pd.concat(results, ignore_index=True)
      .sort_values(["Year","tunneling_score_z"], ascending=[True, False])
)
print(me_leaderboard.head(15))