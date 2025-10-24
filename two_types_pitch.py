import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Basic data cleaning
pitches_22 = pd.read_csv("updated_pitches_22.csv")
pitches_23 = pd.read_csv("updated_pitches_23.csv")

pitches_22["Year"] = 2022
pitches_23["Year"] = 2023
pitches_all = pd.concat([pitches_22, pitches_23], ignore_index=True)
pitches_all = pitches_all.sort_values(by=["gameid", "ab", "pitchnum"]).reset_index(drop=True)
pitch_counts = pitches_all.groupby("pitcher").size()

# Only include pitchers that have pitched over 200 times in two seasons
valid_pitchers = pitch_counts[pitch_counts > 200].index
filtered = pitches_all.set_index(["pitcher","Year"]).loc[valid_pitchers].reset_index()

keys = ["gameid", "ab", "pitcher", "Year"]
pitch_ct   = filtered.groupby(keys)["pitchnum"].transform("count")
type_ct    = filtered.groupby(keys)["pitchname_desc"].transform("nunique")

mask = (pitch_ct > 1) & (type_ct > 1)
filtered_multiAB = filtered[mask].copy()

# Find top two pitch types throughout the whole season
pitch_type_counts = filtered['pitchname_desc'].value_counts()
top_two_types = pitch_type_counts.head(2).index.tolist()
print(top_two_types)

top_two_types = set(top_two_types)

# Filtering to ensure each pitch type occurs in the at bat
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

# Compute distances to create simple tunnel score
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

pitcher_season = pitcher_season[
    (pitcher_season["n_innings"] >= 20)
].copy()

# Computes z-score
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

eps = 1e-9

# Compute the legnth of each at bat for fixed effects in the model
ab_len = (
    fb_sl_pitches.groupby(["gameid","ab","pitcher","Year"])
    .size().rename("ab_len").reset_index()
)

# Remove rows with na values
clean_ab = (
    atbat_df.merge(ab_len, on=["gameid","ab","pitcher","Year"], how="left")
            .assign(tunnel_ratio=lambda d: d["mean_end_atbat_dist"]/(d["mean_init_atbat_dist"]+eps),
                    log_tunnel_ratio=lambda d: np.log(d["tunnel_ratio"] + eps))
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["log_tunnel_ratio","pitcher","Year","ab_len"])
)

# Removes pitchers who havent pitched over 20 ABs  
stable_pitchers = (clean_ab.groupby("pitcher").size()
                   .reset_index(name="n_fbsl_ABs_total")
                   .query("n_fbsl_ABs_total >= 20")["pitcher"])
clean_ab = clean_ab[clean_ab["pitcher"].isin(stable_pitchers)].copy()

# Mixed-effects: fixed = C(Year) + ab_len, random = pitcher
m = smf.mixedlm("log_tunnel_ratio ~ C(Year) + scale(ab_len)",
                data=clean_ab, groups=clean_ab["pitcher"])
r = m.fit(method="lbfgs")
print(r.summary())

# Take the overall tunneling score for each pitcher
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
print(yearly_leaderboard.groupby("Year").head(15))

print(yearly_leaderboard[['pitcher', 'tunneling_score_z']].head(15))

'''
sns.histplot(per_year["tunneling_score_z"], bins=40, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Pitcher random effect (tunneling score)")
plt.title("Histogram of Pitcher Tunneling Scores")
plt.show()
'''