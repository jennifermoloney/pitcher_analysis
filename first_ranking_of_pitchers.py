import pandas as pd
import numpy as np
### Simple tunnel score ranking

# Basic data cleaning
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

# Compute simple tunneling ratio
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

atbat_df = start_df.merge(end_df, on=keys)

# Create actual tunnel ratio
atbat_df["tunnel_ratio"] = atbat_df["mean_end_atbat_dist"] / atbat_df["mean_init_atbat_dist"] 
ab_outcomes = filtered_multiAB.groupby(keys)["eventtype"].first().reset_index()
atbat_df = atbat_df.merge(ab_outcomes, on=keys, how="left")

# Creates df of number of ABs for each pitcher and the median of the tunnel score
pitcher_season = (
    atbat_df.groupby(["pitcher","Year"])
      .agg(
          n_ABs=("tunnel_ratio","size"),
          tunnel_ratio_median=("tunnel_ratio","median")
      )
      .reset_index()
)

# Check how many innings each pitcher plays per season
innings_per_season = (
    filtered_multiAB[["pitcher","Year","gameid","inning"]]
      .drop_duplicates()
      .groupby(["pitcher","Year"])
      .size()
      .reset_index(name="n_innings")
)

# Join dfs together
pitcher_season = pitcher_season.merge(innings_per_season,
                                      on=["pitcher","Year"],
                                      how="left")

# filter for minimum num of innings per season
pitcher_season = pitcher_season[pitcher_season["n_innings"] >= 30]

# Compute the z score of the tunnel ratio to normalize it
pitcher_season["tunnel_ratio_median_z"] = (
    pitcher_season.groupby("Year")["tunnel_ratio_median"]
    .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
)

pitcher_season["proto_tunnel_score"] = pitcher_season["tunnel_ratio_median_z"]

leaderboard = pitcher_season.sort_values(["Year","proto_tunnel_score"], ascending=[True, False])
print(leaderboard[['pitcher', 'proto_tunnel_score']].head(15))

'''
sns.histplot(pitcher_season["proto_tunnel_score"], bins=40, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Pitcher simple tuneling stat (z-score)")
plt.title("Histogram of Pitcher Tunneling Scores")
plt.show()
'''