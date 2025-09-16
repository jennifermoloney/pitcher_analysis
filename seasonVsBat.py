import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt


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

season_var = (
    filtered.groupby(["pitcher","Year"])[["initposx","initposz"]]
    .var()
    .rename(columns={"initposx":"var_x","initposz":"var_z"})
    .reset_index()
)
season_var["total_var"] = season_var["var_x"] + season_var["var_z"]

# Calculates the mean of the Euclidean distances of different pitch types throughout the season
# When this is LOW it means the pitcher is "hiding" their pitch types well
def mean_type_dist(group):
    centroids = (group.groupby("pitchname_desc")[["initposx","initposz"]]
                        .mean()
                        .values)
    if len(centroids) < 2:
        return np.nan
    dists = [np.linalg.norm(a-b) for a,b in combinations(centroids,2)]
    return np.mean(dists)

season_type_dist = (
    filtered.groupby(["pitcher","Year"])
            .apply(mean_type_dist)
            .rename("mean_inter_type_dist")
            .reset_index()
)

pitcher_season = pd.merge(season_var, season_type_dist,
                          on=["pitcher","Year"])

print(pitcher_season.head())

# Calculates avg distance of each atbat
def atbat_mean_dist(group):
    pts = group[["initposx","initposz"]].values
    center = pts.mean(axis=0)
    return np.mean(np.linalg.norm(pts - center, axis=1))

atbat_df = (
    filtered.groupby(["gameid","ab","pitcher","Year"])
            .apply(atbat_mean_dist)
            .rename("mean_atbat_dist")
            .reset_index()
)

print(atbat_df.head())
atbat_outcomes = (
    filtered.groupby(["gameid","ab","pitcher","Year"])["eventtype"]
            .first()
            .reset_index()
)
atbat_full = atbat_df.merge(atbat_outcomes,
                            on=["gameid","ab","pitcher","Year"])

top_events = (atbat_full['eventtype']
              .value_counts()
              .head(10)
              .index)




#sns.boxplot(x="eventtype", y="mean_atbat_dist",
#            data=atbat_full[atbat_full['eventtype'].isin(top_events)])
#plt.show()