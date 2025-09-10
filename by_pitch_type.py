import numpy as np
import pandas as pd

pitches_22 = pd.read_csv("updated_pitches_22.csv")
pitch_counts = pitches_22.groupby("pitcher").size()
valid_pitchers = pitch_counts[pitch_counts > 100].index

filtered_pitches = pitches_22[pitches_22["pitcher"].isin(valid_pitchers)]

variance_df = (
    filtered_pitches.groupby("pitcher")[["initposx", "initposz"]]
    .var()
    .reset_index()
)

variance_df["total_var"] = variance_df["initposx"] + variance_df["initposz"]
variance_df_sorted_total = variance_df.sort_values("total_var")

print(variance_df_sorted_total)


jose_pitches = pitches_22[pitches_22['pitcher'] == 'Rainey, Tanner']
pitch_counts = jose_pitches['pitchname_desc'].value_counts().sort_values(ascending=False)

print("Types of pitches:")
print(pitch_counts)
print("Number of pitches this season")
print(jose_pitches['pitchname_desc'].count())



counts_per_pitcher = filtered_pitches.groupby('pitcher')['pitchname_desc'].count()

summary_stats = pd.DataFrame({
    "mean": [counts_per_pitcher.mean()],
    "median": [counts_per_pitcher.median()],
    "std": [counts_per_pitcher.std()],
    "min": [counts_per_pitcher.min()],
    "max": [counts_per_pitcher.max()],
    "IQR": [counts_per_pitcher.quantile(0.75) - counts_per_pitcher.quantile(0.25)],
    "Q1": [counts_per_pitcher.quantile(0.25)],
    "Q3": [counts_per_pitcher.quantile(0.75)]
})

print(counts_per_pitcher.head())   # counts per pitcher
#print(summary_stats) 
print("baseball")
