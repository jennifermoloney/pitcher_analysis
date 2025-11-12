import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
# Final model with all significant fixed effects

# Calculates pitch scores wit 4 fixed effects
pitches_22 = pd.read_csv("updated_pitches_22.csv")
pitches_23 = pd.read_csv("updated_pitches_23.csv")
pitches_24 = pd.read_csv("updated_pitches_24.csv")

pitches_22["Year"] = 2022
pitches_23["Year"] = 2023
pitches_24["Year"] = 2024
pitches_all = pd.concat([pitches_22, pitches_23, pitches_24], ignore_index=True)
print(len(pitches_all))
pitches_all = pitches_all.sort_values(by=["gameid", "ab", "pitchnum"]).reset_index(drop=True)
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

pitcher_season["tunnel_ratio_z"] = (
    pitcher_season.groupby("Year")["tunnel_ratio_median"]
    .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
)

innings_per_season = (
    filtered_multiAB[["pitcher","Year","gameid","inning"]]
      .drop_duplicates()
      .groupby(["pitcher","Year"])
      .size()
      .reset_index(name="n_innings")
)

ab_pitch_metrics = (
    filtered_multiAB
    .groupby(["gameid", "ab", "pitcher", "Year"])
    .agg(
        mean_relspeed=("relspeed", "mean"),
        mean_spinrate=("spinrate", "mean"),
        metrics_pitching_position=("metrics_pitching_position", "last")
    )
    .reset_index()
)

eps = 1e-9

# AB length
ab_len = (
    filtered_multiAB.groupby(["gameid","ab","pitcher","Year"])
    .size().rename("ab_len").reset_index()
)

clean_ab = (
    atbat_df
    .merge(ab_len, on=["gameid","ab","pitcher","Year"], how="left")
    .merge(ab_pitch_metrics, on=["gameid","ab","pitcher","Year"], how="left")
    .assign(
        tunnel_ratio=lambda d: d["mean_end_atbat_dist"]/(d["mean_init_atbat_dist"]+eps),
        log_tunnel_ratio=lambda d: np.log(d["tunnel_ratio"] + eps)
    )
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["log_tunnel_ratio","pitcher","ab_len",
                    "mean_relspeed","mean_spinrate", 'metrics_pitching_position'])
)

handedness_df = (
    filtered_multiAB
    .groupby(["gameid", "ab", "pitcher", "Year"])
    .agg(
        pitcher_hand=("pitcherthrows", "first"),
        batter_hand=("batterside", "first")
    )
    .reset_index()
)

# Merge onto clean_ab
clean_ab = clean_ab.merge(handedness_df, on=["gameid", "ab", "pitcher", "Year"], how="left")

clean_ab["matchup"] = clean_ab["pitcher_hand"] + " vs " + clean_ab["batter_hand"]
print(len(clean_ab["pitcher"].unique()))
stable_pitchers = (clean_ab.groupby("pitcher").size()
                   .reset_index(name="ABs_total")
                   .query("ABs_total >= 20")["pitcher"])
clean_ab = clean_ab[clean_ab["pitcher"].isin(stable_pitchers)].copy()


terms = ['C(metrics_pitching_position)', "C(matchup)", "scale(ab_len)", "scale(mean_relspeed)", "scale(mean_spinrate)"]
full_formula = "log_tunnel_ratio ~ " + " + ".join(terms)

m_full = smf.mixedlm(full_formula, data=clean_ab, groups=clean_ab["pitcher"])
r_full = m_full.fit(method="lbfgs", reml= False)

print(r_full.summary())

# Overall (pooled) tunneling scores = random intercepts
re = r_full.random_effects
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


### COMPARE IT TO SIMPLE MODEL
comparison_df = (
    pitcher_season
    .merge(per_year, on=["pitcher", 'Year'], how="inner")
    .rename(columns={
        "tunnel_ratio_median": "simple_tunnel_score",
        'tunnel_ratio_z': "simple_tunnel_score_z",
        "tunneling_score_z": "mixed_tunnel_score_z",
        "tunneling_score": "mixed_tunnel_score"
    })
)

# Check result
print(comparison_df.head())

rho, p = spearmanr(comparison_df["simple_tunnel_score_z"], comparison_df["mixed_tunnel_score_z"])
print(f"Spearman ρ = {rho:.3f}, p = {p:.6g}")

sns.regplot(
    x="simple_tunnel_score_z",
    y="mixed_tunnel_score_z",
    data=comparison_df,
    scatter_kws={"alpha":0.6},
    line_kws={"color":"red"}
)
plt.xlabel("Simple Tunnel Ratio (Z-score)")
plt.ylabel("Mixed Effects Tunneling Score")
plt.title(f"Comparison of Simple vs Mixed Tunneling Models\n(Spearman ρ={rho:.2f})")
plt.show()

pitcher_summary = (
    comparison_df.groupby("pitcher")[["simple_tunnel_score_z", "mixed_tunnel_score_z"]]
    .mean()
    .reset_index()
)
print('done')

top_simple = (
    pitcher_summary.sort_values("simple_tunnel_score_z", ascending=False)
    .head(20)[["pitcher", "simple_tunnel_score_z"]]
    .reset_index(drop=True)
)

top_mixed = (
    pitcher_summary.sort_values("mixed_tunnel_score_z", ascending=False)
    .head(20)[["pitcher", "mixed_tunnel_score_z"]]
    .reset_index(drop=True)
)

print("\nTop 10 Pitchers (Simple Model)")
print(top_simple)
print(pitcher_summary['simple_tunnel_score_z'].mean())

print("\nTop 10 Pitchers (Mixed-Effects Model)")
print(top_mixed)
print(pitcher_summary['mixed_tunnel_score_z'].mean())

out_events = ['field_out']
num_outs = (
    filtered_multiAB[filtered_multiAB["eventtype"].isin(out_events)]
    .groupby("pitcher")
    .size()
    .reset_index(name="total_outs")
)

comparison_df = comparison_df.merge(num_outs, on="pitcher", how="left")
comparison_df["total_outs"] = comparison_df["total_outs"].fillna(0)
rho_simple, p_simple = spearmanr(
    comparison_df["simple_tunnel_score_z"], comparison_df["total_outs"]
)
rho_mixed, p_mixed = spearmanr(
    comparison_df["mixed_tunnel_score_z"], comparison_df["total_outs"]
)

print(f"Correlation with outs (simple): ρ={rho_simple:.3f}, p={p_simple:.3g}")
print(f"Correlation with outs (mixed): ρ={rho_mixed:.3f}, p={p_mixed:.3g}")

sns.barplot(x=["Simple", "Mixed"],
            y=[rho_simple, rho_mixed],
            palette="Blues_r")
plt.ylabel("Spearman correlation with Outs")
plt.title("Model Predictive Strength vs. Pitching Outcomes")
plt.show()
