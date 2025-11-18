import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Tests significance of fixed effects with likelihood test 
# Tests for correlation between variables that were selected in the model
# Should be used after performing best subset selection to ensure important features


pitches_22 = pd.read_csv("updated_pitches_22.csv")
pitches_23 = pd.read_csv("updated_pitches_23.csv")
pitches_24 = pd.read_csv("updated_pitches_24.csv")

pitches_22["Year"] = 2022
pitches_23["Year"] = 2023
pitches_24["Year"] = 2024
pitches_all = pd.concat([pitches_22, pitches_23, pitches_24], ignore_index=True)

# filtered out unreasonable pitch estimates
pitches_all = pitches_all[pitches_all["outs"].between(0, 2)]
pitches_all = pitches_all.dropna(subset=["initposx", "initposz", "platelocside", "platelocheight"])
pitches_all = pitches_all[
    (pitches_all["relspeed"].between(50, 110)) & 
    (pitches_all["spinrate"].between(0, 4000))
]
pitches_all = pitches_all.sort_values(["gameid", "ab", "pitchnum"]).reset_index(drop=True)

pitch_counts = pitches_all.groupby("pitcher").size()
valid_pitchers = pitch_counts[pitch_counts > 200].index
pitches_all = pitches_all[pitches_all["pitcher"].isin(valid_pitchers)].copy()

keys = ["gameid", "ab", "pitcher", "Year"]

pitch_ct = pitches_all.groupby(keys)["pitchnum"].transform("count")
type_ct  = pitches_all.groupby(keys)["pitchname_desc"].transform("nunique")
mask = (pitch_ct > 1) & (type_ct > 1)
filtered_multiAB = pitches_all[mask].copy()

# Create simple tunnel score
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

# Important features
ab_pitch_metrics = (
    filtered_multiAB
    .groupby(["gameid", "ab", "pitcher", "Year"])
    .agg(
        mean_relspeed=("relspeed", "mean"),
        mean_spinrate=("spinrate", "mean"),
        visscore_last=("visscore", "last"),
        homscore_last=("homscore", "last"),
        metrics_pitching_position=("metrics_pitching_position", "last")
    )
    .reset_index()
)

ab_pitch_metrics["score_diff"] = (
    ab_pitch_metrics["homscore_last"] - ab_pitch_metrics["visscore_last"]
)

eps = 1e-9

# AB length
ab_len = (
    filtered_multiAB.groupby(["gameid","ab","pitcher","Year"])
    .size().rename("ab_len").reset_index()
)

# ensures no NA values, merges all dfs together into a clean df
clean_ab = (
    atbat_df
    .merge(ab_len, on=["gameid","ab","pitcher","Year"], how="left")
    .merge(ab_pitch_metrics, on=["gameid","ab","pitcher","Year"], how="left")
    .assign(
        tunnel_ratio=lambda d: d["mean_end_atbat_dist"]/(d["mean_init_atbat_dist"]+eps),
        log_tunnel_ratio=lambda d: np.log(d["tunnel_ratio"] + eps)
    )
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["log_tunnel_ratio","pitcher","Year","ab_len",
                    "mean_relspeed","mean_spinrate"])
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


# variables used in formula and as groups
needed = [
    "log_tunnel_ratio", "pitcher",
    "metrics_pitching_position", "matchup",
    "ab_len", "mean_relspeed", "mean_spinrate"
]

clean_ab = (
    clean_ab
      .replace([np.inf, -np.inf], np.nan)
      .dropna(subset=needed)
      .reset_index(drop=True)
)

# Perform likelihood tests by removing one feature at a time
terms = ['C(metrics_pitching_position)', "C(matchup)", "scale(ab_len)", "scale(mean_relspeed)", "scale(mean_spinrate)"]
full_formula = "log_tunnel_ratio ~ " + " + ".join(terms)

m_full = smf.mixedlm(full_formula, data=clean_ab, groups=clean_ab["pitcher"])
r_full = m_full.fit(method="lbfgs", reml= False)

# drop one feature at a time
def fit_reduced(drop_term):
    reduced_terms = [t for t in terms if t != drop_term]
    reduced_formula = "log_tunnel_ratio ~ " + " + ".join(reduced_terms) if reduced_terms else "log_tunnel_ratio ~ 1"
    m_reduced = smf.mixedlm(reduced_formula, data=clean_ab, groups=clean_ab["pitcher"])
    return m_reduced.fit(method="lbfgs", reml = False)

lr_results = []
for term in terms:
    r_reduced = fit_reduced(term)
    LR = 2 * (r_full.llf - r_reduced.llf)
    df = r_full.df_modelwc - r_reduced.df_modelwc
    p_val = chi2.sf(LR, df)
    lr_results.append({"term": term, "LR stat": LR, "df": df, "p": p_val})

lr_table = pd.DataFrame(lr_results).sort_values("p")
print("\n=== Likelihood Ratio Tests ===")
print(lr_table)

# Test correlation of numerical features
X = clean_ab[["ab_len", "mean_relspeed", "mean_spinrate", "score_diff"]]
corr = X.corr()
print(corr)