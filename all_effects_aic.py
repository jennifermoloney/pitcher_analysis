import pandas as pd
import numpy as np
import itertools
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# Best subset selection -> finds the most significant combination of features

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

atbat_df = start_df.merge(end_df, on=keys, how="inner")

ab_len = (
    filtered_multiAB.groupby(keys)
    .size().rename("ab_len")
    .reset_index()
)

ab_mech = (
    filtered_multiAB.groupby(keys)
    .agg(
        mean_relspeed=("relspeed", "mean"),
        mean_spinrate=("spinrate", "mean"),
        horzbreak=("horzbreak", "mean"),
        inducedvertbreak=("inducedvertbreak", "mean")
    )
    .reset_index()
)

ab_ctx = (
    filtered_multiAB.groupby(keys)
    .agg(
        inning=("inning", "last"),
        outs=("outs", "last"),
        visscore_last=("visscore", "last"),
        homscore_last=("homscore", "last"),
        metrics_pitching_position=("metrics_pitching_position", "last"),
        pitcherthrows=("pitcherthrows", "last"),
        batterside=("batterside", "last")
    )
    .reset_index()
)
ab_ctx["score_diff"] = ab_ctx["homscore_last"] - ab_ctx["visscore_last"]

ab_ctx["pitcherthrows"] = ab_ctx["pitcherthrows"].astype(str).str.upper().str[0]
ab_ctx["batterside"]    = ab_ctx["batterside"].astype(str).str.upper().str[0]

ab_ctx = ab_ctx[ab_ctx["pitcherthrows"].isin(["L","R"]) & ab_ctx["batterside"].isin(["L","R"])].copy()
ab_ctx["pl_matchup4"] = ab_ctx["pitcherthrows"] + ab_ctx["batterside"]

atbat_df = (
    atbat_df
    .merge(ab_len, on=keys, how="left")
    .merge(ab_mech, on=keys, how="left")
    .merge(ab_ctx, on=keys, how="left")
)

eps = 1e-9
atbat_df["tunnel_ratio"] = atbat_df["mean_end_atbat_dist"] / (atbat_df["mean_init_atbat_dist"] + eps)
atbat_df["log_tunnel_ratio"] = np.log(atbat_df["tunnel_ratio"] + eps)

needed_cols = [
    "log_tunnel_ratio", "pitcher", "Year", "ab_len",
    "mean_relspeed", "mean_spinrate", "horzbreak", "inducedvertbreak",
    "inning", "outs", "score_diff", "metrics_pitching_position",
    "pl_matchup4"   
]
clean_ab = atbat_df.dropna(subset=needed_cols).copy()

stability = clean_ab.groupby("pitcher").size().rename("n_ABs").reset_index()
keep_pitchers = stability[stability["n_ABs"] >= 20]["pitcher"]
clean_ab = clean_ab[clean_ab["pitcher"].isin(keep_pitchers)].copy()

print(f"Modeling rows: {len(clean_ab):,}, pitchers: {clean_ab['pitcher'].nunique()}")

numeric_effects = [
    "scale(ab_len)",
    "scale(mean_relspeed)",
    "scale(mean_spinrate)",
    "scale(inning)",
    "scale(outs)",
    "scale(horzbreak)",
    "scale(inducedvertbreak)",
    "scale(score_diff)"
]

categorical_effects = [
    "C(Year)",
    "C(metrics_pitching_position)",
    "C(pl_matchup4)"
]

all_effects = categorical_effects + numeric_effects
print("Candidate fixed effects:")
for eff in all_effects:
    print(" •", eff)
print(f"\nTotal candidate predictors: {len(all_effects)}")

results = []
max_predictors = 5

for k in range(1, max_predictors + 1):
    for combo in itertools.combinations(all_effects, k):
        formula = "log_tunnel_ratio ~ " + " + ".join(combo)
        try:
            mdl = smf.mixedlm(formula, data=clean_ab, groups=clean_ab["pitcher"])
            fit = mdl.fit(method="lbfgs", reml=False)
            results.append({
                "formula": formula,
                "AIC": fit.aic,
                "BIC": fit.bic,
                "LogLik": fit.llf,
                "Converged": getattr(fit, "converged", True)
            })
            print(f"Fitted: {formula}")
        except Exception as e:
            print(f"Failed: {formula}\n   Reason: {e}")


results_df = (pd.DataFrame(results)
                .query("Converged")
                .sort_values(["AIC","BIC"])
                .reset_index(drop=True))
print("\nTop 10 models by AIC:")
print(results_df.head(10))

best_formula = results_df.iloc[0]["formula"]
print(f"\n Best model by AIC:\n{best_formula}")
best_by_bic = results_df.sort_values("BIC").iloc[0]
print(f"\n Best model by BIC:\n{best_by_bic['formula']}")

best_model = smf.mixedlm(best_formula, data=clean_ab, groups=clean_ab["pitcher"])
best_fit = best_model.fit(method="lbfgs", reml=True)
print(best_fit.summary())
