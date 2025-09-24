# tunnel.py
import argparse, sys, numpy as np, pandas as pd

REQ_COLS = [
    "gameid","pitcher","ab","pitchnum","batter",
    "initposx","initposy","initposz",
    "platelocside","platelocheight","pitchname"
]

def main():
    ap = argparse.ArgumentParser(description="Compute pitch tunneling metrics.")
    ap.add_argument("input_csv", help="Path to pitch-level CSV")
    ap.add_argument("--pairs_out", default="tunnel_pairs.csv", help="Output CSV for per-pair metrics")
    ap.add_argument("--summary_out", default="tunnel_summary.csv", help="Output CSV for per-pitcher summary")
    ap.add_argument("--same_ab_only", action="store_true", help="Restrict to pairs within the same AB (default on)")
    ap.add_argument("--no_same_ab_only", dest="same_ab_only", action="store_false")
    ap.set_defaults(same_ab_only=True)
    ap.add_argument("--cross_type_only", action="store_true", help="Keep only cross-type pairs (default on)")
    ap.add_argument("--no_cross_type_only", dest="cross_type_only", action="store_false")
    ap.set_defaults(cross_type_only=True)
    args = ap.parse_args()

    # 1) Read data
    df = pd.read_csv(args.input_csv, na_values=["NULL","null","NA",""])
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    # 2) Sort for sequential pairing
    sort_keys = ["gameid","pitcher","ab","pitchnum"]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # 3) Decision-plane coords by linear interpolation (≈23 ft from plate)
    # If initposy is nonzero: lambda = (initposy - 23)/initposy; else use 0.57 fallback
    dec_plane = 23.0
    lam = np.where(df["initposy"].fillna(0) > 0,
                   (df["initposy"] - dec_plane) / df["initposy"],
                   0.57)
    lam = np.clip(lam, 0.0, 1.0)
    df["x_dec"] = (1 - lam) * df["initposx"] + lam * df["platelocside"]
    df["z_dec"] = (1 - lam) * df["initposz"] + lam * df["platelocheight"]

    # 4) Build next-pitch features (same game/pitcher; usually same AB+batter)
    grp = ["gameid","pitcher"] + (["ab","batter"] if args.same_ab_only else [])
    for col in ["x_dec","z_dec","platelocside","platelocheight","relspeed","pitchname","batter","ab","pitchnum"]:
        df[f"{col}_next"] = df.groupby(grp, dropna=False)[col].shift(-1)

    pairs = df.dropna(subset=["x_dec_next","z_dec_next","platelocside_next","platelocheight_next"]).copy()
    if args.cross_type_only:
        pairs = pairs[pairs["pitchname"] != pairs["pitchname_next"]]

    # 5) Distances and Tunnel Ratio (feet)
    pairs["D_tunnel_ft"] = np.hypot(pairs["x_dec_next"]-pairs["x_dec"],
                                    pairs["z_dec_next"]-pairs["z_dec"])
    pairs["D_plate_ft"]  = np.hypot(pairs["platelocside_next"]-pairs["platelocside"],
                                    pairs["platelocheight_next"]-pairs["platelocheight"])
    eps = 1e-3
    pairs["TR"] = pairs["D_plate_ft"] / (pairs["D_tunnel_ft"] + eps)

    # 6) Select readable columns
    out_cols = [
        "gameid","pitcher","batter","ab","pitchnum","pitchname","pitchname_next",
        "x_dec","z_dec","x_dec_next","z_dec_next","D_tunnel_ft","D_plate_ft","TR"
    ]
    pairs_out = pairs[out_cols].sort_values(["gameid","pitcher","ab","pitchnum"]).reset_index(drop=True)

    # 7) Per-pitcher summary
    summary = (pairs.groupby("pitcher")
                    .agg(median_TR=("TR","median"),
                         q25_TR=("TR", lambda s: s.quantile(0.25)),
                         q75_TR=("TR", lambda s: s.quantile(0.75)),
                         median_tunnel_ft=("D_tunnel_ft","median"),
                         n_pairs=("TR","size"))
                    .reset_index()
                    .sort_values("median_TR", ascending=False))

    # 8) Write outputs
    pairs_out.to_csv(args.pairs_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print(f"✅ wrote {args.pairs_out} (pairs) and {args.summary_out} (summary)")
    print(summary.head(10).to_string(index=False))

if __name__ == "__main__":
    main()