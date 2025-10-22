import pandas as pd
import numpy as np

pitches_22 = pd.read_csv("updated_pitches_22.csv")
pitches_23 = pd.read_csv("updated_pitches_23.csv")

pitches_22["Year"] = 2022
pitches_23["Year"] = 2023
pitches_all = pd.concat([pitches_22, pitches_23], ignore_index=True)
pitches_all = pitches_all.sort_values(by=["gameid", "ab", "pitchnum"]).reset_index(drop=True)
pitch_counts = pitches_all.groupby("pitcher").size()
out_event = ['field_out']
outs = pitches_all[pitches_all['eventtype'].isin(out_event)]

print(pitches_all.head(10))

num_outs = (
    outs.groupby('pitcher')
    .size()
    .reset_index(name = 'total outs')
    .sort_values(by = 'total outs', ascending= False)

)

print(num_outs.head(15))
