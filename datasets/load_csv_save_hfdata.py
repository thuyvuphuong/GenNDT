#%%
import numpy as np
import torch
import pandas as pd
from datasets import Dataset

# %%
df = pd.read_csv("dataset_0.1mm_6k_samples_v2_improved.csv")

#%%
training_data = []
for _, row in df.iterrows():
    # Create the string: "X_Y_L_dia_D_r_has_crack_zone_i
    label_str = f"X coordinate: {row['X']}, Y coordinate: {row['Y']}, Length: {row['L']}, Diameter: {row['dia']}, Depth: {row['D']}, radius: {row['r']}, Has crack: {row['has_crack']}, Zone ID: {row['zone_id']}"

    # Get signal array from signal_0 to signal_499
    signal_cols = [f"signal_{i:03d}" for i in range(500)]
    signal_array = row[signal_cols].to_numpy(dtype=np.float32)

    # Append as a dict (or tuple if preferred)
    training_data.append({
        "signal": signal_array,
        "text_condition": label_str
    })

# %%
hf_dataset = Dataset.from_list(training_data)

#%%
print(hf_dataset)
print(hf_dataset[0])

# %%
hf_dataset.save_to_disk("datasets/NDTSignal_fulltext")

# %%
