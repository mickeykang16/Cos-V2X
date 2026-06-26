
import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

os.makedirs('data/kmeans', exist_ok=True)
os.makedirs('vis/kmeans', exist_ok=True)

K = 900
DIS_THRESH = 200 # Increased from 55 to 200

fp = 'data/infos/nuscenes_infos_train.pkl'
print(f"Loading {fp}...")
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
center = []
all_sizes = []

print("Collecting boxes...")
for idx in tqdm(range(len(data_infos))):
    # Check if gt_boxes has enough dims
    boxes = data_infos[idx]['gt_boxes']
    if len(boxes) == 0:
        continue
        
    # Distance check
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    mask = distance < DIS_THRESH
    
    if np.sum(mask) > 0:
        center.append(boxes[mask, :3])
        # Assuming w, l, h are at indices 3, 4, 5
        if boxes.shape[1] >= 6:
            all_sizes.append(boxes[mask, 3:6])

center = np.concatenate(center, axis=0)
if len(all_sizes) > 0:
    all_sizes = np.concatenate(all_sizes, axis=0)
    mean_size = np.mean(all_sizes, axis=0)
    print(f"Computed Mean Size (W, L, H): {mean_size}")
else:
    mean_size = np.array([2.0, 4.0, 1.5]) # Falback
    print("Warning: Could not compute sizes, using default.")

print(f"Start clustering {len(center)} points, may take a few minutes.")
cluster = KMeans(n_clusters=K).fit(center).cluster_centers_

plt.figure(figsize=(10, 10))
plt.scatter(cluster[:,0], cluster[:,1], s=1, alpha=0.5)
plt.title(f"K-Means Anchors (K={K})")
plt.axis('equal')
plt.savefig(f'vis/kmeans/det_anchor_{K}.png', bbox_inches='tight')

# Initialize others
# Previous code: 1,1,1,1,0,0,0,0 -> size(3), sin(1), cos(1), vel(2), ?(1) = 8
# But here we use mean_size for first 3.
# Let's keep sin=1, others=0
others = np.concatenate([
    np.tile(mean_size, (K, 1)),         # W, L, H
    np.tile([1, 0, 0, 0, 0], (K, 1))    # sin, cos, vx, vy, ?
], axis=1)

full_anchors = np.concatenate([cluster, others], axis=1)
print(f"Computed anchors shape: {full_anchors.shape}")

save_path = f'data/kmeans/kmeans_det_{K}.npy'
np.save(save_path, full_anchors)
print(f"Saved to {save_path}")
