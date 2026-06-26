import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

K = 6
SR = 5

fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
navi_trajs = [[], [], []]
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    
    fut_masks_raw = info['gt_ego_fut_masks']
    fut_masks_reshaped = fut_masks_raw.reshape(12, SR)
    plan_mask = fut_masks_reshaped.min(axis=1)
    
    trajs_raw = info['gt_ego_fut_trajs']
    trajs_reshaped = trajs_raw.reshape(12, SR, 2)
    trajs = trajs_reshaped.sum(axis=1)
    plan_traj = trajs.cumsum(axis=-2)

    cmd = info['gt_ego_fut_cmd'].astype(np.int32)
    cmd = cmd.argmax(axis=-1)
    if not plan_mask.sum() == 12:
        continue
    navi_trajs[cmd].append(plan_traj)
# clusters = []
# for trajs in navi_trajs:
#     trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
#     cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
#     cluster = cluster.reshape(-1, 6, 2)
#     clusters.append(cluster)
#     for j in range(K):
#         plt.scatter(cluster[j, :, 0], cluster[j, :,1])
# plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
# plt.close()

# clusters = np.stack(clusters, axis=0)
# np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)

clusters = []
for i, trajs in enumerate(navi_trajs):
    # Skip if the trajs list is empty
    if not trajs:
        print(f"Warning: No valid trajectories found for command {i}. Skipping.")
        continue
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 24)
    
    # Handle the case where there are fewer samples than clusters
    if trajs.shape[0] < K:
        print(f"Warning: Not enough samples for command {i} to perform K-Means. Skipping.")
        continue

    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 12, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])

# Moved the plt.savefig call outside the for loop to visualize all clusters at once
plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
plt.close()

# Stack and save only if at least one cluster was generated
if clusters:
    clusters = np.stack(clusters, axis=0)
    np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)
else:
    print("Error: No clusters were generated for any command.")