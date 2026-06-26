import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

CLASSES = [
    "car",
    "pedestrian",
    "bicycle",
]

def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = - boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    return trajs_new

K = 6
DIS_THRESH = 55
SR = 5 # sampling ratio: if SR=5, then waypoint hz 10 hz -> 2 hz

fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
intention = dict()
for i in range(len(CLASSES)):
    intention[i] = []
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    boxes = info['gt_boxes']
    names = info['gt_names']
    fut_masks_raw = info['gt_agent_fut_masks']
    trajs_raw = info['gt_agent_fut_trajs']
    
    # if 'gt_ego_fut_trajs' in info:
    #     trajs_raw = trajs_raw + info['gt_ego_fut_trajs'][None]

    velos = info['gt_velocity']
    # (num_agents, 60) -> (num_agents, 12, 5)
    fut_masks_reshaped = fut_masks_raw.reshape(fut_masks_raw.shape[0], 12, SR)
    # A group is valid only if all SR steps are valid; if any step is 0, mark the group invalid (use min).
    fut_masks = fut_masks_reshaped.min(axis=2)

    # (num_agents, 60, 2) -> (num_agents, 12, 5, 2)
    trajs_reshaped = trajs_raw.reshape(trajs_raw.shape[0], 12, SR, 2)
    # Sum the per-step offsets within each group into a single 0.5s-interval displacement.
    # gt_agent_fut_trajs stores per-step offsets, so summing groups of SR steps yields
    # the total displacement (x_diff, y_diff) over 0.5s, giving trajs of shape (num_agents, 12, 2).
    # lidar2agent then applies cumsum over these offsets, so trajs must be offsets here.
    trajs = trajs_reshaped.sum(axis=2)

    labels = []
    for cat in names:
        if cat in CLASSES:
            labels.append(CLASSES.index(cat))
        else:
            labels.append(-1)
    labels = np.array(labels)
    if len(boxes) == 0:
        continue    
    for i in range(len(CLASSES)):
        cls_mask = (labels == i)
        box_cls = boxes[cls_mask]
        fut_masks_cls = fut_masks[cls_mask]
        trajs_cls = trajs[cls_mask]
        velos_cls = velos[cls_mask]

        distance = np.linalg.norm(box_cls[:, :2], axis=1)
        
        # Speed Filtering
        final_disp = np.linalg.norm(trajs_cls.sum(axis=1), axis=1)
        
        # Car: no limit (1000), Ped: 15m, Bike: 50m
        speed_lim = 1000 if i == 0 else (15 if i == 1 else 50)
        
        mask = np.logical_and(
            fut_masks_cls.sum(axis=1) == 12,
            distance < DIS_THRESH,
        )
        mask = np.logical_and(mask, final_disp < speed_lim)
        
        trajs_cls = trajs_cls[mask]
        box_cls = box_cls[mask]
        velos_cls = velos_cls[mask]

        trajs_agent = lidar2agent(trajs_cls, box_cls)
        if trajs_agent.shape[0] == 0:
            continue
        
        # Check for extreme values that survived the filtering above.
        if i == 1:
            final_disp_last = np.linalg.norm(trajs_agent[:, -1, :], axis=1) # distance of the last (12th) point, i.e. 6s ahead
            if final_disp_last.max() > 20:
                 print(f"Warning: Found pedestrian with >20m displacement after filtering!")
                 # Apply one more filtering pass here.
                 valid_mask = final_disp_last <= 20
                 trajs_agent = trajs_agent[valid_mask]
                 if trajs_agent.shape[0] == 0: continue

        intention[i].append(trajs_agent)

# clusters = []
# for i in range(len(CLASSES)):
#     if not intention[i]:
#         print(f"Warning: No valid trajectories found for class {CLASSES[i]}. Skipping.")
#         clusters.append(None) # or some other suitable handling
#         continue
    
#     intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, 24)
#     if intention_cls.shape[0] < K:
#         continue
#     cluster = KMeans(n_clusters=K).fit(intention_cls).cluster_centers_
#     cluster = cluster.reshape(-1, 12, 2)
#     clusters.append(cluster)
#     for j in range(K):
#         plt.scatter(cluster[j, :, 0], cluster[j, :,1])
#     plt.savefig(f'vis/kmeans/motion_intention_{CLASSES[i]}_{K}', bbox_inches='tight')
#     plt.close()

# clusters = np.stack(clusters, axis=0)
# np.save(f'data/kmeans/kmeans_motion_{K}.npy', clusters)

clusters = []
for i in range(len(CLASSES)):
    if not intention[i]:
        print(f"Warning: No valid trajectories found for class {CLASSES[i]}. Skipping.")
        # clusters.append(None) # removed this line.
        continue
    
    intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, 24)
    print(f"Class {CLASSES[i]}: {intention_cls.shape[0]} samples")

    # Debug: Check statistics before KMeans
    if CLASSES[i] == 'pedestrian':
        # intention_cls shape: (N, 24) -> (N, 12, 2)
        trajs_debug = intention_cls.reshape(-1, 12, 2)
        # Calculate final displacement for each trajectory (L2 norm of the last point)
        final_disps = np.linalg.norm(trajs_debug[:, -1, :], axis=1)
        
        print(f"--- [DEBUG] Pedestrian Statistics Before KMeans ---")
        print(f"Max Final Displacement: {final_disps.max():.4f} m")
        
        # Save scatter plot of raw data for debugging
        plt.figure(figsize=(10, 10))
        plt.scatter(trajs_debug[:1000, -1, 0], trajs_debug[:1000, -1, 1], s=1, alpha=0.5, label='End Points (Sample 1000)')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.legend()
        plt.title(f"Pedestrian End Points Distribution (Max Disp: {final_disps.max():.2f}m)")
        plt.savefig('vis/kmeans/debug_pedestrian_scatter.png')
        plt.close()
        print("Saved debug scatter plot to vis/kmeans/debug_pedestrian_scatter.png")

        # Check if there are any outliers > 20m
        outliers = final_disps[final_disps > 20.0]
        if len(outliers) > 0:
            print(f"WARNING: Found {len(outliers)} trajectories with > 20m displacement!")
            print(f"Outlier values: {outliers}")
        else:
            print("No trajectories > 20m found.")
        print("---------------------------------------------------")

    if intention_cls.shape[0] < K:
        print(f"Warning: Not enough samples for class {CLASSES[i]} to perform K-Means. Skipping.")
        continue
    cluster = KMeans(n_clusters=K).fit(intention_cls).cluster_centers_
    cluster = cluster.reshape(-1, 12, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
    plt.savefig(f'vis/kmeans/motion_intention_{CLASSES[i]}_{K}', bbox_inches='tight')
    plt.close()

if clusters:
    clusters = np.stack(clusters, axis=0)
    np.save(f'data/kmeans/kmeans_motion_{K}.npy', clusters)
else:
    print("Error: No clusters were generated for any class.")
