import pickle
import re
import os

src = 'data/infos/nuscenes_infos_test.pkl'

def convert_path(path, offset='x+0_y+0'):
    # datasets/v2xreal/data/test/{scene}/{agent_id}/{frame}
    # agent_id > 0 -> test_novel/{scene}/{agent_id}_{offset}/{frame}
    # agent_id < 0 -> test_novel_infra/{scene}/{agent_id}_{offset}/{frame}
    def replace_agent(m):
        agent_id = int(m.group(2))
        split = 'test_novel' if agent_id > 0 else 'test_novel_infra'
        return f'/{split}/{m.group(1)}/{m.group(2)}_{offset}/'
    return re.sub(r'/test/([^/]+)/(-?\d+)/', replace_agent, path)

with open(src, 'rb') as f:
    data = pickle.load(f)

for info in data['infos']:
    for cam_info in info['cams'].values():
        cam_info['data_path'] = convert_path(cam_info['data_path'])
    if 'sweeps' in info and isinstance(info['sweeps'], list):
        for sw in info['sweeps']:
            if isinstance(sw, dict) and 'data_path' in sw:
                sw['data_path'] = convert_path(sw['data_path'])

data['infos'] = [
    info for info in data['infos']
    if all(os.path.exists(cam_info['data_path']) for cam_info in info['cams'].values())
]
print(f'Infos after filtering: {len(data["infos"])}')

dst = 'data/infos/nuscenes_infos_test_novel.pkl'
with open(dst, 'wb') as f:
    pickle.dump(data, f)
print(f'Saved: {dst}')

# Verify
d = pickle.load(open(dst, 'rb'))
for info in d['infos'][:3]:
    for cam_type, cam_info in info['cams'].items():
        print(cam_type, '->', cam_info['data_path'])
    print()
