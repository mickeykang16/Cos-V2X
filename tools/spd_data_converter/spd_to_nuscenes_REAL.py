#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
#----------------------------------------------------------------#

import argparse
import shutil
import os
import os.path as osp
import uuid
import pyquaternion

import re
from typing import Dict, Tuple, Any, List, Optional

from spd_to_uniad_REAL import create_spd_infos, _get_instance_token_mappings, create_spd_infos_coop
from spd_to_uniad_REAL import load_json, write_json, visibility_mappings


def generate_category_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating category json ---------------------")
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'category.json')
    target_file_path = osp.join(data_root, version, 'category.json')
    shutil.copy(sr_file, target_file_path)


def generate_attribute_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating attribute json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'attribute.json')
    target_file_path = osp.join(data_root, version, 'attribute.json')
    shutil.copy(sr_file, target_file_path)


def generate_visibility_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating visibility json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'visibility.json')
    target_file_path = osp.join(data_root, version, 'visibility.json')
    shutil.copy(sr_file, target_file_path)


def generate_instance_json(total_annotations, sample_info_mappings, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating instance json ---------------------")
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    category_data = load_json(osp.join(local_root, 'category.json'))
    category_token_mappings = {}
    for category_type in category_data:
        category_token_mappings[category_type['name']] = category_type['token']

    instance_json_datas = []
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        nbr_annotations = len(cur_instance_samples)
        category_name = cur_instance_samples[0]['annotation']['type']

        json_data = {
            'token': instance_token,
            'category_token': category_token_mappings[category_name],
            'nbr_annotations': nbr_annotations,
            'first_annotation_token': cur_instance_samples[0]['annotation']['token'],
            'last_annotation_token': cur_instance_samples[nbr_annotations - 1]['annotation']['token']
        }

        instance_json_datas.append(json_data)

    target_file_path = osp.join(data_root, version, 'instance.json')
    write_json(instance_json_datas, target_file_path)


def generate_sensor_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating sensor json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'sensor.json')
    target_file_path = osp.join(data_root, version, 'sensor.json')
    shutil.copy(sr_file, target_file_path)


## UniV2X TODO: reduce hard code
def generate_calibrated_sensor_json(spd_infos, data_root, version='v1.0-mini', local_root='', v2x_side=None):
    print("--------------------- Start generating calibrated_sensors json ---------------------")

    def gen_token(sensor_name, str_x, str_y, str_z):
        """
            Args:
                sensor_name: "VEHICLE_CAM_FRONT"
                str_x: str label["3d_location"]["x"]
                str_y: str label["3d_location"]["y"]
                str_z: str label["3d_location"]["z"]
            Returns:
                str(token)
        """
        token_name = sensor_name + str_x + str_y + str_z
        token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
        return str(token)

    sensor_data = load_json(osp.join(local_root, 'sensor.json'))
    # calibrated_sensors = ['VEHICLE_CAM_FRONT', 'LIDAR_TOP']
    if v2x_side == "vehicle-side":
        calibrated_sensors = ['cam1_veh1', 'cam2_veh1', 'cam3_veh1', 'cam4_veh1', 'LIDAR_TOP_veh1', 'cam1_veh2', 'cam2_veh2', 'cam3_veh2', 'cam4_veh2', 'LIDAR_TOP_veh2']
    elif v2x_side == "infrastructure-side":
        calibrated_sensors = ['cam1_inf1', 'cam2_inf1', 'LIDAR_TOP_inf1', 'cam1_inf2', 'cam2_inf2', 'LIDAR_TOP_inf2']
    elif v2x_side == "cooperative":
        calibrated_sensors = ['cam1_veh1', 'cam2_veh1', 'cam3_veh1', 'cam4_veh1', 'LIDAR_TOP_veh1', 'cam1_veh2', 'cam2_veh2', 'cam3_veh2', 'cam4_veh2', 'LIDAR_TOP_veh2']

    def parse_scene_ids(scene_name: Any) -> Tuple[str, str]:
        """
        scene_name에서 veh_id, infra_id를 파싱 (folder 뒤 기준).
        예) '..._folder_1_-1' -> ('1', '-1')

        실패 시 'unknown' 반환.
        """
        if scene_name is None:
            return ("unknown", "unknown")
        s = str(scene_name)

        key = "_folder_"
        if key not in s:
            return ("unknown", "unknown")

        _, suffix = s.split(key, 1)  # suffix: '1_-1' (혹은 더 길 수도)
        parts = suffix.split("_")

        veh_id = "unknown"
        infra_id = "unknown"

        if len(parts) >= 1 and re.fullmatch(r"-?\d+", parts[0]):
            veh_id = parts[0]
        if len(parts) >= 2 and re.fullmatch(r"-?\d+", parts[1]):
            infra_id = parts[1]

        return (veh_id, infra_id)

    def build_first_index_maps(spd_infos: List[dict], v2x_side: str):
        veh_first_idx: Dict[str, int] = {}
        infra_first_idx: Dict[str, int] = {}

        for idx, info in enumerate(spd_infos):
            if info is None:
                continue
            scene_name_like = info.get("scene_token", None)
            veh_id, infra_id = parse_scene_ids(scene_name_like)

            # 🔧 FIX 1) key를 'veh{veh_id}'로 저장
            if veh_id != "unknown" and f"veh{veh_id}" not in veh_first_idx:
                veh_first_idx[f"veh{veh_id}"] = idx

            # 🔧 FIX 2) infra는 '-1' -> 'inf1', '-2' -> 'inf2'로 저장
            if infra_id != "unknown":
                if infra_id.startswith("-") and infra_id[1:].isdigit():
                    inf_key = f"inf{infra_id[1:]}"
                else:
                    # 혹시 infra_id가 '1','2'로 오는 케이스도 안전하게 처리
                    inf_key = f"inf{infra_id}" if infra_id.isdigit() else f"inf{infra_id}"
                if inf_key not in infra_first_idx:
                    infra_first_idx[inf_key] = idx

        if v2x_side == "vehicle-side":
            return veh_first_idx
        if v2x_side == "infrastructure-side":
            return infra_first_idx
        if v2x_side == "cooperative":
            return veh_first_idx
        raise ValueError(f"Unknown v2x_side: {v2x_side}")

    first_idx_map = build_first_index_maps(spd_infos=spd_infos, v2x_side=v2x_side) 
    # breakpoint()
    
    def parse_agent_suffix(channel: str) -> Optional[str]:
        """
        'cam1_veh1' -> 'veh1'
        'LIDAR_TOP_inf2' -> 'inf2'
        못 찾으면 None
        """
        if channel is None:
            return None
        s = str(channel)

        # 가장 보수적으로: 끝이 veh숫자 or inf숫자 인지 확인
        m = re.search(r'(veh\d+|inf\d+)$', s)
        if m:
            return m.group(1)
        return None

    def pick_rep_index_from_channel(channel: str, first_idx_map: Dict[str, int]) -> int:
        suffix = parse_agent_suffix(channel)

        if suffix not in first_idx_map:
            raise KeyError(
                f"suffix={suffix} not found in first_idx_map keys={list(first_idx_map.keys())[:10]}..."
            )
        return first_idx_map[suffix]

    def strip_agent_suffix(channel: str) -> str:
        """
        'cam1_veh1' -> 'cam1'
        'LIDAR_TOP_inf2' -> 'LIDAR_TOP'
        """
        if channel is None:
            return channel
        return re.sub(r'_(veh\d+|inf\d+)$', '', str(channel))


    calibrated_sensor_infos = []
    for sensor_info in sensor_data:
        if sensor_info['channel'] in calibrated_sensors:
            spd_infos_first_idx = pick_rep_index_from_channel(sensor_info["channel"], first_idx_map)
            base_ch = strip_agent_suffix(sensor_info["channel"])  # ✅ cam1 / cam2 / cam3 / cam4 / LIDAR_TOP
            if sensor_info['modality'] == 'camera':
                rotation = spd_infos[spd_infos_first_idx]['cams'][base_ch]['sensor2ego_rotation']
                translation = spd_infos[spd_infos_first_idx]['cams'][base_ch]['sensor2ego_translation']
                camera_intrinsic = spd_infos[spd_infos_first_idx]['cams'][base_ch]['cam_intrinsic']
                token = gen_token(sensor_info['channel'], str(translation[0]), str(translation[1]), str(translation[2]))
                # NOTE: 기원 수정. explict 하게 알수 있게
                # token = sensor_info['channel']
                info = {
                    'token': token,
                    'sensor_token': sensor_info['token'],
                    'translation': translation.tolist(),
                    'rotation': rotation.tolist(),
                    'camera_intrinsic': camera_intrinsic.tolist()
                }
            else:
                # breakpoint()
                rotation = spd_infos[spd_infos_first_idx]['lidar2ego_rotation']
                translation = spd_infos[spd_infos_first_idx]['lidar2ego_translation']
                token = gen_token(sensor_info['channel'], str(translation[0]), str(translation[1]), str(translation[2]))
                # token = sensor_info['channel']
                info = {
                    'token': token,
                    'sensor_token': sensor_info['token'],
                    'translation': translation.tolist(),
                    'rotation': rotation.tolist(),
                    'camera_intrinsic': []
                }
            calibrated_sensor_infos.append(info)
    
    # breakpoint()

    target_file_path = osp.join(data_root, version, 'calibrated_sensor.json')
    write_json(calibrated_sensor_infos, target_file_path)


def generate_ego_pose_json(spd_infos, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating ego_pose json ---------------------")

    ego_pose_infos = []
    # breakpoint()
    for spd_info in spd_infos:
        info = {
            'token': spd_info['token'],
            'timestamp': spd_info['timestamp'],
            'rotation': spd_info['ego2global_rotation'].tolist(),
            'translation': spd_info['ego2global_translation'].tolist()
        }

        ego_pose_infos.append(info)

    target_file_path = osp.join(data_root, version, 'ego_pose.json')
    write_json(ego_pose_infos, target_file_path)


def generate_log_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating log json ---------------------")

    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'log.json')
    target_file_path = osp.join(data_root, version, 'log.json')
    shutil.copy(sr_file, target_file_path)


def generate_scene_json(sample_info_mappings, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating scene json ---------------------")

    scene_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        if scene_token not in scene_mappings.keys():
            scene_mappings[scene_token] = []

        scene_mappings[scene_token].append(sample_info_mappings[sample_token])

    log_data = load_json(osp.join(local_root, 'log.json'))
    log_mappings = {}
    for log in log_data:
        log_mappings[log['location']] = log['token']

    scene_infos = []
    for scene_token in scene_mappings.keys():
        nbr_samples = len(scene_mappings[scene_token])
        # breakpoint()
        location = scene_mappings[scene_token][0]['location']
        # breakpoint()
        info = {
            'token': scene_token,
            'log_token': log_mappings[location],
            'nbr_samples': nbr_samples,
            'first_sample_token': scene_mappings[scene_token][0]['token'],
            'last_sample_token': scene_mappings[scene_token][nbr_samples - 1]['token'],
            'name': scene_token,
            'description': ''
        }

        scene_infos.append(info)

    target_file_path = osp.join(data_root, version, 'scene.json')
    write_json(scene_infos, target_file_path)


## UniV2X TODO: remove the hard code about sensor_name
def generate_sample_json(sample_info_mappings, data_root, version='v1.0-mini', local_root='', v2x_side=None):
    print("--------------------- Start generating sample json ---------------------")

    def gen_token(sensor_name, sample_token):
        """
            Args:
                sensor_name: "VEHICLE_CAM_FRONT"
                str_x: str label["3d_location"]["x"]
                str_y: str label["3d_location"]["y"]
                str_z: str label["3d_location"]["z"]
            Returns:
                str(token)
        """
        token_name = sensor_name + sample_token
        token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
        return str(token)

    # sensor_name = 'VEHICLE_CAM_FRONT'
    if v2x_side == "vehicle-side":
        sensor_names = ['cam1', 'cam2', 'cam3', 'cam4', 'LIDAR_TOP']
    elif v2x_side == "infrastructure-side":
        sensor_names = ['cam1', 'cam2', 'LIDAR_TOP']
    elif v2x_side == "cooperative":
        sensor_names = ['cam1', 'cam2', 'cam3', 'cam4', 'LIDAR_TOP']
    else:
        raise ValueError("Wrong v2x-side")
        
    # sensor_names = ['cam1', 'cam2', 'cam3', 'cam4', 'LIDAR_TOP']        # NOTE: Nuscenes는 LIDAR_TOP을 기준으로 토큰을 생성
    # breakpoint()
    for sample_token in sample_info_mappings.keys():
        for sensor in sensor_names:
            sample_info_mappings[sample_token][f'image_token_{sensor}'] = sample_token + "_" + sensor # gen_token(sensor, sample_token)

    image_sample_infos = []
    for sample_token in sample_info_mappings.keys():
        # for sensor in sensor_names:
        sample_info = sample_info_mappings[sample_token]
        # prev_token = '' if sample_info['prev'] == '' else sample_info_mappings[sample_info['prev']][f'image_token_{sensor}']
        # next_token = '' if sample_info['next'] == '' else sample_info_mappings[sample_info['next']][f'image_token_{sensor}']
        
        info = {
            'token': sample_token,
            'timestamp': sample_info['image_timestamp'],
            'prev': sample_info['prev'], 
            'next': sample_info['next'],
            'scene_token': sample_info['scene_token']
        }

        image_sample_infos.append(info)

    target_file_path = osp.join(data_root, version, 'sample.json')
    write_json(image_sample_infos, target_file_path)


def generate_sample_data_json(spd_infos, data_root, version='v1.0-mini', local_root='', v2x_side=None):
    print("--------------------- Start generating sample data json ---------------------")

    calibrated_sensor_data = load_json(osp.join(data_root, version, 'calibrated_sensor.json'))
    # for calibrated_sensor in calibrated_sensor_data:
    #     if calibrated_sensor['camera_intrinsic'] is not []:
    #         calibrated_sensor_token = calibrated_sensor['token']

    sample_data_infos = []
    # breakpoint()

    # if v2x_side == "vehicle-side":
    #     cam_sensors_names = ['cam1_veh1', 'cam2_veh1', 'cam3_veh1', 'cam4_veh1', 'cam1_veh2', 'cam2_veh2', 'cam3_veh2', 'cam4_veh2']
    #     lidar_sensors_names = ['LIDAR_TOP_veh1', 'LIDAR_TOP_veh2']
    # elif v2x_side == "infrastructure-side":
    #     cam_sensors_names = ['cam1_inf1', 'cam2_inf1', 'cam1_inf2', 'cam2_inf2']
    #     lidar_sensors_names = ['LIDAR_TOP_inf1', 'LIDAR_TOP_inf2']
    # elif v2x_side == "cooperative":
    #     cam_sensors_names = ['cam1_veh1', 'cam2_veh1', 'cam3_veh1', 'cam4_veh1', 'cam1_veh2', 'cam2_veh2', 'cam3_veh2', 'cam4_veh2']
    #     lidar_sensors_names = ['LIDAR_TOP_veh1', 'LIDAR_TOP_veh2']
    # else:
    #     raise ValueError("Wrong v2x-side")

    def parse_scene_ids_from_sample(sample_info):
        """
        sample_info에서 scene_token(또는 scene name)을 읽어서
        veh_id ('1','2',...) / infra_id ('-1','-2',...) 를 반환.
        실패하면 (None, None)
        """
        s = sample_info.get('scene_token', None)
        if s is None:
            return (None, None)
        s = str(s)

        key = "_folder_"
        if key not in s:
            return (None, None)

        suffix = s.split(key, 1)[1]  # e.g. '1_-1'
        parts = suffix.split("_")

        veh_id = parts[0] if len(parts) >= 1 and re.fullmatch(r"\d+", parts[0]) else None
        infra_id = parts[1] if len(parts) >= 2 and re.fullmatch(r"-\d+", parts[1]) else None

        return (veh_id, infra_id)


    for sample_info in spd_infos:
        veh_id, infra_id = parse_scene_ids_from_sample(sample_info)

        # v2x_side 별로 이번 sample이 어떤 agent에 해당하는지 결정
        if v2x_side in ["vehicle-side", "cooperative"]:
            # veh_id가 '1'이면 veh1, '2'이면 veh2
            if veh_id == "1":
                cam_sensors_names_cur = ['cam1_veh1','cam2_veh1','cam3_veh1','cam4_veh1']
                lidar_sensors_names_cur = ['LIDAR_TOP_veh1']
                # lidar_sensors_names_cur = ['LIDAR_TOP']
            elif veh_id == "2":
                cam_sensors_names_cur = ['cam1_veh2','cam2_veh2','cam3_veh2','cam4_veh2']
                lidar_sensors_names_cur = ['LIDAR_TOP_veh2']
                # lidar_sensors_names_cur = ['LIDAR_TOP']
            else:
                # 예상 밖이면 skip or raise (원하는대로)
                continue

        elif v2x_side == "infrastructure-side":
            # infra_id가 '-1'이면 inf1, '-2'이면 inf2
            if infra_id == "-1":
                cam_sensors_names_cur = ['cam1_inf1','cam2_inf1']
                lidar_sensors_names_cur = ['LIDAR_TOP_inf1']
                # lidar_sensors_names_cur = ['LIDAR_TOP']
            elif infra_id == "-2":
                cam_sensors_names_cur = ['cam1_inf2','cam2_inf2']
                lidar_sensors_names_cur = ['LIDAR_TOP_inf2']
                # lidar_sensors_names_cur = ['LIDAR_TOP']
            else:
                continue
        
        
        for calibrated_sensor in calibrated_sensor_data:
            prev_token = '' if sample_info['prev'] == '' else sample_info['prev'] + "_" + calibrated_sensor['sensor_token']
            next_token = '' if sample_info['next'] == '' else sample_info['next'] + "_" + calibrated_sensor['sensor_token']
            if calibrated_sensor['sensor_token'] in lidar_sensors_names_cur:
                lidar_info = {
                    'token': sample_info['token'] + "_" + calibrated_sensor['sensor_token'],
                    'sample_token': sample_info['token'],
                    'ego_pose_token': sample_info['token'],
                    'calibrated_sensor_token': calibrated_sensor['token'],
                    'timestamp': sample_info['timestamp'],
                    'fileformat': 'pcd',
                    'is_key_frame': bool(1),
                    'height': 0,
                    'width': 0,
                    'filename': sample_info['lidar_path'],
                    'prev': prev_token,
                    'next': next_token
                }

                sample_data_infos.append(lidar_info)
            elif calibrated_sensor['sensor_token'] in cam_sensors_names_cur:
                # breakpoint()
                cam_key = calibrated_sensor['sensor_token']
                if cam_key not in sample_info['cams']:
                    cam_key_base = re.sub(r'_(veh\d+|inf\d+)$', '', cam_key)  # cam1
                    if cam_key_base in sample_info['cams']:
                        cam_key = cam_key_base
                    else:
                        # 해당 카메라 데이터가 이 sample에 없으면 skip
                        continue

                filename = sample_info['cams'][cam_key]['data_path']
                cams_info = {
                    'token': sample_info['token'] + "_" + calibrated_sensor['sensor_token'],
                    'sample_token': sample_info['token'],
                    'ego_pose_token': sample_info['token'],
                    'calibrated_sensor_token': calibrated_sensor['token'],      # 얘만 추가해주면 될 거 같음. 새로운 sensor token으로
                    'timestamp': sample_info['timestamp'],
                    'fileformat': 'jpeg',
                    'is_key_frame': bool(1),
                    'height': 1080,
                    'width': 1920,
                    'filename': sample_info['cams'][cam_key]['data_path'],
                    'prev': prev_token,
                    'next': next_token
                }

                sample_data_infos.append(cams_info)
            else: 
                continue
                # raise ValueError("Wrong Type of sensor_token!!")

    target_file_path = osp.join(data_root, version, 'sample_data.json')
    write_json(sample_data_infos, target_file_path)


def rotation_z2quaternion(rotation_z):
    # https://www.zhihu.com/question/23005815/answer/33971127
    import math

    q = [math.cos(rotation_z / 2), 0, 0, 1 * math.sin(rotation_z / 2)]

    return q


def generate_sample_annotation_json(total_annotations, sample_info_mappings, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating sample annotation json ---------------------")
    import numpy as np
    from pyquaternion import Quaternion

    sample_annotation_infos = []
    # breakpoint()
    for sample_token in total_annotations.keys():
        annotations = total_annotations[sample_token]
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        timestamp = sample_info_mappings[sample_token]['timestamp']

        # for info in spd_infos:
        #     if info['token'] == sample_token[-6:]:
        #         sample_lidar2ego_rotation = Quaternion(info['lidar2ego_rotation'])
        #         sample_lidar2ego_translation = np.array(info['lidar2ego_translation'])
        #         sample_ego2global_rotation = Quaternion(info['ego2global_rotation'])
        #         sample_ego2global_translation = np.array(info['ego2global_translation'])
        #         break

        for anno_token in annotations.keys():
            annotation = annotations[anno_token]

            # import pdb; pdb.set_trace()
            
            # cvt global
            center = np.array([annotation['3d_location']['x'], annotation['3d_location']['y'],
                               annotation['3d_location']['z']])
            rot = Quaternion(axis=[0, 0, 1], radians=annotation['rotation'])

            # # lidar2ego
            # center = np.dot(sample_lidar2ego_rotation.rotation_matrix, center) + sample_lidar2ego_translation
            # rot = sample_lidar2ego_rotation * rot
            
            # # ego2global
            # center = np.dot(sample_ego2global_rotation.rotation_matrix, center) + sample_ego2global_translation
            # rot = sample_ego2global_rotation * rot

            # NOTE: visibility_token 은 일단 대부분 다 보인다고 가정하고 넣음.
            # breakpoint()
            info = {
                'token': annotation['token'],
                'sample_token': sample_token,
                'scene_token': scene_token,
                'instance_token': annotation['instance_token'],
                'visibility_token': visibility_mappings[annotation['occluded_state']],
                'attribute_tokens': [],
                'translation': center.tolist(),
                'size': [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                         annotation['3d_dimensions']['h']],
                'rotation': rot.elements.tolist(),
                'prev': annotation['prev'],
                'next': annotation['next'],
                'num_lidar_pts': 100,
                'num_radar_pts': 0
            }

            sample_annotation_infos.append(info)

    target_file_path = osp.join(data_root, version, 'sample_annotation.json')
    write_json(sample_annotation_infos, target_file_path)


def generate_map_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating map json ---------------------")

    log_data = load_json(osp.join(local_root, 'log.json'))
    log_mappings = {}
    for log in log_data:
        log_mappings[log['location']] = log['token']

    map_infos = []
    for location in log_mappings.keys():
        info = {
            'category': 'semantic_prior',
            'token': location,
            'filename': '',
            'log_tokens': [log_mappings[location]]
        }

        map_infos.append(info)

    target_file_path = osp.join(data_root, version, 'map.json')
    write_json(map_infos, target_file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./datasets/v2xreal")
    parser.add_argument('--save-root', type=str, default="./datasets/v2xreal")
    parser.add_argument('--split-file', type=str, default="./data/split_datas_V2XREAL/split_datas_V2XREAL.json")
    parser.add_argument('--local-root', type=str, default="./tools/spd_data_converter/nuscenes_jsons_V2XREAL")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="spd")
    parser.add_argument('--skip-noinfra', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    import os

    curDirectory = os.getcwd()
    basepath = os.path.basename(os.path.normpath(curDirectory))
    if basepath != 'UniV2X_REAL' and os.path.isdir('UniV2X_REAL/'):
        os.chdir('UniV2X_REAL/')

    curDirectory = os.getcwd()
    print(curDirectory)

    v2x_side = args.v2x_side
    version = args.version
    data_root = args.data_root
    save_root = args.save_root
        
    local_root = args.local_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix
    skip_noinfra = args.skip_noinfra

    if v2x_side == 'cooperative':
        print('this is cooperative!')
        total_annotations, sample_info_mappings, spd_infos, spd_infos_infra = create_spd_infos_coop(data_root,
                            save_root,
                            v2x_side,
                            split_path, 
                            can_bus_root_path,
                            info_prefix,
                            version=version,
                            max_sweeps=10,
                            flag_save=False)  
    else:      
        print('this is single!')  
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos(data_root,
                                                                            save_root,
                                                                            v2x_side,
                                                                            split_path,
                                                                            can_bus_root_path,
                                                                            info_prefix,
                                                                            version=version,
                                                                            max_sweeps=10,
                                                                            flag_save=False,
                                                                            skip_noinfra=skip_noinfra)

    # breakpoint()
    
    # NOTE: Inplace 연산 방지용 ======= ##
    if v2x_side == "cooperative":
        import copy
        sample_info_mappings_infra = copy.deepcopy(sample_info_mappings)
        total_annotations_infra = copy.deepcopy(total_annotations)
    ## ============================= ##
    
    save_root = osp.join(args.save_root, v2x_side)

    generate_category_json(save_root,
                        version=version,
                        local_root=local_root)

    generate_attribute_json(save_root,
                            version=version,
                            local_root=local_root)

    generate_visibility_json(save_root,
                            version=version,
                            local_root=local_root)

    generate_instance_json(total_annotations,
                        sample_info_mappings,
                        save_root,
                        version=version,
                        local_root=local_root)

    generate_sensor_json(save_root,
                        version=version,
                        local_root=local_root)

    generate_calibrated_sensor_json(spd_infos,
                                    save_root,
                                    version=version,
                                    local_root=local_root,
                                    v2x_side=v2x_side)

    generate_ego_pose_json(spd_infos,
                        save_root,
                        version=version,
                        local_root=local_root)

    generate_log_json(save_root,
                    version=version,
                    local_root=local_root)

    generate_scene_json(sample_info_mappings,
                        save_root,
                        version=version,
                        local_root=local_root)

    generate_sample_json(sample_info_mappings,
                        save_root,
                        version=version,
                        local_root=local_root,
                        v2x_side=v2x_side)

    generate_sample_data_json(spd_infos,
                            save_root,
                            version=version,
                            local_root=local_root,
                            v2x_side=v2x_side)

    generate_sample_annotation_json(total_annotations,
                                    sample_info_mappings,
                                    save_root,
                                    version=version,
                                    local_root=local_root)

    # NOTE: V2X-REAL에서는 Map 없음
    generate_map_json(save_root,
                    version=version,
                    local_root=local_root)
    
    if v2x_side == 'cooperative':
        save_root_infra = osp.join(args.save_root, v2x_side, "inf_in_coop")
        
        generate_category_json(save_root_infra,
                            version=version,
                            local_root=local_root)

        generate_attribute_json(save_root_infra,
                                version=version,
                                local_root=local_root)

        generate_visibility_json(save_root_infra,
                                version=version,
                                local_root=local_root)

        generate_instance_json(total_annotations_infra,
                            sample_info_mappings_infra,
                            save_root_infra,
                            version=version,
                            local_root=local_root)

        generate_sensor_json(save_root_infra,
                            version=version,
                            local_root=local_root)

        generate_calibrated_sensor_json(spd_infos_infra,
                                        save_root_infra,
                                        version=version,
                                        local_root=local_root,
                                        v2x_side="infrastructure-side")

        generate_ego_pose_json(spd_infos_infra,
                            save_root_infra,
                            version=version,
                            local_root=local_root)

        generate_log_json(save_root_infra,
                        version=version,
                        local_root=local_root)

        generate_scene_json(sample_info_mappings_infra,
                            save_root_infra,
                            version=version,
                            local_root=local_root)

        generate_sample_json(sample_info_mappings_infra,
                            save_root_infra,
                            version=version,
                            local_root=local_root,
                            v2x_side="infrastructure-side")

        generate_sample_data_json(spd_infos_infra,
                                save_root_infra,
                                version=version,
                                local_root=local_root,
                                v2x_side="infrastructure-side")

        generate_sample_annotation_json(total_annotations_infra,
                                        sample_info_mappings_infra,
                                        save_root_infra,
                                        version=version,
                                        local_root=local_root)

        # NOTE: V2X-REAL에서는 Map 없음
        generate_map_json(save_root_infra,
                        version=version,
                        local_root=local_root)