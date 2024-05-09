import os
import json
import numpy as np
import pandas as pd
import toml
from scipy.spatial.transform import Rotation as R
from PySBA import PySBA

# 2D 포즈 추정 데이터 로드
def load_2d_points(json_folder):
    points2D = []
    for json_file in sorted(os.listdir(json_folder)):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file), 'r') as file:
                data = json.load(file)
                if 'people' in data and len(data['people']) > 0:
                    keypoints = data['people'][0]['pose_keypoints_2d']
                    points = np.array(keypoints).reshape(-1, 3)[:, :2]  # x, y 좌표만 추출
                    points2D.append(points)
                else:
                    print(f"No pose keypoints found in {json_file}")
                    continue
    return np.array(points2D).reshape(-1, 2)  # 평탄화하여 반환

# 3D 포인트 데이터 로드
def load_3d_points(trc_file):
    df = pd.read_csv(trc_file, sep='\t', skiprows=5)
    points3D = df.iloc[:, 2:].to_numpy()  # 첫 두 열은 시간 정보이므로 제외
    return points3D

# 카메라 파라미터 로드
def load_camera_params(toml_file):
    cam_params = []
    data = toml.load(toml_file)
    for key in data:
        if key.startswith('int_cam'):
            cam = data[key]
            matrix = np.array(cam['matrix'])
            distortions = np.array(cam['distortions'])
            rotation = np.array(cam['rotation'])
            translation = np.array(cam['translation']).flatten()
            focal_length = matrix[0, 0]
            k1, k2 = distortions[0], distortions[1]
            rvec = R.from_euler('xyz', rotation).as_rotvec()  # 회전 벡터로 변환
            cam_params.append(np.concatenate([rvec, translation, [focal_length, k1, k2]]))
    return np.array(cam_params)

# 여러 폴더에서 2D 포인트 데이터를 로드
def load_multiple_cameras(json_folders):
    all_points2D = []
    camera_indices = []
    num_frames_per_camera = []
    
    for cam_idx, folder in enumerate(json_folders):
        points2D = load_2d_points(folder)
        if points2D.size == 0:
            print(f"No valid 2D points found in folder {folder}")
            continue
        all_points2D.append(points2D)
        num_frames = len(os.listdir(folder))
        num_points_per_frame = points2D.shape[0] // num_frames
        camera_indices.append(np.repeat(cam_idx, points2D.shape[0]))
        num_frames_per_camera.append(num_frames)

    all_points2D = np.vstack(all_points2D)
    camera_indices = np.hstack(camera_indices)
    
    return all_points2D, camera_indices, num_frames_per_camera[0], num_points_per_frame

# 경로 설정
json_folders = [r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\pose\ex_json1', 
r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\pose\ex_json2',
r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\pose\ex_json3',
r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\pose\ex_json4']  # 각 카메라 폴더 경로
trc_file = r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\pose-3d\S01_Demo_SingleTrial_0-2028.trc'
toml_file = r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\calibration\calib.toml'

# 데이터 로드
points2D_flat, cameraIndices, num_frames, num_points_per_frame = load_multiple_cameras(json_folders)
points3D = load_3d_points(trc_file)
cameraArray = load_camera_params(toml_file)

# point2DIndices 설정
point2DIndices = np.tile(np.arange(num_points_per_frame), num_frames * len(json_folders))

# PySBA 클래스 인스턴스 생성
sba = PySBA(cameraArray, points3D, points2D_flat, cameraIndices, point2DIndices)

# 번들 조정 실행
optimized_camera_params, optimized_points3D = sba.bundleAdjust()

# 최적화된 결과 출력
print("최적화된 카메라 파라미터:\n", optimized_camera_params)
print("최적화된 3D 포인트:\n", optimized_points3D)
