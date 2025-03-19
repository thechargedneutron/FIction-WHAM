import os
import joblib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def load_wham_outputs(folder_path):
    cam_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    cam_outputs = {}
    
    for cam_dir in cam_dirs:
        cam_name = cam_dir
        cam_path = os.path.join(folder_path, cam_dir, 'wham_output.pkl')
        if os.path.exists(cam_path):
            cam_outputs[cam_name] = joblib.load(cam_path)
    
    return cam_outputs

def load_tracking_outputs(folder_path):
    cam_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    cam_outputs = {}
    
    for cam_dir in cam_dirs:
        cam_name = cam_dir
        cam_path = os.path.join(folder_path, cam_dir, 'tracking_results.pth')
        if os.path.exists(cam_path):
            cam_outputs[cam_name] = joblib.load(cam_path)
    
    return cam_outputs

def calculate_l2_distance(person_data_cam1, person_data_cam2, idx_cam1, idx_cam2):
    # Calculate L2 distance between poses and betas for the specific indices
    # print(person_data_cam1['pose'].shape)
    # print(person_data_cam1['pose_world'].shape)
    # print(person_data_cam1.keys())
    # exit()
    pose_distances = np.linalg.norm(person_data_cam1['pose_world'] - person_data_cam2['pose_world'])
    # beta_distances = np.linalg.norm(person_data_cam1['betas'] - person_data_cam2['betas'])
    
    total_distance = pose_distances# + beta_distances
    return total_distance

def unify_smpl_outputs(cam_outputs, tracking_outputs):
    unified_dict = defaultdict(dict)
    
    for cam_name, cam_data in cam_outputs.items():
        for person_id, person_data in cam_data.items():
            frame_ids = person_data['frame_ids']
            assert np.array_equal(frame_ids, tracking_outputs[cam_name][person_id]['frame_id'])
            
            for idx, frame_id in enumerate(frame_ids):
                if frame_id not in unified_dict:
                    unified_dict[frame_id] = {}
                
                if cam_name not in unified_dict[frame_id]:
                    unified_dict[frame_id][cam_name] = []
                
                unified_dict[frame_id][cam_name].append({
                    'person_id': person_id,
                    'data': {key: val[idx] for key, val in person_data.items() if key != 'frame_ids'},
                    'frame_idx': idx,  # Store the frame index for later use,
                    'bbox_area': tracking_outputs[cam_name][person_id]['bbox_area'][idx],
                    'num_visible_joints': np.count_nonzero(tracking_outputs[cam_name][person_id]['bbox_mask'][idx])
                })
    
    # Now process the unified_dict to find the best person per frame per camera
    final_unified_dict = defaultdict(dict)
    print("Ok so far...")
    # exit()
    
    for frame_id, cams_data in tqdm(unified_dict.items()):
        for cam_name, persons in cams_data.items():
            if frame_id == 1031 and cam_name == 'cam03':
                print_debug = False
            else:
                print_debug = False

            best_person = None
            largest_area = -1.0

            # Find the dictionary with the highest 'bbox_area' value
            best_person = max(persons, key=lambda x: x['bbox_area'])
            
            if best_person:
                params_to_save = best_person['data'].copy()
                del params_to_save['kp3d']
                params_to_save['num_visible_joints'] = best_person['num_visible_joints']
                final_unified_dict[frame_id][cam_name] = params_to_save
            else:
                print(f"Best person not found for {frame_id}, {cam_name}")
    
    return final_unified_dict

def get_max_visible_joints_camera(frames_dict):
    last_used = {}  # To keep track of the last used camera for tie-breaking
    result = {}  # To store the result for each frame

    # Sort frame indices numerically
    sorted_frames = sorted(frames_dict.keys(), key=lambda x: int(x))

    for frame_index in sorted_frames:
        camera_data = frames_dict[frame_index]
        max_joints = -1
        selected_camera = None

        # Iterate over all cameras for the current frame
        for camera_name, data in camera_data.items():
            visible_joints = data['num_visible_joints']

            # Check if the current camera has more visible joints or if it's a tie
            if visible_joints > max_joints:
                max_joints = visible_joints
                selected_camera = camera_name
            elif visible_joints == max_joints:
                # Handle tie by selecting the least recently used camera
                if (selected_camera is None or 
                    last_used.get(selected_camera, -1) < last_used.get(camera_name, -1)):
                    selected_camera = camera_name

        # Update last used time for the selected camera
        last_used[selected_camera] = int(frame_index)
        # Store the selected camera for the current frame
        result[frame_index] = (selected_camera, max_joints)

    return result


# Folder path to the main directory containing camera subfolders
folder_path = '/path/to/temp_data/WHAM2'

with open('/path/to/Detic/all_takes_list.txt') as f:
    all_takes = [x.strip() for x in f.readlines()]

from tqdm import tqdm
for take_name in tqdm(all_takes):
    # Load the wham outputs
    try:
        cam_outputs = load_wham_outputs(os.path.join(folder_path, take_name))
        tracking_outputs = load_tracking_outputs(os.path.join(folder_path, take_name))
    except:
        print(f"Files not found for {take_name}...")
        continue

    # Unify the outputs into a single dictionary
    unified_dict = unify_smpl_outputs(cam_outputs, tracking_outputs)

    per_camera_count = {}
    results = get_max_visible_joints_camera(unified_dict)
    for key, val in results.items():
        if val[0] not in per_camera_count:
            per_camera_count[val[0]] = 0
        per_camera_count[val[0]] += 1
    # print(per_camera_count)
    best_camera_sorted = sorted(per_camera_count, key=per_camera_count.get, reverse=True)
    # Step 2: Iterate over frames in dict2
    for frame, frame_data in unified_dict.items():
        best_key = None
        # Step 3: Find the best key present in this frame
        for key in best_camera_sorted:
            if key in frame_data:
                best_key = key
                break    
        # Step 4: Keep only the best key in the current frame
        if best_key:
            unified_dict[frame] = {best_key: frame_data[best_key]}
        else:
            unified_dict[frame] = {}  # If no key from dict1 is in the frame


    # Now `unified_dict` contains the unified data indexed by frame_id
    # You can save this dict or use it as needed
    joblib.dump(unified_dict, f'/path/to/WHAM_largest_area_2/{take_name}_largest_area.pkl')
