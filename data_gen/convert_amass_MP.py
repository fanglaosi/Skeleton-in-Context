import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data as data

from angle2joint import ang2joint


def main():
    clip_len = 16   
    amass_input_length = 10
    amass_target_length = 10
    test_shift = 20
    train_shift = 20
    save_root_path = f"data/AMASS"

    amass_anno_dir = 'data/source_data/AMASS/'
    amass_support_dir = 'data/support_data/'
    motion_dim = 54

    # --------------------------------------------- Train --------------------------------------------------------------------------
    amass_dataset = AMASSDataset('train', amass_anno_dir, amass_support_dir, amass_input_length, amass_target_length, motion_dim, train_shift)
    print(f'Train sample count: {len(amass_dataset)}')
    save_path = os.path.join(save_root_path, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(amass_dataset)):
        amass_motion_input, amass_motion_target = amass_dataset[i]
        if max(amass_motion_input.shape[0], amass_motion_target.shape[0]) < clip_len:
            resample_id_input = np.append(np.zeros(clip_len - amass_input_length, dtype=int), np.arange(amass_input_length))
            amass_motion_input = amass_motion_input[resample_id_input]
            resample_id_target = np.append(np.arange(amass_target_length), np.array([amass_target_length-1 for _ in range(clip_len - amass_target_length)]))
            amass_motion_target = amass_motion_target[resample_id_target]
        else:
            raise ValueError('clip_len must be larger than sample length')
        data_dict = {
            "data_input": amass_motion_input.data.numpy(),
            "data_label": amass_motion_target.data.numpy()
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict, myprofile)

    # --------------------------------------------- Test --------------------------------------------------------------------------
    amass_dataset_test = AMASSEval('test', amass_anno_dir, amass_support_dir, amass_input_length, amass_target_length, motion_dim, test_shift)
    print(f'Test sample count: {len(amass_dataset_test)}')
    save_path = os.path.join(save_root_path, 'test')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(amass_dataset_test)):
        amass_motion_input, amass_motion_target = amass_dataset_test[i]
        if max(amass_motion_input.shape[0], amass_motion_target.shape[0]) < clip_len:
            resample_id_input = np.append(np.zeros(clip_len - amass_input_length, dtype=int), np.arange(amass_input_length))
            amass_motion_input = amass_motion_input[resample_id_input]
            resample_id_target = np.append(np.arange(amass_target_length), np.array([amass_target_length-1 for _ in range(clip_len - amass_target_length)]))
            amass_motion_target = amass_motion_target[resample_id_target]
        else:
            raise ValueError('clip_len must be larger than sample length')
        data_dict = {
            "data_input": amass_motion_input.data.numpy(),
            "data_label": amass_motion_target.data.numpy()
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict, myprofile)


class AMASSEval(data.Dataset):
    def __init__(self, split_name, amass_anno_dir, support_dir, amass_input_length, amass_target_length, motion_dim, shift_step):
        super(AMASSEval, self).__init__()
        self._split_name = split_name
        self._amass_anno_dir = amass_anno_dir
        self._support_dir = support_dir

        self._amass_file_names = self._get_amass_names()

        self.amass_motion_input_length = amass_input_length
        self.amass_motion_target_length = amass_target_length

        self.motion_dim = motion_dim
        self.shift_step = shift_step

        self._load_skeleton()
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._amass_file_names)

    def _get_amass_names(self):

        # create list
        seq_names = []
        assert self._split_name == 'test'

        seq_names += open(
            os.path.join(self._support_dir, "amass_test.txt"), 'r'
            ).readlines()

        file_list = []
        for dataset in seq_names:
            dataset = dataset.strip()
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        return file_list

    def _load_skeleton(self):

        skeleton_info = np.load(
                os.path.join(self._support_dir, 'body_models', 'smpl_skeleton.npz')
                )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _collect_all(self):
        self.amass_seqs = []
        self.data_idx = []
        idx = 0
        for amass_seq_name in tqdm(self._amass_file_names):
            amass_info = np.load(amass_seq_name)
            amass_motion_poses = amass_info['poses'] # 156 joints(all joints of SMPL)
            N = len(amass_motion_poses)        
            if N < self.amass_motion_target_length + self.amass_motion_input_length:
                continue

            frame_rate = amass_info['mocap_framerate']  
            sample_rate = int(frame_rate // 25)    
            sampled_index = np.arange(0, N, sample_rate)
            amass_motion_poses = amass_motion_poses[sampled_index]

            T = amass_motion_poses.shape[0]     
            amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
            amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
            amass_motion_poses[:, 0] = 0

            p3d0_tmp = self.p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
            amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), self.parent)
            amass_motion_poses = amass_motion_poses.reshape(-1, 52, 3)[:, 4:22].reshape(T, 54)     

            self.amass_seqs.append(amass_motion_poses)
            valid_frames = np.arange(0, T - self.amass_motion_input_length - self.amass_motion_target_length + 1, self.shift_step)  
            # np.arange(0, T-50-25, 75)

            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))     
            idx += 1
            
    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.amass_motion_input_length + self.amass_motion_target_length)
        motion = self.amass_seqs[idx][frame_indexes]
        motion = motion.reshape(self.amass_motion_input_length + self.amass_motion_target_length, 18, 3)
        amass_motion_input = motion[:self.amass_motion_input_length]        # (input_length, 18, 3)
        amass_motion_target = motion[self.amass_motion_input_length:]       # (target_length, 18, 3)
        return amass_motion_input, amass_motion_target


class AMASSDataset(data.Dataset):
    def __init__(self, split_name, amass_anno_dir, support_dir, amass_input_length, amass_target_length, motion_dim, train_shift, data_aug=True):
        super(AMASSDataset, self).__init__()
        np.random.seed(0)
        if train_shift > 0:
            self.shift_step = train_shift
        else:
            self.shift_step = None
        self._split_name = split_name  
        self._data_aug = data_aug  
        self._support_dir = support_dir

        self._amass_anno_dir = amass_anno_dir

        self._amass_file_names = self._get_amass_names()

        self.amass_motion_input_length = amass_input_length  
        self.amass_motion_target_length = amass_target_length 

        self.motion_dim = motion_dim 
        self._load_skeleton()
        self._all_amass_motion_poses = self._load_all()

    def __len__(self):
        if self.shift_step is not None:
            return len(self.data_idx)
        else:
            return len(self._all_amass_motion_poses)
    
    def _get_amass_names(self):
        # create list
        seq_names = []
        assert self._split_name == 'train'

        seq_names += np.loadtxt(
            os.path.join(self._support_dir, "amass_train.txt"), dtype=str
        ).tolist()

        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        return file_list

    def _load_skeleton(self):

        skeleton_info = np.load(
            os.path.join(self._support_dir, 'body_models', 'smpl_skeleton.npz')
        )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _load_all(self):
        all_amass_motion_poses = []
        if self.shift_step is not None:
            self.data_idx = []
            idx = 0
        for amass_motion_name in tqdm(self._amass_file_names):
            amass_info = np.load(amass_motion_name)
            amass_motion_poses = amass_info['poses']  # 156 joints(all joints of SMPL)
            N = len(amass_motion_poses)
            if N < self.amass_motion_target_length + self.amass_motion_input_length:
                continue

            frame_rate = amass_info['mocap_framerate']  
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            amass_motion_poses = amass_motion_poses[sampled_index]

            T = amass_motion_poses.shape[0]

            if T < self.amass_motion_target_length + self.amass_motion_input_length:
                continue

            amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
            amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
            amass_motion_poses[:, 0] = 0

            p3d0_tmp = self.p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
            amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), self.parent).reshape(-1, 52, 3)[:, 4:22].reshape(T, -1)

            all_amass_motion_poses.append(amass_motion_poses)

            if self.shift_step is not None:
                valid_frames = np.arange(0, T - self.amass_motion_input_length - self.amass_motion_target_length + 1, self.shift_step)
                self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                idx += 1
        if self.shift_step is not None:
            print(len(self.data_idx))

        return all_amass_motion_poses

    def _preprocess(self, amass_motion_feats):
        amass_seq_len = amass_motion_feats.shape[0]
        start = np.random.randint(
            amass_seq_len - self.amass_motion_input_length - self.amass_motion_target_length + 1)
        end = start + self.amass_motion_input_length

        amass_motion_input = torch.zeros((self.amass_motion_input_length, amass_motion_feats.shape[1]))
        amass_motion_input[:end - start] = amass_motion_feats[start:end]

        amass_motion_target = torch.zeros((self.amass_motion_target_length, amass_motion_feats.shape[1]))
        amass_motion_target[:self.amass_motion_target_length] = amass_motion_feats[
                                                                end:end + self.amass_motion_target_length]

        amass_motion = torch.cat([amass_motion_input, amass_motion_target], axis=0)

        return amass_motion

    def __getitem__(self, index):
        if self.shift_step is not None:
            idx, start_frame = self.data_idx[index]
            frame_indexes = np.arange(start_frame, start_frame + self.amass_motion_input_length + self.amass_motion_target_length)
            motion = self._all_amass_motion_poses[idx][frame_indexes]
            motion = motion.reshape(self.amass_motion_input_length + self.amass_motion_target_length, 18, 3)
            amass_motion_input = motion[:self.amass_motion_input_length].float()        # (input_length, 18, 3)
            amass_motion_target = motion[self.amass_motion_input_length:].float()       # (target_length, 18, 3)
        else:
            amass_motion_poses = self._all_amass_motion_poses[index]
            amass_motion = self._preprocess(amass_motion_poses)     

            amass_motion = amass_motion.reshape(self.amass_motion_input_length+self.amass_motion_target_length, 18, 3)

            amass_motion_input = amass_motion[:self.amass_motion_input_length].float()      # (input_length, 18, 3)
            amass_motion_target = amass_motion[-self.amass_motion_target_length:].float()   # (target_length, 18, 3)
        return amass_motion_input, amass_motion_target
    

if __name__ == "__main__":
    main()
