from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
from exercise_code.data_utils import get_keypoints
from exercise_code.data_utils import get_image


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform



    def __len__(self):
        #######################################################################
        # TODO:                                                               #
        # Return the length of the dataset                                    #
        #######################################################################

        return len(self.key_pts_frame)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def __getitem__(self, idx):
        sample = {'image': None, 'keypoints': None}
        #######################################################################
        # TODO:                                                               #
        # Return the idx sample in the Dataset. A sample should be a          #
        # dictionary where the key, value should be like                      #
        #        {'image': image of shape [C, H, W],                          #
        #         'keypoints': keypoints of shape [num_keypoints, 2]}         #
        #######################################################################
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # sample['image'] = get_image(idx, self.key_pts_frame)
        # sample['keypoints'] = get_keypoints(idx, self.key_pts_frame)
        image_string = self.key_pts_frame.loc[idx]['Image']
        image = np.array([int(item) for item in image_string.split()]).reshape((1, 96, 96))

        keypoint_cols = list(self.key_pts_frame.columns)[:-1]
        keypoint = self.key_pts_frame.iloc[idx][keypoint_cols].values.reshape((15, 2)).astype(np.float32)

        sample = {'image': image, 'keypoints': keypoint}
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample
