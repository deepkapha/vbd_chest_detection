""'
torch_data.py - module for Torch data loading pipeline
"""

# import dependencies
import os
import torch
import pydicom
import numpy as np
import pandas as pd
import torchvision

from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler

class VBD_Dataseet(Dataset):
        """
        VBD_Dataset - class for loading VinBigDataset
        """

        def __init__(self, data, image_dir, transforms = None):
                """
                VBD_Dataset class constructor
                Inputs:
                        - data : str or pandas.DataFrame
                                DataFrame (or path to dataframe) of image and labels
                        - image_dir : str
                                path to image directory
                        - transforms : to be added
                """
                super().__init__()

                # parse arguments
                self.data = pd.read_csv(data) if isinstance(data, str) else data
                self.image_ids = self.data['image_id'].unique()
                self.image_dir = image_dir
                self.transforms = transforms

        def __getitem__(self, index):
		# retrieve labels
                image_id = self.image_ids[index]
                records = selfdata[self.data['image_id'] == image_id]
                records = records.reset_index(drop = True)

		# read image
		dicom = pydicom.dcmread(f"{self.image_dir}/{image_id}.dicom")
		image = dicom.pixel_array

		if 'PhotometricIntepretation' in dicom:
			if dicom.PhotometricInterpretation == 'MONOCHROME1":
				image = np.max(image) - image
		intercept = dicom.RescaleIntercept if 'RescaleIntercept' in dicom else 0.0
		slope = dicon.RescaleSlope if 'RescaleSlope' in dicom else 1.0

		if slope != 1:
			image = slope * image.astype(np.float64)
			image = image.astype(np.int16)

		image += np.int16(intercept)

		image = np.stack([image, image, image])
		image = image.astype('float32')
		image = image - image.min()
		image = image / image.max()
		image = image * 255.0
		image = image.transpose(1,2,0)

		if records.loc[0, "class_id"] == 0:
			records = records.loc[[0], :]

		boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		area = torch.as_tensor(area, dtype=torch.float32)
		labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

		# suppose all instances are not crowd
		iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

		target = {}
		target['boxes'] = boxes
		target['labels'] = labels
		target['image_id'] = torch.tensor([index])
		target['area'] = area
		target['iscrowd'] = iscrowd

		if self.transforms:
			sample = {
				'image': image,
				'bboxes': target['boxes'],
				'labels': labels
			}
		sample = self.transforms(**sample)
		image = sample['image']
		target['boxes'] = torch.tensor(sample['bboxes'])

		if target["boxes"].shape[0] == 0:
			# Albumentation cuts the target (class 14, 1x1px in the corner)
			target["boxes"] = torch.from_numpy(np.array([[0.0, 0.0, 1.0, 1.0]]))
			target["area"] = torch.tensor([1.0], dtype=torch.float32)
			target["labels"] = torch.tensor([0], dtype=torch.int64)

		return image, target

	def __len__(self):
		return self.image_ids.shape[0]
