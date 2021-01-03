"""---------------pytorch-data-loading-------------"""
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
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
                image_id = self.image_ids[index]
                records = selfdata[self.data['image_id'] == image_id]
                records = records.reset_index(drop = True)

                dicom = pydicom.
