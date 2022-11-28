import os

#These are all data augmentation things so I don't think we need these, we can find the tensorflow equivalent
# from .rand import Uniform
# from .transforms import Rot90, Flip, Identity, Compose
# from .transforms import GaussianBlur, Noise, Normalize, RandSelect
# from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
# from .transforms import NumpyType

from .data_utils import pkload
import numpy as np

#Make subclass of tensorflow equivalent
class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False,transforms=''):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path + 'data_f32.pkl')
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        #x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        #return [torch.cat(v) for v in zip(*batch)]