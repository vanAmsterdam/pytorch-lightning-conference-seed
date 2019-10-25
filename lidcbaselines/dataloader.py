import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import transforms
from functools import partial

def normalize(x, window = None, level = None, hu_min = -1100, hu_max = 600, center=0.25, clip=False):
    """
    Normalize array to values between 0 and 1, possibly clipping and centering
    """
    assert type(x) is np.ndarray

    if (not window is None) & (not level is None) :
        hu_min = level - (window / 2)
        hu_max = level + (window / 2)

    x -= hu_min               # add zero point
    x = x / (hu_max - hu_min) # scale
    x -= center               # whiten
    if clip:
        x = x.clip(0.0, 1.0)

    return x

def normalize_tensor(x, window = None, level = None, hu_min = -1100, hu_max = 600, center=0.25, clip=False):
    """
    Normalize tensor to values between 0 and 1, possibly clipping and centering
    """
    assert type(x) is torch.Tensor

    if (not window is None) & (not level is None) :
        hu_min = level - (window / 2)
        hu_max = level + (window / 2)

    x -= hu_min               # add zero point
    x = x / (hu_max - hu_min) # scale
    x -= center               # whiten
    if clip:
        x = x.clamp(0.0, 1.0)

    return x

def random_crop(x, num_vox=None, size=63, max_crop_translation=None):
    """
    num_vox: number of voxels to cut off
    size: out size
    max_crop_translation: when size is not None, max translation from center in voxels
    Crop by cutting off a certain number of voxels, or by cropping from the center with random translations
    """
    if num_vox is not None:
        starts = np.random.choice(range(num_vox), replace=True, size=(x.ndim,))
        ends = x.shape - (num_vox - starts)
        for i in range(x.ndim):
            x = x.take(indices=range(starts[i],ends[i]), axis=i)

    else:
        center = np.array(x.shape)/2
        # assert all(np.array(x.shape) > size)
        margins = (np.array(x.shape) - np.array(size)) / 2
        assert all(margins>=0)
        if max_crop_translation is not None:
            margins.clip(min=0, max=max_crop_translation)
        else:
            max_crop_translation = max(x.shape)

        translations = []
        for margin in margins:
            if margin > 0 and max_crop_translation > 0:
                translations.append(np.random.randint(0, max(int(margin), 0), size=1))
            else:
                translations.append(int(0))

        mins = np.floor(center + np.array(translations).reshape(1,-1) - size/2).astype(np.int32).reshape(-1,)
        maxs = (mins + size).astype(np.int32)
        x = x[tuple(slice(min_i, max_i) for min_i, max_i in zip(mins, maxs))]

    return x

def center_crop(x, num_vox=(1,1,1)):
    return x[1:,1:,1:]

def random_flip(x, axes=(0,1)):
    """
    Flip volume along any of the axes
    """
    assert isinstance(x, np.ndarray)
    flips = np.random.randint(0,2, size=len(axes))
    for flip, ax in zip(flips, axes):
        if flip == 1:
            x = np.flip(x, ax)
    
    return x

def random_rotation_90(x):
    """
    randomly rotate image along z-axis with multiple of 90 degrees
    """
    n_flip = np.random.randint(0, 4, size=1)
    x = np.rot90(x, axes=(0,1), k=n_flip)

    return x

def add_channel_dim(x):
    '''
    add a dummy channel dimension
    '''
    return np.expand_dims(x, 0)

def safe_tensor(x):
    """
    create contiguous tensor when needed
    """
    # print(x.shape)
    try:
        x = torch.from_numpy(x)
    except:
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
    return x.float()

def get_tfms(split='train'):
    tfms = [normalize]
    tfms = []
    if split == 'train':
        tfms += [
            random_crop,
            # random_flip,
            random_rotation_90
        ]
    else:
        # tfms += [center_crop]
        tfms += [random_crop]
    tfms += [add_channel_dim, safe_tensor]
    return transforms.Compose(tfms)

class NoduleDataset(Dataset):
    def __init__(self, df, hparams, split='train'):
        df['fpath']  = df.apply(lambda x: Path(x['filedir']) / x['filename'], axis=1)
        self.df      = df
        self.nodules = []
        print(f"loading {len(df)} nodules")
        for fpath in tqdm(df.fpath.values):
            # add a first dimension for the 'channels'
            self.nodules.append(np.load(fpath))
        self.labels  = df.label.values

        self.transform = get_tfms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.transform(self.nodules[idx])
        y = self.labels[idx]
        return x, y