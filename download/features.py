import os
import os.path
import sys
from os import listdir as ld
from os.path import join as pj
import time

import h5py
from   PIL import Image
from tqdm import tqdm
import numpy as np
import nonechucks as nc

import torch
import torch.utils.data as data
from torchvision import set_image_backend
import torchvision.transforms as transforms

set_image_backend("accimage")


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, exclude=None):
        classes, class_to_idx = self._find_classes(root, exclude)
        print("Found {} classes.".format(len(classes)))
        print("Scanning dataset. This may take some time...")
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        print("Done.")
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir, exclude):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        if exclude:
            classes = list(filter(lambda x: not x in exclude, classes))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, get_id(path)


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        #print("Using accimage")
        return accimage_loader(path)
    else:
        #print("Using PIL")
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, exclude=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform, exclude=exclude,
                                          target_transform=target_transform)
        self.imgs = self.samples

def get_id(p):
    return int(p.split("_")[-1].split(".")[0])

def extract_features(self, x):
    with torch.no_grad():
        x = self.conv1(x.cuda())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x.cpu().numpy()

def extract(model, dl, out_file, progress=100):
    stack = {}
    ds = dl.safe_dataset.dataset
    start = time.time()
    for i,(x,y,z) in enumerate(dl):
        if (i+1)%progress==0:
            end = time.time()
            print("{} batches done in {}s".format(progress, int(start-end)))
            start = end
        
        y   = y.numpy()
        z   = z.numpy()
        out = extract_features(model, x)
        current_idx = set(y)

        ## Finished idx
        for idx in set(stack) - current_idx:
            # Dump
            wnid = ds.classes[idx]
            im_data = np.concatenate(stack[idx][0])
            id_data = np.concatenate(stack[idx][1])
            with h5py.File(out_file, "a") as f:
                f.create_dataset("/images/{}".format(wnid), data=im_data)
                f.create_dataset("/idx/{}".format(wnid), data=id_data)
            # Remove from stack
            del stack[idx]

        for idx in current_idx:
            msk = y==idx
            if idx in stack: # Already seen idx
                stack[idx][0].append(out[msk])
                stack[idx][1].append(z[msk])
            else: # New idx
                stack[idx]=([out[msk]], [z[msk]])
                
    for idx in set(stack):
        wnid = ds.classes[idx]
        im_data = np.concatenate(stack[idx][0])
        id_data = np.concatenate(stack[idx][1])
        with h5py.File(out_file, "a") as f:
            f.create_dataset("/images/{}".format(wnid), data=im_data)
            f.create_dataset("/idx/{}".format(wnid), data=id_data)
            
            
# Test
class Sampler(data.Sampler):
    def __init__(self, wnid, ds):
        self.mask = np.where(np.array(ds.targets)==ds.class_to_idx[wnid])[0]

    def __iter__(self):
        return iter(self.mask)

    def __len__(self):
        return len(self.mask)

def test_extraction(model, dl, indir, outpath, nids=10):
    test_class_number(indir, outpath)
    test_sample_number(indir, outpath)
    test_id_match(indir, outpath)
    test_random_features(model, dl, indir, outpath, nids)
    
def test_class_number(indir, outpath):
    print("Testing Class numbers...")
    with h5py.File(outpath, "r") as f:
        nfeat=len(f["images"])
        nlabel=len(f["idx"])
    nimgs = len(os.listdir(indir))
    assert nfeat==nlabel==nimgs
    print("Done.")
    
def test_sample_number(indir, outpath):
    print("Testing Sample numbers...")
    wnids = os.listdir(indir)
    with h5py.File(outpath, "r") as f:
        wnids = list(f["images"].keys())
        for wnid in tqdm(wnids):
            nimgs  = len(os.listdir(os.path.join(indir, wnid)))
            nfeat  = len(f["images/{}".format(wnid)])
            nlabel = len(f["idx/{}".format(wnid)])
            assert nfeat==nlabel==nimgs
    print("Done.")

def test_id_match(indir, outpath):
    print("Testing ID match...")
    wnids = os.listdir(indir)
    with h5py.File(outpath, "r") as f:
        #wnids = list(f["images"].keys())
        for wnid in tqdm(wnids):
            target_fn = set(os.listdir(os.path.join(indir, wnid)))
            out_fn = set(map(lambda x: wnid + "_" + str(x) +  ".JPEG", f["idx/{}".format(wnid)])) 
            assert target_fn==out_fn
    print("Done.")

def test_random_features(model, dl, indir, outpath, nids):
    print("Testing Random Features...")
    wnids = os.listdir(indir)
    #with h5py.File(outpath, "r") as f:
    #    wnids = list(f["images"].keys())
    for wnid in tqdm(np.random.choice(wnids, nids)):
        test_features(model, dl.dataset, indir, outpath, wnid)
    print("Done.")

def test_features(model, ds, indir, outpath, wnid):
    all_files = map(lambda x: os.path.join(indir, x), os.listdir(indir))
    sampler = Sampler(wnid, ds)
    dl_test = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True, sampler=sampler)
    with h5py.File(outpath, "r") as f:
        for x,y,z in dl_test:
            y = y.numpy()
            z = z.numpy()
            out = extract_features(model, x)
            msk = np.isin(f["idx/{}".format(wnid)], z)
            target = f["images/{}".format(wnid)][:][msk]
            assert np.allclose(out, target, atol=10**-6, rtol=10**-4)
            
normalize = transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])