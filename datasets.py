from argparse import ArgumentParser
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets
from torchvision import transforms as T
from wilds import get_dataset

class CorrelatedWILDSDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        mode="train",
        transform=None,
        root_dir="./data/",
        size=96,  # 448 for iwildcam
        domains=[0, 1],
        normalize=False,
        seed=42,
    ):
        torch.manual_seed(seed)
        self.normalize = normalize
        self.transform = transform
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.preprocess = T.Compose(
            [
                T.Resize((size, size)),
                T.ToTensor(),
            ]
        )

        self.domains = domains
        dataset = get_dataset(dataset=dataset_name, download=True)
        data = dataset.get_subset(mode)
        self.data_dir = os.path.join(root_dir, f"{dataset_name}_v{dataset.version}")

        # indexes into 'hospital' field (camelyon) or 'location' (iwildcam)
        idx1 = (data.metadata_array[:, 0] == domains[0])
        idx2 = (data.metadata_array[:, 0] == domains[1])
        self.labels = data.y_array[idx1 | idx2]
        files = np.array(data.dataset._input_array)
        self.paths = files[data.indices[idx1 | idx2]]
        self.domains = data.metadata_array[idx1 | idx2, 0]

    def get_correlation_sampler(self, spurious_match_prob):
        """
            This sampler needs to satisfy:

            Pr[Y == Z] = spurious_match_prob
        """
        assert spurious_match_prob >= 0 and spurious_match_prob <= 1
        weights = (self.labels == self.domains) * (2 * spurious_match_prob - 1) + (1 - spurious_match_prob)
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler

    def __getitem__(self, idx):
        X = self.get_input(idx)
        X = self.preprocess(X)
        if self.transform:
            X = self.transform(X)
        if self.normalize:
            X = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
            )(X)
        y = self.labels[idx]
        z = self.domains[idx]
        return X, y, z

    def __len__(self):
        return len(self.labels)


class CorrelatedCamelyon17(CorrelatedWILDSDataset):
    def __init__(self, **kwargs):
        super().__init__('camelyon17', size=96, **kwargs)

    def get_input(self, idx):
        img_filename = os.path.join(
           self.data_dir,
           self.paths[idx])
        return Image.open(img_filename).convert('RGB')


class CorrelatedIWildcam(CorrelatedWILDSDataset):
    def __init__(self, **kwargs):
        super().__init__('iwildcam', size=448, **kwargs)

    def get_input(self, idx):
        img_path = self.data_dir / 'train' / self.paths[idx]
        return Image.open(img_path)


class CorrelatedMNIST(Dataset):
    def __init__(
        self,
        mode="train",
        transform=None,
        spurious_feature_fn=T.RandomRotation(degrees=90),
        spurious_match_prob=0.7,
        digits=[2, 8],
        normalize=False,
        seed=42,
    ):
        torch.manual_seed(seed)

        self.mode = mode
        self.normalize = normalize
        self.transform = transform

        self.spurious_feature_fn = spurious_feature_fn
        self.spurious_match_prob = spurious_match_prob
        self.digits = digits
        if mode == "train":
            data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
            )
        elif mode == "test":
            data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
            )
        else:
            raise ValueError("Invalid mode.")
        idx1 = (data.targets == digits[0])
        idx2 = (data.targets == digits[1])
        self.labels = data.targets[idx1 | idx2]
        self.labels[self.labels == digits[0]] = 0
        self.labels[self.labels == digits[1]] = 1
        self.images = data.data[idx1 | idx2]

    def __getitem__(self, idx):
        X = self.images[idx].unsqueeze(0).float()
        X = X.repeat([3, 1, 1])
        y = self.labels[idx]
        """
            Randomly draw domain:

            z ~ Ber(sqrt(spurious_match_prob) * y + (1 - sqrt(spurious_match_prob)) * (1-y))
        """
        domain_threshold = self.spurious_match_prob if y else 1 - self.spurious_match_prob
        z = (torch.rand(1) < domain_threshold).int().squeeze()
        if z == 1:
            X = self.spurious_feature_fn(X)
        if self.transform:
            X = self.transform(X)
        if self.normalize:
            X = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )(X)
        return X, y, z 

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    print("Testing correlation...")

    psr = ArgumentParser()
    psr.add_argument("--dataset", type=str, choices=['mnist', 'wilds'], default='mnist')
    psr.add_argument("--corr", type=float, default=0.7)
    psr.add_argument("--tol", type=float, default=0.1)
    psr.add_argument("--seed", type=int, default=42)
    args = psr.parse_args()

    if args.dataset == 'mnist':
        dataset = CorrelatedMNIST(
                spurious_match_prob=args.corr,
                seed=args.seed,
            )
        dataloader = DataLoader(dataset, batch_size=1000)
    elif args.dataset == 'wilds':
        dataset = CorrelatedCamelyon17(seed=args.seed)
        dataloader = DataLoader(dataset, batch_size=100, sampler=dataset.get_correlation_sampler(args.corr))
    else:
        raise ValueError(f"Dataset `{args.dataset}` not supported.")

    _, all_y, all_z = next(iter(dataloader))

    print("Results:")
    p_match = np.count_nonzero(all_y == all_z) / len(all_z)
    print("% match:", p_match)
    if np.abs(p_match - args.corr) > args.tol:
        print(f"FAIL (tol: {args.tol})")
    else:
        print("OK")



'''
# Sample code for visulization:

labels_map = {2: "2", 8: "8"}
training_data = baseline_dataloader("train", transform_dict["train"])
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''
