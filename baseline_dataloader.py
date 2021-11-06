import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T


class CorrelatedMNIST(Dataset):
    def __init__(
        self,
        mode="train",
        spurious_feature_fn=T.RandomRotation(degrees=90),
        spurious_corr_coeff=0.7,
        digits=[2, 8],
        normalize=False,
        seed=42,
    ):
        torch.manual_seed(seed)

        self.mode = mode
        self.normalize = normalize

        self.spurious_feature_fn = spurious_feature_fn
        self.spurious_corr_coeff = spurious_corr_coeff
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
        X = self.images[idx].unsqueeze(0)
        y = self.labels[idx]
        """
            Randomly draw domain:

            z ~ Ber(spurious_corr_coeff * y + (1 - spurious_corr_coeff) * (1-y))
        """
        domain_threshold = self.spurious_corr_coeff if y else 1 - self.spurious_corr_coeff
        z = (torch.rand(1) < domain_threshold).int()
        if z:
            X = self.spurious_feature_fn(X)
        if self.normalize:
            X = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )(X)
        return X, y, z 

    def __len__(self):
        return len(self.labels)





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
