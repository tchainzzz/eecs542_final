import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

transform_dict = {
        'train': transforms.Compose(
            [transforms.RandomCrop(25),
             transforms.RandomRotation(degrees=180),
             transforms.ToTensor(),
			 #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  #std=[0.229, 0.224, 0.225]), for ResNet
			]),
        'test': transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  #std=[0.229, 0.224, 0.225]), for ResNet
             ])}


def baseline_dataloader(mode="train", transform=transform_dict["train"], digits=[2,8]):
	'''
	Return MINIST dataloader.

	mode: "train" or "test"
	transform: transform_dict["train"] or transform_dict["test"]
	digits: 2 digits used as baseline
	
	'''
	if mode == "train":
		data = datasets.MNIST(
			root="data",
			train=True,
			download=True,
			transform=transform_dict["train"]
		)
		idx1 = (data.train_labels==digits[0])
		idx2 = (data.train_labels==digits[1])
		idx = torch.logical_or(idx1, idx2)
		data.targets = data.targets[idx]
		data.data = data.data[idx]
	elif mode == "test":
		data = datasets.MNIST(
			root="data",
			train=False,
			download=True,
			transform=transform_dict["test"]
		)
		idx1 = (data.test_labels==digits[0])
		idx2 = (data.test_labels==digits[1])
		idx = torch.logical_or(idx1, idx2)
		data.targets = data.targets[idx]
		data.data = data.data[idx]
	else:
		print("Invalid mode.")
	return data
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