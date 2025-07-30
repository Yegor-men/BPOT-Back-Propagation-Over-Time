import torch
import random
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


def get_random_mnist_image(dataset):
	"""
	Retrieve a random image and its label from the specified MNIST dataset.

	Args:
		dataset: A PyTorch MNIST dataset object (either train or test)

	Returns:
		image: A tensor of size [28,28] representing the image
		label: An integer representing the class label (0-9)
	"""
	# Select a random index from the dataset
	idx = random.randint(0, len(dataset) - 1)
	# Get the image and label at that index
	image, label = dataset[idx]
	# Remove the channel dimension from [1,28,28] to [28,28]
	image = image.squeeze(0)
	return image, label

# Example usage:
# image, label = get_random_mnist_image(train_dataset)
# print(image.shape)  # torch.Size([28, 28])
# print(label)       # e.g., 5
