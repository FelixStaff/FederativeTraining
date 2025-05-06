from torchvision import datasets, transforms
# Image folder structure:
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

def load_mnist():
    transform = transforms.ToTensor()
    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train, test

def partition_dataset(dataset, num_clients=10):
    data_per_client = len(dataset) // num_clients
    client_indices = [list(range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]
    return [Subset(dataset, idxs) for idxs in client_indices]

def load_mnist_partitioned(path):
    transform = transforms.ToTensor()
    train = ImageFolder(root=path + '/train', transform=transform)
    test = ImageFolder(root=path + '/test', transform=transform)
    return train, test
    
def load_mnist_size(size=1000):
    transform = transforms.ToTensor()
    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # El subset
    indices = list(range(len(train)))
    subset_indices = indices[:size]
    train = Subset(train, subset_indices)
    return train, test