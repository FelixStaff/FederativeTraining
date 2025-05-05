from torchvision import datasets, transforms
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
