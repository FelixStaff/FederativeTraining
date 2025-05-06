import torch
from torch import optim, nn
from sklearn.metrics import classification_report

def train_local(model, dataloader, device, epochs=2, test_loader=None):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    test_accuracies = []
    for _ in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            # Add the l2 regularization term
            l2_lambda = 0.01  # Regularization strength
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            _, predicted = torch.max(model(x), 1)
            acc = (predicted == y).float().mean()
            accuracies.append(acc.item())
            # See the accuracy of the test
            if test_loader is not None and i % 20 == 0:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for test_x, test_y in test_loader:
                        test_x, test_y = test_x.to(device), test_y.to(device)
                        outputs = model(test_x)
                        _, predicted = torch.max(outputs.data, 1)
                        total += test_y.size(0)
                        correct += (predicted == test_y).sum().item()
                    acc = correct / total
                    test_accuracies.append(acc)
                    
    return losses, accuracies, test_accuracies
