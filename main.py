import torch
from torch.utils.data import DataLoader
from models.mlp import SimpleMLP
from data.mnist import load_mnist, partition_dataset
from train.local import train_local
from train.evaluate import evaluate
from train.fedavg import average_models
from sklearn.metrics import classification_report
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_clients = 6
    epochs = 3

    # Carga de datos y partición de clientes
    train_dataset, test_dataset = load_mnist()
    client_datasets = partition_dataset(train_dataset, num_clients=num_clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    models = []
    local_accuracies = []

    for i, client_data in enumerate(client_datasets):
        client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)

        print(f"[Cliente {i+1}] Entrenando...")
        model = SimpleMLP()
        train_local(model, client_loader, device, epochs=epochs)
        acc = evaluate(model, test_loader, device)
        local_accuracies.append(acc)
        models.append(model)
        print(f"[Cliente {i+1}] Accuracy global: {acc:.4f}")

    print("\n--- Resultados por Cliente ---")
    for i, acc in enumerate(local_accuracies):
        print(f"Cliente {i+1}: {acc:.4f}")
    print(f"Promedio de accuracies individuales: {np.mean(local_accuracies):.4f}")

    global_model = average_models(models)
    global_acc = evaluate(global_model, test_loader, device)
    print(f"\nModelo global (promediado): Accuracy global = {global_acc:.4f}")

    # Classification report
    all_preds = []
    all_labels = []

    global_model.to(device)
    global_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = global_model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == '__main__':
    main()
