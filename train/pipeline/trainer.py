import os
import torch
from torch.utils.data import DataLoader
from models.TheModel import SimpleMLP
from train.local import train_local
from train.evaluate import evaluate
from train.fedavg import average_models
from train.fedavg_weighted import average_models_weighted
from train.bayesian import build_multiplicative_ensemble
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

class FederatedTrainer:
    def __init__(self, client_datasets, test_dataset, save_path='models/weights', device=None):
        self.client_datasets = client_datasets
        self.test_loader = DataLoader(test_dataset, batch_size=64)
        self.num_clients = len(client_datasets)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.models = []
        self.accuracies = []

    def train_or_load_clients(self, epochs=2, v=False):
        for i, dataset in enumerate(self.client_datasets):
            model_path = os.path.join(self.save_path, f"model_{i}.pt")
            model = SimpleMLP().to(self.device)
            if os.path.exists(model_path):
                print(f"[Cliente {i}] Cargando modelo desde disco.")
                model.load_state_dict(torch.load(model_path))
            else:
                print(f"[Cliente {i}] Entrenando modelo desde cero.")
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                losses, accuracies, test_acc = train_local(model, loader, self.device, epochs, test_loader=self.test_loader)
                torch.save(model.state_dict(), model_path)
                # Save the training history if needed
                np.save(os.path.join(self.save_path, f"losses_{i}.npy"), losses)
                np.save(os.path.join(self.save_path, f"accuracies_{i}.npy"), accuracies)
                np.save(os.path.join(self.save_path, f"test_acc_{i}.npy"), test_acc)
                
            self.models.append(model)
            acc = evaluate(model, self.test_loader, self.device)
            self.accuracies.append(acc)
            print(f"[Cliente {i}] Accuracy en test global: {acc:.4f}")
            # Print the classification report for each client
            y_true = []
            y_pred = []
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            print(classification_report(y_true, y_pred, digits=4))
            # Ver los gráficos de pérdidas y precisión
            if v:
                self.plot_training_history(i)

    def plot_training_history(self, client_id):
        losses = np.load(os.path.join(self.save_path, f"losses_{client_id}.npy"))
        accuracies = np.load(os.path.join(self.save_path, f"accuracies_{client_id}.npy"))
        test_acc = np.load(os.path.join(self.save_path, f"test_acc_{client_id}.npy"))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Pérdida')
        plt.title(f'Cliente {client_id} - Pérdida')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Precisión', color='orange')
        plt.title(f'Cliente {client_id} - Precisión')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()

        if len(test_acc) > 0:
            plt.subplot(1, 2, 2)
            # Reescalamos el eje x para que coincida con el número de épocas
            epochs = np.arange(0, len(test_acc) * 20, 20)
            plt.plot(epochs, test_acc, label='Precisión de prueba', color='green')
            plt.title(f'Cliente {client_id} - Precisión de prueba')
            plt.xlabel('Épocas')
            plt.ylabel('Precisión')
            plt.legend()
            

        plt.tight_layout()
        plt.show()
            

    def aggregate_simple(self):
        return average_models(self.models)

    def aggregate_weighted(self, weights):
        return average_models_weighted(self.models, weights)

    def build_ensemble_model(self, top_k=3):
        return build_multiplicative_ensemble(self.models, self.accuracies, top_k)

    def evaluate_global(self, model, name='Modelo combinado'):
        acc = evaluate(model, self.test_loader, self.device)
        print(f"[Global] {name} Accuracy en test global: {acc:.4f}")
        return acc

    def report(self):
        print("\n--- Reporte por Cliente ---")
        for i, acc in enumerate(self.accuracies):
            print(f"Cliente {i}: {acc:.4f}")
        print(f"Promedio individual: {sum(self.accuracies)/len(self.accuracies):.4f}")

    def classification_report(self, model):
        y_true = []
        y_pred = []
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        print(classification_report(y_true, y_pred, digits=4))
        return classification_report(y_true, y_pred, digits=4)