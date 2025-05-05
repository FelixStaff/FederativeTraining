import os
import torch
from torch.utils.data import DataLoader
from models.mlp import SimpleMLP
from train.local import train_local
from train.evaluate import evaluate
from train.fedavg import average_models
from train.fedavg_weighted import average_models_weighted
from train.bayesian import build_multiplicative_ensemble

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

    def train_or_load_clients(self, epochs=2):
        for i, dataset in enumerate(self.client_datasets):
            model_path = os.path.join(self.save_path, f"model_{i}.pt")
            model = SimpleMLP().to(self.device)
            if os.path.exists(model_path):
                print(f"[Cliente {i}] Cargando modelo desde disco.")
                model.load_state_dict(torch.load(model_path))
            else:
                print(f"[Cliente {i}] Entrenando modelo desde cero.")
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                train_local(model, loader, self.device, epochs)
                torch.save(model.state_dict(), model_path)
            self.models.append(model)
            acc = evaluate(model, self.test_loader, self.device)
            self.accuracies.append(acc)
            print(f"[Cliente {i}] Accuracy en test global: {acc:.4f}")

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
