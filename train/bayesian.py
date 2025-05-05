import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplicativeEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False  # congelar

    def forward(self, x):
        outputs = [F.softmax(model(x), dim=-1) for model in self.models]
        combined = torch.ones_like(outputs[0])
        for out in outputs:
            combined *= out  # producto punto a punto

        combined /= combined.sum(dim=-1, keepdim=True) + 1e-8  # normalización
        return combined

def build_multiplicative_ensemble(models, accuracies, top_k=3):
    if len(models) != len(accuracies):
        raise ValueError("Modelos y accuracies deben tener misma longitud.")
    
    top_indices = sorted(range(len(accuracies)), key=lambda i: -accuracies[i])[:top_k]
    selected_models = [models[i] for i in top_indices]
    return MultiplicativeEnsemble(selected_models)
