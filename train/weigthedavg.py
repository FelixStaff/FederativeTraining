import copy
import torch

def average_models_weighted(models, weights):
    """Promedia modelos usando pesos externos (deben ser normalizados)."""
    if not models or not weights or len(models) != len(weights):
        raise ValueError("Modelos y pesos deben tener el mismo tamaño.")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("La suma de los pesos no puede ser cero.")
    normalized = [w / total_weight for w in weights]

    averaged_model = copy.deepcopy(models[0])
    state_dicts = [model.state_dict() for model in models]
    avg_state_dict = {}

    for key in state_dicts[0]:
        avg_state_dict[key] = sum(
            state_dicts[i][key].float() * normalized[i] for i in range(len(models))
        )

    averaged_model.load_state_dict(avg_state_dict)
    return averaged_model
