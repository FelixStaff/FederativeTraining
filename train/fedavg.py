import copy
import torch

def average_models(models):
    """
    Promedia los pesos de una lista de modelos de PyTorch para realizar federated learning.
    """
    if len(models) == 0:
        raise ValueError("La lista de modelos está vacía.")

    # Crear una copia del primer modelo para inicializar el modelo federado
    federated_model = copy.deepcopy(models[0])
    avg_state_dict = federated_model.state_dict()

    # Inicializar acumuladores en cero
    for key in avg_state_dict:
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])

    # Sumar los parámetros de todos los modelos
    for model in models:
        state_dict = model.state_dict()
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # Promediar
    for key in avg_state_dict:
        avg_state_dict[key] /= len(models)

    # Cargar los pesos promediados en el modelo federado
    federated_model.load_state_dict(avg_state_dict)

    return federated_model