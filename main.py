from data.mnist import load_mnist, partition_dataset
from train.pipeline.trainer import FederatedTrainer

def main():
    train_data, test_data = load_mnist()
    client_data = partition_dataset(train_data, num_clients=6)

    federated = FederatedTrainer(client_data, test_data)
    federated.train_or_load_clients()

    federated.report()

    # FedAvg simple
    model_avg = federated.aggregate_simple()
    federated.evaluate_global(model_avg, name='Modelo combinado (FedAvg simple)')
    # [Classification report] (opcional)
    federated.classification_report(model_avg)

    # FedAvg ponderado (por accuracy)
    model_weighted = federated.aggregate_weighted(federated.accuracies)
    federated.evaluate_global(model_weighted, name='Modelo combinado (FedAvg ponderado)')
    # [Classification report] (opcional)
    federated.classification_report(model_weighted)

    # Ensamble bayesiano
    model_ensemble = federated.build_ensemble_model(top_k=3)
    federated.evaluate_global(model_ensemble, name='Modelo combinado (Ensamble bayesiano)')
    # [Classification report] (opcional)
    federated.classification_report(model_ensemble)

if __name__ == '__main__':
    main()
