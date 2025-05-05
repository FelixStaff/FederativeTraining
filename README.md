# Federated Learning with MNIST

Este proyecto implementa un sistema de aprendizaje federado utilizando el dataset MNIST en PyTorch. Cada cliente entrena su modelo local de forma independiente, sin compartir datos, y los modelos se combinan usando promedio de pesos (FedAvg). Se incluye evaluación global con `classification_report` para una visión detallada del rendimiento por clase.

## Características

- Partición de datos estratificada (`StratifiedKFold`) para asegurar una distribución equilibrada de clases entre clientes.
- Modelo MLP compacto con `LayerNorm`, `Dropout` y una capa oculta de 128 unidades.
- Promediado de pesos (`FedAvg`) sin sesgos de casting o tipo de tensor.
- Evaluación individual y global de accuracy, además de reporte detallado con `sklearn.metrics.classification_report`.

## Arquitectura del Modelo

```python
class SimpleMLP(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128, bias=bias),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10, bias=bias)
        )

    def forward(self, x):
        return self.net(x)
```

## Estructura

```
project/
│
├── models/
│   └── mlp.py               # Define SimpleMLP
│
├── data/
│   └── mnist.py             # Carga y partición estratificada del MNIST
│
├── train/
│   ├── local.py             # Entrenamiento local de cada cliente
│   ├── evaluate.py          # Evaluación del modelo en el test set
│   └── fedavg.py            # Promedio de modelos
│
└── main.py                  # Entrenamiento federado y evaluación global
```

## Requisitos

- Python ≥ 3.8
- PyTorch ≥ 1.12
- scikit-learn
- torchvision

## Uso

```
python main.py
```

Este ejecutará 3 épocas de entrenamiento local por cliente (por defecto 6 clientes), promediará los modelos y generará métricas globales. Al final se imprime el `classification_report`.

## Futuras Extensiones

- Añadir regularización basada en normas de los pesos.
- Soporte para datos no-IID más extremos.
- Implementación de más rondas federadas y agregación personalizada.
- Guardado automático de resultados y modelos.
