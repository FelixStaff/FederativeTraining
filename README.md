# Federated MNIST Learning - ITESM Cloud Computing

Este repositorio contiene una implementación completa de aprendizaje federado sobre la base de datos MNIST, desarrollado para el curso de Cloud Computing (ITESM). El flujo fue construido en PyTorch, adaptando la arquitectura y las reglas del proyecto original de clase.

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
.
├── data/                        # Carga y partición de MNIST
│   └── mnist.py
├── models/                     # Modelos base
│   └── mlp.py                  # Red neuronal simple (2 capas densas, sin convs)
├── train/
│   ├── pipeline/               # Entrenador federado
│   │   └── trainer.py
│   ├── local.py                # Entrenamiento local
│   ├── evaluate.py             # Evaluación por accuracy
│   ├── fedavg.py               # Agregación clásica
│   ├── fedavg_weighted.py      # FedAvg ponderado por accuracy
│   ├── fedmedian.py            # Agregado por mediana
│   └── multiplicative_ensemble.py  # Ensemble bayesiano multiplicativo (top-k)
├── main.py                     # Pipeline principal
├── requirements.txt
└── README.md
```

## Cómo correrlo

Primero instala las dependencias con:

```bash
pip install -r requirements.txt
```

Luego ejecuta el pipeline principal:

```bash
python main.py
```

Esto entrenará (o cargará desde disco) los modelos locales para 10 clientes, y luego construirá los modelos globales utilizando distintos métodos de agregación.

Nota: la partición real de los datos debe hacerse fuera del repositorio para cumplir con los requisitos de confidencialidad. Esta implementación simula eso dividiendo los datos de forma estadísticamente equivalente internamente.

## Métodos de agregación implementados

- FedAvg: promedio clásico de parámetros.
- FedAvg ponderado: cada cliente aporta proporcionalmente a su accuracy.
- FedMedian: se toma la mediana por parámetro entre modelos.
- Multiplicative Ensemble: se toman los top-k modelos por accuracy, se combinan sus salidas probabilísticas vía multiplicación y se normalizan, generando una decisión robusta.

## Archivos clave

- `main.py`: ejecuta el flujo completo de entrenamiento local, evaluación y agregación global.
- `train/pipeline/trainer.py`: clase `FederatedTrainer` que gestiona la lógica federada.
- `train/local.py`: función de entrenamiento individual.
- `train/evaluate.py`: cálculo del accuracy global.
- `models/mlp.py`: modelo utilizado.
- `train/multiplicative_ensemble.py`: implementación del ensemble bayesiano multiplicativo.
- `train/fedavg*.py`: métodos de agregación.

## Confidencialidad

La división de datos entre integrantes se hace fuera del repositorio, como lo exige el enunciado del proyecto. Esta versión es totalmente funcional pero simula localmente la división para que cada quien corra el codigo con 1 cliente y pase sus resultados al resto y se guarde dentro de los `models/weights/` para que cada quien pueda correr el código y obtener los resultados.
