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
│   ├── weights/             # Pesos de los modelos locales
│   │   ├── model_*.pt
│   └── TheModel.py                  # Red neuronal simple (2 capas densas, sin convs)
├── train/
│   ├── pipeline/               # Entrenador federado
│   │   └── trainer.py
│   ├── local.py                # Entrenamiento local para cada cliente
│   ├── evaluate.py             # Evaluación por accuracy
│   ├── fedavg.py               # Agregación clásica por promedio
│   ├── fedavg_weighted.py      # FedAvg ponderado por accuracy
│   └── bayesian.py  # Ensemble bayesiano multiplicativo (top-k) Explicado abajo
├── main.py                     # Pipeline principal donde se comparan los resultados de los 3 modelos
├── main.ipynb                  # Notebook para correr el pipeline de cada modelo global de aprendizaje federado
├── local_training.ipynb        # Notebook para entrenamiento local de cada cliente
├── requirements.txt
└── README.md
```

## Flujo de trabajo por integrante

Primero instala las dependencias con:

```bash
pip install -r requirements.txt
```

Cada integrante del equipo debe ejecutar el archivo `train/local.ipynb`, proporcionando la ruta a su propia **base de datos particionada**, que debe estar preprocesada de forma estadísticamente equivalente al resto y cuando se cree el modelo cambie el nombre para que coincida con el cliente que lo esta ejecutando.

Esto entrenará un modelo local con los datos de ese cliente y guardará los pesos resultantes como:

```bash
models/weights/model_*.pt
```

Una vez que **todos los modelos locales han sido entrenados y guardados**, se puede ejecutar el archivo `main.py` o `main.ipynb` para realizar la agregación federada, evaluación con el conjunto MNIST completo y visualización de los resultados pero es opcional:

```bash
python main.py
```
Esto entrenará (o cargará desde disco) los modelos locales para 10 clientes, y luego construirá los modelos globales utilizando distintos métodos de agregación.

El otro metodo y el RECOMENDABLE es que corra el archivo `main.ipynb` el cual hara las comparativas entre los modelos globales y locales, mostrando los resultados de cada uno de ellos. Este archivo es el que se recomienda para correr el flujo completo de entrenamiento local, evaluación y agregación global.


> Nota: no es necesario volver a entrenar los modelos locales si los archivos `.pt` ya existen, ya lo cargara desde el disco. Si se desea volver a entrenar, simplemente elimine los archivos `.pt` en `models/weights/` y ejecute nuevamente el notebook.


## Métodos de agregación implementados

- FedAvg: promedio clásico de parámetros $\theta = \frac{1}{N} \sum_{i=1}^{N} \theta_i$.
- FedAvg ponderado: cada cliente aporta proporcionalmente a su accuracy, $\theta = \frac{1}{N} \sum_{i=1}^{N} \frac{acc_i}{\sum_{j=1}^{N} acc_j} \theta_i$.
- Multiplicative Ensemble: se toman los top-k modelos por accuracy, se combinan sus salidas probabilísticas vía multiplicación y se normalizan, generando una decisión robusta con $P(y|x) = \frac{1}{k} \Pi_{i=1}^{k} P(y|x, \theta_i)$.

## Archivos clave

- `main.py`: ejecuta el flujo completo de entrenamiento local, evaluación y agregación global.
- `local.ipynb`: notebook para entrenamiento local de cada cliente.
- `train/pipeline/trainer.py`: clase `FederatedTrainer` que gestiona la lógica federada.
- `train/local.py`: función de entrenamiento individual.
- `train/evaluate.py`: cálculo del accuracy global.
- `models/mlp.py`: modelo utilizado.
- `train/bayesian.py`: implementación del ensemble bayesiano multiplicativo.
- `train/fedavg*.py`: métodos de agregación.

## Confidencialidad

La división de datos entre integrantes se hace fuera del repositorio, como lo exige el enunciado del proyecto. Esta versión es totalmente funcional pero simula localmente la división para que cada quien corra el codigo con 1 cliente y pase sus resultados al resto y se guarde dentro de los `models/weights/` para que cada quien pueda correr el código y obtener los resultados.
