
# Trabajo de Titulación: Automatización de la Recomendación de Tareas para Gestores de Pacientes Adultos Mayores en Sistema de Atención Primaria de Salud Mediante Aprendizaje por Refuerzo

## Descripción
Este proyecto se centra en el desarrollo de un modelo de aprendizaje por refuerzo, específicamente utilizando el algoritmo DQN (Deep Q-Network), escrito en Python. El propósito principal es reorganizar las tareas que debe realizar un agente en un simulador, optimizando así la gestión de pacientes adultos mayores en el sistema de atención primaria de salud. Este modelo se conecta con un simulador a través de una API, facilitando una interacción eficiente y efectiva.

## Tecnologías Utilizadas
- Python
- TensorFlow
- NumPy
- argparse
- json
- os
- random
- requests
- time
- collections.deque
- tensorflow.keras

## Instalación
Para utilizar este proyecto, necesitas instalar las siguientes librerías de Python:

```
pip install tensorflow numpy argparse json os random requests time collections tensorflow.keras
```

## Cómo Usarlo
Para ejecutar el modelo, usa el siguiente comando:

```
python main.py [modo] [config] [time_mode]
```

- `modo`: Modo de ejecución del script. Puede ser 1, 2, 3, 4 o 5.
- `config`: Modo de configuración del script. Puede ser 0, 1, 2, 3 o 4.
- `time_mode`: Modo de tiempo variable. Puede ser 0 (desactivado) o 1 (activo).
