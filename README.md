# cnn
Este trabajo practico es de la materia de Procesamiento del Aprendizaje Automatico, esta hecho por Zoe Mlinarevic Medl y Ramiro Ottone.

Para agilizar el trabajo en equipo decidimos hacer el desarrollo en un ide local como vscode y subir los cambios a un gestor de versiones como github.


# Arquitectura del modelo CNN

Este proyecto utiliza una red neuronal convolucional (CNN) para clasificar imágenes del dataset CIFAR-10.  
A continuación se describen los componentes principales, sus definiciones y cómo pueden mejorarse:

### Estructura de la red

- **Capas Convolucionales (`Conv2D`)**  
  Extraen características de las imágenes usando filtros llamados **kernels**.  
  - **Kernel (filtro):** Es una pequeña matriz de pesos (por ejemplo, 3x3 o 5x5) que se desliza sobre la imagen para detectar patrones locales, como bordes o texturas.  
  *Mejoras:* aumentar el número de filtros, agregar más capas, probar diferentes tamaños de kernel.

- **Capas de Agrupamiento (`MaxPool2D`)**  
  Reducen la dimensión espacial, conservando la información más relevante.
  - **Tamaño del pool:** Es el tamaño de la ventana utilizada en las capas de agrupamiento (por ejemplo, 2x2). Un pool de 2x2 reduce la dimensión espacial tomando el valor máximo (o promedio) de cada bloque 2x2, ayudando a reducir la cantidad de parámetros y el riesgo de sobreajuste.  
  *Mejoras:* ajustar el tamaño del pool, probar `AveragePooling`.

- **Capa Flatten**  
  Convierte la salida 2D en un vector 1D para conectarla a las capas densas.

- **Capas Densas (`Dense`)**  
  Realizan la clasificación final.  
  *Mejoras:* aumentar neuronas, agregar más capas densas, usar regularización.

- **Regularización:**  
  Son técnicas que ayudan a evitar el sobreajuste del modelo.
  - **Dropout:** Apaga aleatoriamente algunas neuronas durante el entrenamiento.
  - **L2 (weight decay):** Penaliza los pesos grandes en la función de pérdida.
  - **Data augmentation:** Genera nuevas imágenes a partir de las originales aplicando transformaciones (rotación, escalado, etc.).

- **Funciones de Activación (`relu`)**  
  Introducen no linealidad.  
  *Mejoras:* probar otras funciones como `LeakyReLU` o `ELU`.

### Otras técnicas recomendadas

- **Data Augmentation:** Generar más datos artificialmente para mejorar la generalización.
- **Batch Normalization:** Estabiliza y acelera el entrenamiento.
- **Early Stopping:** Detiene el entrenamiento si la validación deja de mejorar.
- **Ajuste de hiperparámetros:** Probar diferentes combinaciones de los parámetros anteriores.

# Implemento
Aumentar la Capacidad del Modelo (Más Filtros y Neuronas)
Un modelo con más filtros y neuronas puede aprender patrones más complejos y detallados de las imágenes, lo que a menudo conduce a una mayor precisión.