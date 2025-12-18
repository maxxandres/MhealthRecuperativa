# 📘 Documentación Completa del Proyecto HAR (Human Activity Recognition)

## 1. Visión General
Este sistema utiliza Inteligencia Artificial para identificar actividades físicas humanas (como caminar, correr, sentarse) basándose en datos de sensores corporales. El proyecto implementa una arquitectura "Full-Stack" que va desde el procesamiento de datos crudos hasta una interfaz de usuario interactiva para el análisis.

### Arquitectura del Sistema
*   **Frontend (Vue.js):** Dashboard interactivo para cargar archivos, visualizar la línea de tiempo de actividades y comparar datos sensor por sensor.
*   **Backend (FastAPI):** Servidor API que recibe los archivos de log, procesa los datos y ejecuta el modelo de IA.
*   **Pipeline de IA (Python/Scikit-Learn):** Módulo encargado del entrenamiento, extracción de características y exportación del modelo.

---

## 2. El "Corazón" del Sistema: Procesamiento de Datos

El modelo no "ve" el movimiento como nosotros (video), lo "siente" a través de números provenientes de acelerómetros, giroscopios y magnetómetros.

### A. Sensores Utilizados (26 Canales)
Utilizamos el conjunto de datos **MHEALTH**. A diferencia de otros enfoques, nos centramos exclusivamente en el movimiento, descartando señales biomédicas como el ECG.

| Ubicación | Sensores | Canales |
| :--- | :--- | :--- |
| **Pecho** | Acelerómetro (X,Y,Z) | 3 |
| **Tobillo Izq.** | Acelerómetro, Giroscopio, Magnetómetro (X,Y,Z) | 9 |
| **Brazo Der.** | Acelerómetro, Giroscopio, Magnetómetro (X,Y,Z) | 9 |
| **Calculados** | Magnitudes Vectoriales (Acel/Giro de cada parte) | 5 |
| **TOTAL** | | **26 Canales** |

### B. Ventaneo (Windowing)
El movimiento continuo se divide en pequeños fragmentos para ser analizados.
*   **Tamaño de Ventana:** 2.00 segundos (100 muestras a 50Hz).
*   **Solapamiento (Overlap):** 50% (Cada segundo se hace una nueva predicción basada en los últimos 2 segundos).

### C. Ingeniería de Características (Feature Engineering)
El modelo no recibe los 100 datos crudos por sensor (eso sería demasiado ruido). En su lugar, resumimos cada ventana en **182 características matemáticas**.

Para **cada uno de los 26 canales**, calculamos 7 estadísticas:
1.  **Media (Mean):** Indica la dirección promedio (gravedad/orientación).
2.  **Desviación Estándar (Std):** Indica la intensidad del movimiento.
3.  **Mínimo:** Pico más bajo.
4.  **Máximo:** Pico más alto.
5.  **Mediana:** Valor central (robusto a picos aislados).
6.  **Asimetría (Skewness):** ¿La señal se inclina a un lado?
7.  **Energía (FFT):** Ritmo y periodicidad del movimiento.

> **Matemática:** 26 canales × 7 estadísticas = **182 Features**.

### D. Normalización
Antes de entrar al modelo, todos los datos pasan por un **StandardScaler**. Esto convierte los valores a "Z-Scores" (cuántas desviaciones estándar se alejan del promedio), permitiendo comparar peras con manzanas (ej. magnetómetro vs giroscopio).

---

## 3. Entrenamiento del Modelo

El cerebro del sistema es un modelo de **Random Forest**, elegido por su robustez y capacidad para manejar múltiples características.

### Flujo de Entrenamiento (`har_mhealth_pipeline.py`)

1.  **Carga de Datos:** Se leen los archivos `.log` del dataset MHEALTH (sujetos 1-9 para entrenamiento).
2.  **Limpieza:** Se filtran las filas con etiqueta 0 (Null/Sin actividad).
3.  **Extracción de Features:** Se aplica el proceso descrito arriba (ventaneo + cálculo estadístico) para crear una tabla gigante de entrenamiento `(n_muestras, 182)`.
4.  **Escalado:** Se entrena el `StandardScaler` con los datos de entrenamiento y se guardan sus parámetros (`mean`, `scale`).
5.  **Entrenamiento:** Se entrena el clasificador `RandomForestClassifier` con 100 árboles.
    *   *Nota:* Se usa `class_weight='balanced'` para evitar que el modelo se obsesione con las actividades más comunes.
6.  **Exportación:**
    *   El modelo se guarda en formato **ONNX** (`har_mhealth_model.onnx`) para ser universal y rápido.
    *   Los parámetros de escalado se guardan en `scaler_params.json`.

---

## 4. Funcionamiento del Análisis (Inferencia)

Cuando subes un archivo en el Frontend:

1.  **Frontend:** Envía el archivo `.log` al Backend.
2.  **Backend (`/predict/log`):**
    *   Lee el archivo y lo convierte a un DataFrame.
    *   Limpia nulos y calcula las magnitudes.
    *   **Importante:** Genera ventanas **Secuenciales** (cronológicas) para poder reconstruir la línea de tiempo.
    *   Extrae las 182 características por ventana.
    *   **Normaliza** usando el `scaler_params.json` cargado previamente.
    *   Ejecuta el modelo ONNX para predecir la etiqueta.
3.  **Respuesta:** Devuelve un JSON con la línea de tiempo: `[{inicio, fin, predicción, realidad, features}, ...]`.
4.  **Frontend:**
    *   Dibuja las barras de colores.
    *   Calcula estadísticas (errores, duración, actividad dominante).
    *   Permite comparar los **Feature Vectors** de diferentes segmentos para depurar errores (ej. Magnetómetro desviado).

---

## 5. Actividades Reconocidas

El sistema detecta 12 actividades específicas:
1.  De pie (Standing vs. Still)
2.  Sentado (Sitting)
3.  Acostado (Lying down)
4.  Caminando (Walking)
5.  Subiendo escaleras (Climbing stairs)
6.  Flexión de cintura (Waist bends forward)
7.  Elevación de brazos (Frontal elevation of arms)
8.  Flexión de rodillas (Knees bending/Crouching)
9.  Ciclismo (Cycling)
10. Trotar (Jogging)
11. Correr (Running)
12. Saltos (Jump front & back)
