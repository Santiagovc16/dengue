# Taller Final - Aplicación de los conceptos de Analítica de Datos + Machine Learning + IA
# Proyecto: Proyección de casos de Dengue en el Valle del Cauca (2020 - 2025)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# 1. CARGA DEL DATASET
# ----------------------

# Reemplaza con la ruta de tu archivo .csv real cuando lo tengas
DATA_PATH = 'data/dengue_valle.csv'
df = pd.read_csv(DATA_PATH)

# Cargar información de puntos de vacunación contra el dengue
df_vacunacion = pd.read_csv('data/puntos_vacunacion_dengue.csv')

# -----------------------------
# 2. EXPLORACIÓN DE LOS DATOS
# -----------------------------

print("Primeros registros del dataset:")
print(df.head())

print("\nInformación general del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

# -----------------------------
# 2.1 NOMBRES MÁS CLAROS DE COLUMNAS IMPORTANTES
# -----------------------------

columnas_clave = {
    'fec_not': 'Fecha de notificación',
    'Grup_Etario': 'Grupo Etario',
    'clas_edad': 'Clasificación Edad',
    'año': 'Año',
    'sexo_': 'Sexo',
    'tip_ss_': 'Tipo Seguridad Social',
    'nmun_resi': 'Municipio de residencia',
    'fiebre': 'Síntoma: Fiebre',
    'cefalea': 'Síntoma: Cefalea',
    'dolrretroo': 'Síntoma: Dolor Retroocular'
}

print("\n🔎 Nombres más comprensibles de algunas columnas clave:")
for col, descripcion in columnas_clave.items():
    if col in df.columns:
        print(f"{col} ➝ {descripcion}")

# Agregar columna 'Casos' como contador de eventos por año (si no existe)
if 'Casos' not in df.columns:
    df_casos = df.groupby('año').size().reset_index(name='Casos')
    df = df_casos

# -----------------------------
# 4. PREPARACIÓN DE DATOS
# -----------------------------

# Eliminar registros con valores faltantes (por ejemplo, 2025 sin dato real)
df_limpio = df.dropna()

X = df_limpio[['año']]
y = df_limpio['Casos']

# Ajustar modelo y agregar proyección de 2025
modelo = LinearRegression()
modelo.fit(X, y)
anio_pred = np.array([[2025]])
casos_pred = modelo.predict(anio_pred)
df_pred = pd.DataFrame({'año': [2025], 'Casos': [int(casos_pred[0])]})
df_limpio = pd.concat([df_limpio, df_pred], ignore_index=True)

# -----------------------------
# 3. VISUALIZACIÓN DE DATOS
# -----------------------------

X_full = df_limpio[['año']]
y_pred_full = modelo.predict(X_full)

plt.figure(figsize=(10, 5))
sns.scatterplot(x='año', y='Casos', data=df_limpio, label='Datos reales')
sns.lineplot(x=df_limpio['año'], y=y_pred_full, color='green', label='Línea de tendencia')
plt.scatter(2025, casos_pred, color='red', label='Predicción 2025', s=100)
plt.title('Proyección Realista de Casos de Dengue en el Valle del Cauca (2020-2025)')
plt.xlabel('Año')
plt.ylabel('Número de Casos')
plt.legend()
plt.grid(True)

# -----------------------------
# 8. VISUALIZACIÓN DE VACUNACIÓN
# -----------------------------

plt.figure(figsize=(12, 6))
sns.barplot(
    x='Punto',
    y='Capacidad diaria (personas)',
    data=df_vacunacion,
    color='steelblue'
)
plt.title('Capacidad diaria de vacunación por punto (Dengue)')
plt.xlabel('Punto de Vacunación')
plt.ylabel('Capacidad Diaria')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('resultados/grafica_vacunacion_dengue.png')
plt.show()

# -----------------------------
# 5. APLICACIÓN DE MODELO ML
# -----------------------------

modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para 2025
anio_pred = np.array([[2025]])
casos_pred = modelo.predict(anio_pred)

print(f"\n🔍 Proyección de casos de dengue para el año 2025: {int(casos_pred[0])} casos")

# -----------------------------
# 6. EVALUACIÓN DEL MODELO
# -----------------------------

X_eval = X  # solo años con datos reales
y_eval_pred = modelo.predict(X_eval)
rmse = np.sqrt(mean_squared_error(y, y_eval_pred))
r2 = r2_score(y, y_eval_pred)

print(f"📊 RMSE del modelo: {rmse:.2f}")
print(f"📈 R2 Score (precisión): {r2:.2f}")

# -----------------------------
# Definición de y_pred para todos los datos proyectados (incluyendo 2025)
# -----------------------------

X_full = df_limpio[['año']]
y_pred = modelo.predict(X_full)

# -----------------------------
# 7. VISUALIZACIÓN DE LA PROYECCIÓN
# -----------------------------

plt.figure(figsize=(10, 5))
sns.scatterplot(x='año', y='Casos', data=df_limpio, label='Datos reales')
sns.lineplot(x=df_limpio['año'], y=y_pred, color='green', label='Modelo Regresión')
plt.scatter(2025, casos_pred, color='red', label='Predicción 2025', s=100)
plt.title('Proyección Casos de Dengue (ML)')
plt.xlabel('Año')
plt.ylabel('Casos')
plt.legend()
plt.grid(True)
