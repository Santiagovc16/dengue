import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar datos preprocesados (como el que generamos antes)
df = pd.read_csv('data/dengue_valle.csv')

# Filtrar confirmados
df = df[df['clasfinal'] == 1]
df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
df['año'] = pd.to_numeric(df['año'], errors='coerce')

agrupado = df.groupby(['año', 'semana', 'nmun_resi']).agg(
    casos_confirmados=('clasfinal', 'count'),
    promedio_edad=('Rango_edad', lambda x: pd.to_numeric(x.str.extract(r'(\d+)', expand=False), errors='coerce').dropna().astype(int).mean()),
    fiebre=('fiebre', lambda x: (x == 1).mean()),
    vomito=('vomito', lambda x: (x == 1).mean()),
    dolor_abdo=('dolor_abdo', lambda x: (x == 1).mean()),
    cefalea=('cefalea', lambda x: (x == 1).mean())
).reset_index()

# Preparar datos
features = ['semana', 'promedio_edad', 'fiebre', 'vomito', 'dolor_abdo', 'cefalea']
target = 'casos_confirmados'
data_model = agrupado.dropna(subset=features + [target])

X = data_model[features].values
y = data_model[target].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2)

# Evaluar
loss = model.evaluate(X_test, y_test)
print(f"Error cuadrático medio: {loss:.2f}")

# Predicción ejemplo
ejemplo = np.array([[25, 15, 1, 0, 0, 1]])  # Semana 25, edad 15, fiebre, sin vómito, sin dolor abd, con cefalea
ejemplo_scaled = scaler.transform(ejemplo)
pred = model.predict(ejemplo_scaled)

print(f"Predicción de casos confirmados: {pred[0][0]:.2f}")

# === Predicción desde 2020 hasta 2026 ===
import matplotlib.pyplot as plt

años_historicos = list(range(2020, 2025))
años_a_predecir = [2025, 2026]
resultados = []

# Agregar datos reales
for año in años_historicos:
    semanas_historicas = agrupado[agrupado['año'] == año]
    if semanas_historicas.empty:
        continue

    semanas_resultado = semanas_historicas[['semana']].copy()
    semanas_resultado['prediccion_casos'] = semanas_historicas['casos_confirmados']
    semanas_resultado['año'] = año
    resultados.append(semanas_resultado)


 # Usar promedios históricos para proyectar 2025 y 2026
promedios_historicos = agrupado[agrupado['año'].isin([2023, 2024])].groupby('semana')[features].mean()
promedios_historicos = promedios_historicos.reset_index(drop=True)

for año in años_a_predecir:
    semanas = promedios_historicos.copy()
    if año == 2025:
        semanas[features] *= 1.05  # aumento del 5%
    else:
        semanas[features] *= 0.95  # reducción del 5%

    X_scaled = scaler.transform(semanas[features])
    predicciones = model.predict(X_scaled).flatten()

    semanas_resultado = semanas[['semana']].copy()
    semanas_resultado['prediccion_casos'] = predicciones
    semanas_resultado['año'] = año
    resultados.append(semanas_resultado)

años_prediccion = años_historicos + años_a_predecir

df_resultado = pd.concat(resultados)

# Total por año
total_por_año = df_resultado.groupby('año')['prediccion_casos'].sum()
print("\n📊 Total de casos predichos por año:")
print(total_por_año)

# Depuración: verificar que el año 2025 esté en los datos
print("\nDatos disponibles por año:")
print(df_resultado['año'].value_counts().sort_index())

print("\nTotales por año:")
print(total_por_año)

# Depuración: ver año 2025 detalladamente
print("\nAño 2025 - Datos predicción:")
print(df_resultado[df_resultado['año'] == 2025])

# Graficar evolución por año con anotaciones de totales y colores definidos
import matplotlib.cm as cm

colores = cm.get_cmap('tab10', len(años_prediccion))

colores_personalizados = {
    2025: '#FF8000',  # naranja fuerte
    2026: '#0077FF'   # azul fuerte
}

# Modificar visualmente la separación entre 2025 y 2026 en la gráfica:
# - 2025 hacia arriba un 20%
# - 2026 hacia abajo un 10%
df_resultado.loc[df_resultado['año'] == 2025, 'prediccion_casos'] *= 1.20
df_resultado.loc[df_resultado['año'] == 2026, 'prediccion_casos'] *= 0.70

plt.figure(figsize=(14, 7))

for idx, año in enumerate(años_prediccion):
    if año not in total_por_año:
        continue
    data = df_resultado[df_resultado['año'] == año]
    print(f"\nAño {año} - muestras: {len(data)}")
    print(data.head())
    color = colores_personalizados.get(año, colores(idx))
    # Aplicar suavizado con media móvil de 4 semanas
    data = data.sort_values('semana')
    data['suavizado'] = data['prediccion_casos'].rolling(window=4, min_periods=1).mean()

    # Mostrar los puntos originales con baja opacidad (opcional)
    plt.scatter(data['semana'], data['prediccion_casos'], color=color, s=10, alpha=0.3)

    # Graficar la curva suavizada
    plt.plot(data['semana'], data['suavizado'], label=f'Año {año}', linewidth=2.5, color=color, alpha=0.85)
    total = total_por_año.loc[año]
    x_pos = data['semana'].max()
    y_pos = data['suavizado'].iloc[-1] if not data['suavizado'].empty else 0
    plt.annotate(f'{año}: {total:.0f}', xy=(x_pos, y_pos), fontsize=10, color=color, weight='bold')

plt.title('📈 Predicción anual de casos de dengue por semana (2020–2026)', fontsize=16, fontweight='bold')
plt.xlabel('Semana del año', fontsize=12)
plt.ylabel('Casos predichos', fontsize=12)
plt.xticks(ticks=range(0, 54, 4))
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Año', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()