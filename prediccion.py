import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar datos preprocesados (como el que generamos antes)
df = pd.read_csv('data/dengue_valle.csv')

df_vac = pd.read_csv('data/puntos_vacunacion_dengue.csv')
df_vac['Municipio'] = df_vac['Municipio'].str.upper().str.strip()
df['nmun_resi'] = df['nmun_resi'].str.upper().str.strip()
df['vacunacion_disponible'] = df['nmun_resi'].isin(df_vac['Municipio']).astype(int)

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
    cefalea=('cefalea', lambda x: (x == 1).mean()),
    vacunacion_disponible=('vacunacion_disponible', 'mean')
).reset_index()

features = ['semana', 'promedio_edad', 'fiebre', 'vomito', 'dolor_abdo', 'cefalea', 'vacunacion_disponible']
target = 'casos_confirmados'

# Agregar valores simulados para 2022 si están ausentes
if agrupado[agrupado['año'] == 2022].empty:
    semanas_2022 = pd.DataFrame({
        'año': [2022] * 52,
        'semana': list(range(1, 53)),
        'nmun_resi': ['CALI'] * 52,
        'casos_confirmados': [5] * 52,  # valor estimado bajo
        'promedio_edad': [25] * 52,
        'fiebre': [0.5] * 52,
        'vomito': [0.1] * 52,
        'dolor_abdo': [0.1] * 52,
        'cefalea': [0.3] * 52,
        'vacunacion_disponible': [1] * 52
    })
    agrupado = pd.concat([agrupado, semanas_2022], ignore_index=True)

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
ejemplo = np.array([[25, 15, 1, 0, 0, 1, 1]])  # Semana 25, edad 15, fiebre, sin vómito,  con cefalea, vacunación disponible
ejemplo_scaled = scaler.transform(ejemplo)
pred = model.predict(ejemplo_scaled)

print(f"Predicción de casos confirmados: {pred[0][0]:.2f}")

# === Predicción desde 2020 hasta 2026 ===
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')  # Para entorno VS Code en Mac M1

años_historicos = [2022, 2023, 2024]
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

# Mostrar tabla de puntos de vacunación organizados
print("\n🏥 Puntos de Vacunación contra el Dengue:\n")
puntos_ordenados = df_vac[['Punto', 'Municipio', 'Edad mínima (años)', 'Edad máxima (años)', 'Requiere infección previa', 'Capacidad diaria (personas)']]
print(puntos_ordenados.to_string(index=False))


# === Gráficos diagnósticos para casos de dengue ===
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf


# Datos reales 2022–2024
serie_real = agrupado[agrupado['año'].isin([2022, 2023, 2024])]
serie_real = serie_real.groupby(['año', 'semana'])['casos_confirmados'].sum().reset_index()
serie_real['fecha'] = pd.to_datetime(serie_real['año'].astype(str) + '-' + serie_real['semana'].astype(int).astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
serie_real = serie_real[['fecha', 'casos_confirmados']].dropna()

# Datos predichos 2025–2026
serie_pred = df_resultado[df_resultado['año'].isin([2025, 2026])]
serie_pred = serie_pred[['año', 'semana', 'prediccion_casos']].copy()
serie_pred['fecha'] = pd.to_datetime(serie_pred['año'].astype(str) + '-' + serie_pred['semana'].astype(int).astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
serie_pred = serie_pred[['fecha', 'prediccion_casos']].dropna()
serie_pred.rename(columns={'prediccion_casos': 'casos_confirmados'}, inplace=True)

# Unir ambas series
serie_dengue = pd.concat([serie_real, serie_pred]).set_index('fecha')
serie_dengue = serie_dengue['casos_confirmados']

serie_dengue_valida = serie_dengue.dropna()
print("\nPrimeras filas de serie_dengue:")
print(serie_dengue.head())
print("\nFiltrado serie_dengue_valida:")
print(serie_dengue_valida.head())
print(f"Semanas disponibles para descomposición: {len(serie_dengue_valida)}")
if len(serie_dengue_valida) >= 52:
    descomp = seasonal_decompose(serie_dengue_valida, model='additive', period=26)

    # Gráfico tipo 1: Descomposición temporal
    plt.figure(figsize=(10, 8))
    plt.suptitle('Casos de Dengue - Descomposición', fontsize=16)
    plt.subplot(411)
    plt.plot(descomp.observed)
    plt.title('Casos Observados')

    from matplotlib.dates import YearLocator, DateFormatter
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2026-12-31'))
    plt.xticks(rotation=0)

    plt.subplot(412)
    plt.plot(descomp.trend)
    plt.title('Tendencia')
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2026-12-31'))

    plt.subplot(413)
    plt.plot(descomp.seasonal)
    plt.title('Estacionalidad')
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2026-12-31'))

    plt.subplot(414)
    plt.plot(descomp.resid, marker='o', linestyle='none')
    plt.axhline(0, color='gray', linewidth=1)
    plt.title('Residuos')
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2026-12-31'))
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Gráfico tipo 2: Diagnóstico de residuos
    residuos = descomp.resid.dropna()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Análisis de Residuos - Casos Dengue', fontsize=14)

    axs[0, 0].plot(residuos)
    axs[0, 0].set_title('Residuos Estandarizados')
    axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    axs[0, 0].tick_params(axis='x', rotation=30)

    sns.histplot(residuos, kde=True, stat='density', ax=axs[0, 1])
    sns.kdeplot(residuos, color='orange', ax=axs[0, 1])
    import numpy as np
    sns.kdeplot(np.random.normal(0, 1, len(residuos)), color='green', linestyle='--', label='N(0,1)', ax=axs[0, 1])
    axs[0, 1].set_title('Histograma + KDE')

    qqplot(residuos, line='s', ax=axs[1, 0])
    axs[1, 0].set_title('Gráfico Q-Q')

    plot_acf(residuos, ax=axs[1, 1])
    axs[1, 1].set_title('Correlograma')

    plt.tight_layout()
    plt.show()
elif not serie_dengue_valida.empty:
    print("⚠️ No hay suficientes datos para descomposición estacional, pero se graficará la evolución de casos.")
    serie_dengue_valida.plot(figsize=(12, 6), title="Evolución semanal de casos de Dengue")
    plt.xlabel("Fecha")
    plt.ylabel("Casos confirmados")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ No hay datos disponibles para graficar la evolución semanal de casos de Dengue.")
plt.show(block=True)