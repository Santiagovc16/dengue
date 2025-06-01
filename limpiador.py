import pandas as pd

# Cargar el dataset
DATA_PATH = 'data/dengue_valle.csv'
df = pd.read_csv(DATA_PATH)

# Filtrar por departamento del Valle del Cauca
df = df[df['cod_dpto_r'] == 76]  # 76 es el código DANE para Valle del Cauca

# Filtrar solo casos de dengue clásico o hemorrágico
df = df[df['cod_eve'].isin([210, 220])]  # 210: Dengue clásico, 220: Dengue grave/hemorrágico

# Guardar el nuevo dataset filtrado
df.to_csv("dataset_dengue_valle.csv", index=False)

print("✅ Datos filtrados por Valle del Cauca y casos de dengue guardados en 'dataset_dengue_valle.csv'")