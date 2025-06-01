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

# -----------------------------
# Limpieza del archivo Excel adicional
# -----------------------------
try:
    df_excel = pd.read_excel("data/dengue2_valle.xlsx")

    print("\n🧾 Columnas disponibles en 'dengue2_valle.xlsx':")
    print(df_excel.columns.tolist())

    # Filtrar por Valle del Cauca
    df_excel = df_excel[df_excel['COD_DPTO_R'] == 76]

    # Filtrar por dengue clásico o grave/hemorrágico
    df_excel = df_excel[df_excel['COD_EVE'].isin([210, 220])]

    # Guardar como nuevo archivo limpio en Excel
    df_excel.to_excel("data/dengue2_valle_limpio.xlsx", index=False)
    print("✅ Excel limpiado y guardado como 'dengue2_valle_limpio.xlsx'")

    # Guardar también como CSV
    df_excel.to_csv("data/dengue2_valle_limpio.csv", index=False)
    print("✅ También guardado como 'dengue2_valle_limpio.csv'")
except Exception as e:
    print(f"❌ No se pudo limpiar 'dengue2_valle.xlsx': {e}")