import csv
import os

# en fcsv_files estan puntos de referencia extraidos de femur.stl
# tomo 2 del tuberculo mayor y uno del tuberculo menor

# Carpeta donde están los .fcsv
carpeta = "fcsv_files"  # cambiala si hace falta
archivos = [f for f in os.listdir(carpeta) if f.endswith(".fcsv")]

# Archivo de salida
salida = "Table.csv"

# Campos que querés conservar
columnas_salida = ["label", "x", "y", "z"]

# Lista para guardar los puntos
todos_los_puntos = []

for archivo in archivos:
    with open(os.path.join(carpeta, archivo), newline='') as f:
        reader = csv.reader(f)
        for fila in reader:
            if not fila or fila[0].startswith("#"):
                continue  # salteá líneas de encabezado
            # Extraer columnas: id,x,y,z,...,label,...
            punto = {
                "label": fila[11],
                "x": float(fila[1]),
                "y": float(fila[2]),
                "z": float(fila[3])
            }
            todos_los_puntos.append(punto)

# Guardar en Table.csv
with open(salida, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columnas_salida)
    writer.writeheader()
    writer.writerows(todos_los_puntos)

print(f"Archivo combinado guardado como: {salida}")
