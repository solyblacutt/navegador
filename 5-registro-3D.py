import open3d as o3d
import numpy as np

# --- Crear modelo STL sintético (cubo) ---
mesh_modelo = o3d.io.read_triangle_mesh("Segmentation_Segment_3_femur.stl")
mesh_modelo.compute_vertex_normals()
mesh_modelo.paint_uniform_color([0.8, 0.8, 0.8])

# Cargar puntos del modelo STL (por ejemplo, desde un .csv exportado de Slicer)
puntos_modelo = np.loadtxt("Table.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3))

# Cargar puntos del sistema de cámara (calculados desde solvePnP)
puntos_camara = np.loadtxt("puntos_3d_en_camara.csv", delimiter=",", skiprows=1)

# Crear nubes de puntos
pcd_modelo = o3d.geometry.PointCloud()
pcd_modelo.points = o3d.utility.Vector3dVector(puntos_modelo)
pcd_modelo.paint_uniform_color([0, 0, 1])

pcd_camara = o3d.geometry.PointCloud()
pcd_camara.points = o3d.utility.Vector3dVector(puntos_camara)
pcd_camara.paint_uniform_color([1, 0, 0])

# Estimar transformación (registro)
threshold = 10.0  # mm
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_camara, pcd_modelo, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Transformar los puntos de cámara
pcd_camara_alineado = pcd_camara.transform(reg_p2p.transformation)

# Mostrar
ejes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
o3d.visualization.draw_geometries([
    mesh_modelo,
    pcd_modelo,
    pcd_camara,
    pcd_camara_alineado,
    ejes
])
