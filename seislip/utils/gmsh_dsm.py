import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# 1. 生成真实形变场
np.random.seed(42)
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
D = 2 * x  # 线性趋势
def gaussian_kernel(x, y, x0, y0, sigma, amplitude):
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
D += gaussian_kernel(x, y, 0.3, 0.7, 0.1, 5)  # 隆起
D -= gaussian_kernel(x, y, 0.7, 0.3, 0.15, 3)  # 沉降
D += 0.5 * np.random.rand(100, 100)  # 噪声

# 2. 四叉树降采样模拟（生成网格大小场）
def quadtree_size_field(data, x_range, y_range, threshold, size_min=0.01, size_max=0.5):
    """递归分割生成网格大小场"""
    size_field = np.ones_like(data) * size_max  # 默认最大网格
    def subdivide(x_min, x_max, y_min, y_max, depth=0, max_depth=5):
        if depth >= max_depth:
            return

        # 计算当前块的索引范围
        i_min, i_max = int(y_min * data.shape[0]), int(y_max * data.shape[0])
        j_min, j_max = int(x_min * data.shape[1]), int(x_max * data.shape[1])
        block = data[i_min:i_max, j_min:j_max]

        # 计算梯度方差作为分割依据
        grad_y, grad_x = np.gradient(block)
        grad_var = np.var(np.sqrt(grad_x**2 + grad_y**2))

        # 如果方差超过阈值，细分
        if grad_var > threshold:
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            size = size_max / (2 ** (depth + 1))  # 网格大小随深度减小
            size_field[i_min:i_max, j_min:j_max] = max(size, size_min)

            # 递归四分
            subdivide(x_min, x_mid, y_min, y_mid, depth + 1)
            subdivide(x_mid, x_max, y_min, y_mid, depth + 1)
            subdivide(x_min, x_mid, y_mid, y_max, depth + 1)
            subdivide(x_mid, x_max, y_mid, y_max, depth + 1)

    subdivide(x_range[0], x_range[1], y_range[0], y_range[1])
    return size_field

# 生成网格大小场
threshold = 0.1  # 梯度方差阈值
mesh_size = quadtree_size_field(D, (0, 1), (0, 1), threshold)

# 3. 将网格大小场写入 .pos 文件
with open("quadtree_size.pos", "w") as f:
    f.write("View \"background\" {\n")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            x_coord, y_coord = j / (D.shape[1] - 1), i / (D.shape[0] - 1)
            f.write(f"SP({x_coord},{y_coord},0){{{mesh_size[i, j]}}};\n")
    f.write("};\n")

# 4. 使用 Gmsh 生成三角形网格
gmsh.initialize()
gmsh.model.add("insar_mesh")
gmsh.model.geo.addPoint(0, 0, 0, 0.5, 1)
gmsh.model.geo.addPoint(1, 0, 0, 0.5, 2)
gmsh.model.geo.addPoint(1, 1, 0, 0.5, 3)
gmsh.model.geo.addPoint(0, 1, 0, 0.5, 4)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()

gmsh.merge("quadtree_size.pos")
gmsh.model.mesh.field.add("PostView", 1)
gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.model.mesh.generate(2)
gmsh.write("insar_mesh.msh")

# 5. 提取网格数据
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
element_types, element_tags, node_connectivity = gmsh.model.mesh.getElements(2)
gmsh.finalize()

nodes = np.array(node_coords).reshape(-1, 3)[:, :2]
triangles = np.array(node_connectivity[0]).reshape(-1, 3) - 1

# 6. 可视化
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 原始形变图
im1 = axs[0].imshow(D, cmap='jet', extent=[0, 1, 0, 1])
axs[0].set_title("Simulated Deformation Map")
plt.colorbar(im1, ax=axs[0])

# 网格大小场（模拟四叉树结果）
im2 = axs[1].imshow(mesh_size, cmap='viridis', extent=[0, 1, 0, 1])
axs[1].set_title("Quadtree-like Mesh Size Field")
plt.colorbar(im2, ax=axs[1])

# 三角形网格
triang = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
axs[2].triplot(triang, 'b-', lw=0.5)
axs[2].set_title("Adaptive Triangular Mesh")
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)

plt.tight_layout()
plt.show()
