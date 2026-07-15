# SAT Morphing FEM

这是一个用于复现 Lloyd 等人论文中 SAT（subcutaneous adipose tissue，皮下脂肪组织）增减思路的研究原型。它不是 Sim4Life 源码，也不是临床软件。核心思想是：

```text
F = F_el F_in
F_in = lambda I
det(F_in) = lambda^3
```

SAT 单元被赋予各向同性的目标增长或收缩，周围软组织采用可压缩 Neo-Hookean 超弹性模型，骨骼相关节点采用零位移约束。程序通过增量 Newton 法求静力平衡，并输出实际 SAT 体积变化。

## 已实现功能

- 多材料线性四面体网格。
- SAT 区域的乘法分解增长/收缩模型。
- 可压缩 Neo-Hookean 超弹性材料。
- 骨骼节点固定的 Dirichlet 边界条件。
- 一致材料切线、稀疏 Newton 求解和回溯线搜索。
- Gmsh/VTU 输入和 NPZ 输入。
- VTU、NPZ、JSON 结果输出。
- 独立高分辨率表面的位移映射。
- 三角面中心坐标映射和质量报告。
- 73 标签 MATLAB 体素 atlas 到粗四面体网格和独立体表面的转换。
- PyMeshFix 表面修复入口。

## 仍未实现

- 组织接触、滑移和自接触。
- 近不可压缩材料的混合位移-压力单元。
- Mooney-Rivlin 和 St. Venant-Kirchhoff 材料模型。
- 骨骼旋转和姿态调整。
- PETSc/MPI 百万单元并行求解。

## 安装

推荐在 Linux 或 WSL Ubuntu 中运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

如果要使用 MeshFix 表面修复：

```bash
python -m pip install -e '.[preprocess]'
```

## 先运行验证算例

增加 SAT 到无约束目标体积的 1.5 倍：

```bash
satmorph demo --target-volume-ratio 1.5 --output examples/demo-expand
```

减少 SAT 到无约束目标体积的 0.6 倍：

```bash
satmorph demo --target-volume-ratio 0.6 --output examples/demo-shrink
```

每次会生成：

- `*.vtu`：可在 ParaView 中查看的粗四面体变形结果。
- `*.npz`：包含原始点、变形点、位移、单元标签等数组。
- `*.json`：包含目标体积、实际 SAT 体积比、Jacobian 和迭代记录。

论文中的 `lambda=0.7` 和 `lambda=1.3` 可以这样跑：

```bash
satmorph demo --lambda-sat 0.7 --output examples/demo-paper-shrink
satmorph demo --lambda-sat 1.3 --output examples/demo-paper-expand --increments 24
```

## 真实人体网格输入要求

输入必须是共享界面节点的 conforming 多材料四面体网格。每个四面体必须有一个组织标签，例如：

- `BONE`：骨骼或固定组织。
- `SOFT`：普通软组织。
- `SAT`：皮下脂肪组织。
- `SKIN`：皮肤。

Gmsh 推荐使用三维 physical groups 命名这些区域。默认读取 `gmsh:physical`：

```bash
satmorph solve \
  --input torso.msh \
  --cell-data gmsh:physical \
  --sat-tag SAT \
  --bone-tag BONE \
  --target-volume-ratio 0.6 \
  --output outputs/torso-sat-060
```

如果文件没有 physical group 名称，也可以直接使用整数标签：

```bash
satmorph solve \
  --input torso.vtu \
  --cell-data region \
  --sat-tag 3 \
  --bone-tag 1 \
  --target-volume-ratio 1.25 \
  --output outputs/torso-sat-125
```

坐标和材料参数必须使用一致单位。推荐坐标用米，Young 模量用 Pa。

## 数据模型条件清单

本工具处理的数据最好分成两类：第一类是用于 FEM 求解的粗四面体体网格，第二类是可选的独立高分辨率表面网格。真正决定 SAT 增减结果的是粗体网格；高分辨率表面只负责接收粗网格位移，用于最终显示或导出。

### 必须输入：粗四面体体网格

粗体网格需要满足：

| 条件 | 要求 |
|---|---|
| 维度 | 三维人体或局部人体模型 |
| 网格类型 | 四面体体网格，不是单独的 STL/OBJ 表面 |
| 单元类型 | 线性四面体 `tetra`；`tetra10` 只会取前 4 个节点 |
| 组织标签 | 每个四面体都要有一个 tissue/material/region 标签 |
| 几何关系 | 不同组织之间最好共享边界节点，也就是 conforming mesh |
| 坐标单位 | 推荐米；如果用毫米，材料参数和容差也要保持一致 |
| 网格质量 | 四面体不能退化、翻转或体积接近 0 |
| 覆盖范围 | 粗体网格要覆盖需要变形的 SAT 和外部表面区域 |

最少需要这些组织区域：

```text
SAT   皮下脂肪组织，需要被增加或减少
BONE  骨骼或固定区域，用来限制整体刚体漂移
SOFT  其他软组织，可选但推荐
SKIN  皮肤，可选但推荐
```

也就是说，每个四面体单元要知道自己属于哪类组织。可以用整数标签：

```text
1 = BONE
2 = SOFT
3 = SAT
4 = SKIN
```

也可以用 Gmsh physical group 名称：

```text
BONE
SOFT
SAT
SKIN
```

推荐格式：

```text
.msh   Gmsh 四面体网格
.vtu   VTK/ParaView 四面体网格
.npz   本项目的 NumPy 网格格式
```

不适合作为核心 FEM 输入：

```text
.stl   通常只有表面，不能直接做 SAT 体积变形
.obj   通常只有表面，不能直接做 FEM
.ply   多数情况下也是表面
```

这些表面格式可以用于后面的高分辨率表面映射，但不能直接作为 `satmorph solve` 的粗体网格输入。

### 可选输入：高分辨率表面网格

如果有精细皮肤表面、SAT 外表面或器官表面，可以作为第二类输入，用来接收粗网格位移。它可以是：

```text
.stl
.obj
.ply
.vtp
.npz
```

高分辨率表面需要满足：

| 条件 | 要求 |
|---|---|
| 网格类型 | 三角面表面网格 |
| 坐标系 | 必须和粗四面体体网格在同一个坐标系 |
| 单位 | 必须和粗体网格一致 |
| 位置关系 | 最好落在粗体网格内部或边界附近 |
| 拓扑 | 不需要和粗网格共享节点 |
| 分辨率 | 可以比粗网格高很多 |

真实人体模型建议满足：

```text
1. SAT 是一个独立可识别的体区域。
2. BONE 或其他固定区域必须存在，不能完全没有约束。
3. SAT 与皮肤、肌肉、其他软组织之间不能是完全分离的散乱表面。
4. 四面体不能大量瘦长、翻转或接近零体积。
5. 高分辨率表面和粗体网格必须配准在同一个人体姿态和坐标系里。
6. 粗体网格要包住高分辨率表面，否则映射报告会出现很多 outside_points。
```

一句话标准：

```text
一个带组织标签的 conforming 多材料四面体人体网格，
其中至少包含 SAT 和 BONE；
如果有独立高分辨率皮肤、SAT 或器官表面，
它们必须和粗体网格在同一坐标系下。
```

如果手头只有 STL/OBJ 表面模型，下一步不是直接跑 SAT，而是先做：表面修复、多组织封闭表面、四面体体网格生成、组织标签赋值。

## 材料参数

默认材料参数只是演示值，不是 ViP 的 73 种组织材料表。可以用 JSON 覆盖：

```json
{
  "default": {"young": 10000.0, "poisson": 0.45},
  "SAT": {"young": 5000.0, "poisson": 0.45},
  "SKIN": {"young": 50000.0, "poisson": 0.45}
}
```

运行：

```bash
satmorph solve \
  --input torso.msh \
  --sat-tag SAT \
  --bone-tag BONE \
  --target-volume-ratio 0.6 \
  --materials materials.json \
  --output outputs/result
```

## 粗网格位移到高分辨率表面的映射

真实人体模型通常会有两套网格：

- 粗四面体体网格：用于 FEM 求解。
- 独立高分辨率表面：用于最终几何、可视化或导出。

先对粗体网格求解：

```bash
satmorph solve \
  --input torso.msh \
  --cell-data gmsh:physical \
  --sat-tag SAT \
  --bone-tag BONE \
  --target-volume-ratio 0.6 \
  --output outputs/torso-sat-060
```

再把粗网格位移映射到独立表面：

```bash
satmorph map-surface \
  --coarse-result outputs/torso-sat-060.npz \
  --surface skin-highres.stl \
  --output outputs/skin-highres-deformed.vtp \
  --centers-output outputs/skin-highres-centers.npz \
  --report outputs/skin-highres-map.json
```

映射方式：

- 对每个高分辨率表面顶点，寻找附近粗四面体。
- 计算该点在粗四面体中的重心坐标。
- 用粗四面体四个节点的位移进行插值。
- 对每个三角面中心也做同样的映射，输出到 `--centers-output`。

如果表面点略微落在粗体网格外，默认 `--outside-mode clamp` 会把重心坐标夹到合理范围，避免远距离线性外推导致变形炸掉。可选模式：

```bash
--outside-mode clamp
--outside-mode linear
--outside-mode fail
```

质量报告里重点看：

- `outside_points`：落在粗四面体网格外的表面点数量。
- `outside_triangle_centers`：落在粗四面体网格外的三角面中心数量。
- `maximum_point_residual`：表面点偏离粗体网格内部的程度。
- `maximum_center_residual`：三角面中心偏离粗体网格内部的程度。

如果 outside 数量很多，说明高分辨率表面和粗体网格没有对齐，应该先做配准或扩大粗体网格覆盖范围。

## NPZ 格式

### 73 标签体素 MAT 转换

如果 MAT 保存的是 `MaterialLabelGrid` 体素标签，而不是现成的节点和四面体，使用专门的体素转换命令。下面的标签规则适用于当前 73 标签 atlas：

```text
SAT   = 1
SKIN  = 2
BONE  = 19, 68, 69, 70
SOFT  = 其他所有非零标签
0     = 人体外背景，不生成四面体
```

先用较粗的网格验证完整流程：

```bash
satmorph convert-voxel-mat \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-coarse.npz \
  --surface-output outputs/human-surface.npz \
  --report outputs/human-convert.json \
  --stride 20 \
  --surface-stride 4
```

`--stride 20` 表示粗 FEM 网格每个体素块约为原始网格的 `20×20×20` 个体素；`--surface-stride 4` 会生成分辨率更高的独立体表面。粗网格采用共享节点的六四面体立方块划分，组织交界面保持 conforming。为避免把外轮廓边缘的一层皮肤放大成 20 mm 厚层，低占比边界块会归入 `SOFT`；可通过 `--envelope-fraction` 和 `--skin-fraction` 调整阈值。默认最多生成 50 万个四面体，超过时程序会要求增大 stride 或明确提高 `--max-tetrahedra`。

转换后可直接求解并映射表面：

```bash
satmorph solve \
  --input outputs/human-coarse.npz \
  --sat-tag SAT \
  --bone-tag BONE \
  --target-volume-ratio 1.2 \
  --output outputs/human-sat-120

satmorph map-surface \
  --coarse-result outputs/human-sat-120.npz \
  --surface outputs/human-surface.npz \
  --output outputs/human-surface-deformed.npz \
  --report outputs/human-surface-map.json
```

该转换是规则体素块四面体化，适合验证 SAT 增减和粗到细位移映射。它不会自动平滑台阶状体表，也不等同于经过 TetWild、CGAL 或 Gmsh 优化的临床级网格。正式计算前仍应检查标签、表面和四面体质量。

### 平滑可视化表面

`.vtu` 是粗四面体 FEM 结果，主要用于检查 `region`、`growth_lambda`、`J_total` 和 `J_elastic`。如果目标是得到更好看的外表面，应额外从体素 atlas 提取三角表面，再把粗 FEM 位移映射到这个表面。

安装 marching cubes 可选依赖：

```bash
python -m pip install -e '.[visual]'
```

从 `MaterialLabelGrid` 提取平滑体表：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-visual-surface.vtk \
  --report outputs/human-visual-surface.json \
  --surface-stride 2 \
  --method marching-cubes \
  --pre-smooth-sigma 0.5 \
  --smooth-method taubin \
  --smooth-iterations 20
```

如果没有安装 `scikit-image`，可以先用块状回退版本：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-visual-surface.npz \
  --report outputs/human-visual-surface.json \
  --surface-stride 2 \
  --method blocks \
  --smooth-method taubin \
  --smooth-iterations 20
```

然后把 SAT 求解结果映射到这个平滑表面：

```bash
satmorph map-surface \
  --coarse-result outputs/human-sat-120.npz \
  --surface outputs/human-visual-surface.vtk \
  --output outputs/human-sat-120-visual.vtk \
  --centers-output outputs/human-sat-120-visual-centers.npz \
  --report outputs/human-sat-120-visual-map.json
```

也可以导出给 Blender 或其他软件：

```bash
satmorph map-surface \
  --coarse-result outputs/human-sat-120.npz \
  --surface outputs/human-visual-surface.vtk \
  --output outputs/human-sat-120-visual.ply \
  --report outputs/human-sat-120-visual-map.json
```

建议用于展示的文件是 `human-sat-120-visual.vtk`、`.ply` 或 `.obj`，而不是粗网格 `.vtu`。映射后的表面坐标已经变形，不需要在 ParaView 里再次使用 `Warp By Vector`。

如果原始数据是 MATLAB `.mat` 文件，可以先查看其中的变量：

```bash
satmorph convert-mat --input model.mat --list-variables
```

常见变量名（如 `nodes`、`elements`、`materials`、`skin_vertices`、`skin_faces`）会自动识别：

```bash
satmorph convert-mat \
  --input model.mat \
  --output model-coarse.npz \
  --surface-output model-surface.npz \
  --report model-convert.json
```

若变量名比较特殊，可通过 `--points-key`、`--tetra-key`、`--tags-key`、`--surface-points-key` 和 `--surface-faces-key` 指定。MATLAB 常见的从 1 开始的节点编号会自动转换；也可用 `--one-based` 或 `--zero-based` 明确指定。

MATLAB v7.3 文件需要额外安装 HDF5 支持：

```bash
python -m pip install -e '.[mat]'
```

最小粗四面体输入：

```python
numpy.savez(
    "model.npz",
    points=points,          # (N, 3)
    tetra=tetra,            # (M, 4)
    cell_tags=cell_tags,    # (M,)
)
```

独立表面 NPZ 输入：

```python
numpy.savez(
    "surface.npz",
    points=surface_points,      # (N, 3)
    triangles=triangles,        # (M, 3)
)
```

## 表面预处理

MeshFix 可用于修复自相交、孔洞和非流形三角面：

```bash
satmorph repair-surface --input sat-raw.stl --output sat-repaired.stl
```

修复表面后，仍需要使用 Gmsh、TetGen 或其他工具生成 conforming 多区域四面体体网格。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试包括：

- Neo-Hookean 一致切线的有限差分检查。
- `lambda=1` 时零残差检查。
- SAT 增大和骨骼固定的端到端检查。
- 粗四面体位移到独立表面点的重心坐标映射检查。
- 三角面中心坐标映射检查。
