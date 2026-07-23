# SAT Morphing FEM

这是一个用于复现论文中“通过体积增长/收缩改变皮下脂肪组织 SAT”的研究原型。它不是 Sim4Life 源码，也不是完整商业仿真平台；目前更准确的定位是：

```text
带 73 组织标签的人体体素模型
  -> 粗四面体 FEM 网格
  -> SAT 体积增减求解
  -> 高分辨率表面位移映射
  -> ParaView / Blender 可视化
```

核心力学思想是乘法分解：

```text
F = F_el F_in
F_in = lambda I
det(F_in) = lambda^3
```

其中 SAT 单元被施加各向同性的目标增长或收缩，周围组织通过有限元静力平衡被动变形。求解器当前使用可压缩 Neo-Hookean 超弹性材料、Newton 迭代和线搜索。

## 当前离“近似 Sim4Life 器官变形”还差多少

如果目标是“做一个能真实替代 Sim4Life 的人体多物理场器官变形平台”，现在还差很远；如果目标是“针对 SAT 增减做一个可解释、可运行、可视化的人体形变原型”，现在已经有了主干流程。

粗略估计：

| 目标 | 当前完成度 | 说明 |
|---|---:|---|
| SAT 增减论文功能复现 | 82%-87% | 已有外层校正、动态接触、体积审计、形态指标、粗细表面映射 |
| 73 标签人体模型接入 | 80%-85% | 支持累计审计驱动三级加密、多组织表面和纤维方向场 |
| 好看的外表面可视化 | 85%-90% | 支持独立 VTP、VTM、透明多组织三视图和自动解剖方向校正 |
| 通用器官变形 | 55%-65% | 已有按标签变形和动态有限滑移近似接触，未实现器官姿态控制 |
| 接近 Sim4Life 工作流 | 55%-65% | 已有验证、批处理和多组织输出，仍缺 GUI、模型管理和多物理场 |
| 工程级/发表级可靠性 | 50%-60% | 已有真实 73 标签验证，材料参数仍需文献标定和实验对照 |

一句话：现在已经不是“玩具脚本”，而是一个 SAT 变形研究原型；但离真正 Sim4Life 式通用器官变形，还差接触、材料、网格、边界条件、验证和大规模求解这几座山。

## 下一步优化路线

### 第 1 阶段：把现有 SAT 流程跑稳

目标：每次输入同一份 73 标签体素模型，都能稳定得到可检查的 `.npz/.vtu/.vtp/.json`。

需要做：

- 固定推荐命令参数，例如 `stride`、`surface_stride`、`max_tetrahedra`。
- 检查转换报告中的组织体积是否合理。
- 在 ParaView 中确认 `source_label`、`mechanical_group_id`、`growth_lambda` 能正常着色。
- 对 `target-volume-ratio = 0.8, 1.0, 1.2` 做三组对比。
- 重点看 `J_total` 是否为正，`outside_points` 是否过多。

### 第 2 阶段：提高人体表面质量

目标：不要只看到白色方块人，而是得到可展示的人体表面。

需要做：

- 用 marching cubes 提取表面。
- 用 Taubin smoothing 平滑。
- 重新计算法向量。
- 把粗 FEM 位移映射到平滑表面。
- 导出 `.vtp` 给 ParaView，或 `.ply/.obj` 给 Blender。
- 分别提取 `SAT`、`Skin`、`Fat`、`Muscle`、`Bone` 等组织表面。

### 第 3 阶段：从 SAT 变形扩展到器官变形

目标：让某个指定器官也能被放大、缩小、移动或姿态调整。

需要做：

- 已把 `sat_cells` 扩展为通用 `target_cells`。
- 已支持按 `source_label` 选择目标组织，例如 `--target-label 1` 表示 SAT，`--target-label 35` 表示 Liver。
- 支持不同类型的目标变形：
  - 已支持体积增减：`lambda I`
  - 指定位移：器官表面/中心点移动
  - 局部缩放：按器官中心缩放
  - 形状模板匹配：向目标表面配准
- 已输出 `volume_by_region` 和 `volume_by_source_label` 体积变化报告。

### 第 4 阶段：改善生物材料真实性

目标：不要把所有组织都当成同一种软材料。

已完成：

- 保留 73 种原始组织标签 `source_label`。
- 增加机械分组 `mechanical_group_id`。
- 默认按机械组给不同演示级 Young 模量和泊松比。
- 将 SAT、普通脂肪、黄骨髓分开。
- 实现参考方向上的纤维增强超弹性项。
- 自动生成皮肤切向和肌肉/肌腱组织主轴方向场。
- 将 `fiber_direction` 保存为三分量单元数据。

还需要做：

- 从文献整理各组织 Young 模量、泊松比、密度。
- 把材料表从“电磁/密度属性”扩展成“力学属性表”。
- 用真实 DTI、解剖纤维图谱或实验数据替换当前规则/PCA 估计方向。
- 标定纤维刚度；当前数值是演示级参数，不能直接作为临床结论。

### 第 5 阶段：加入接触和边界条件

目标：器官之间不要互相穿透，骨骼和皮肤约束更合理。

已完成：

- 可从两组 `source_label` 的表面邻近关系建立节点-三角面接触候选。
- 求解器中加入无穿透罚函数的能量、残差和一致切线刚度。
- JSON 输出接触数量、激活数量、最小间隙和最大穿透量。
- 动态模式在每次装配时更新最近主面、投影点和当前法向。

还需要做：

- 对骨骼从“固定节点”升级到“刚体骨架约束”。
- 动态接触当前使用冻结法向的准 Newton 切线；后续可实现严格几何一致切线和摩擦。
- 支持局部固定、对称面固定、表面牵引等边界条件。

### 第 6 阶段：大规模求解和验证

目标：让真实人体模型不只是能跑，而是可信。

需要做：

- 引入更好的体网格生成工具，例如 Gmsh、TetGen、fTetWild、CGAL 流程。
- 已增加平均比质量、边长比、负体积和变形前后质量报告。
- 已增加批处理脚本和目标 SAT 体积外层校正。
- 已增加原始体素、转换网格、变形结果之间的逐组织体积对比。
- 已增加腰围曲线、面积、封闭体积、Hausdorff/Chamfer 距离和 SAT 厚度统计。
- 线搜索残差已向量化，单元切线由批量 `einsum` 生成，外层校正复用上轮位移作为初值。
- 对线性四面体增加 `kappa/mu` 上限作为锁死缓解措施。
- 仍需与论文结果或人工测量做误差对比。
- 当前锁死控制不是真正的混合 `u-p` 单元；高精度近不可压缩分析仍应迁移到 FEniCSx/PETSc。
- 对百万级单元考虑 PETSc/FEniCSx/MPI，而不是纯 Python 稀疏装配。

## 安装

推荐在 WSL Ubuntu 或 Linux 环境运行。

```bash
cd /mnt/c/Users/lenovo/Documents/Codex/2026-07-13/covering-population-variability-morphing-of-computation/outputs/sat-morphing-fem/sat-morphing-fem

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[visual,preprocess,mat]'
```

如果只想运行核心 FEM，不需要可视化增强：

```bash
python -m pip install -e .
```

Windows PowerShell 也可以运行，但真实大模型更推荐 WSL。

## 快速验证

先跑一个小型合成模型，确认安装没问题。

```bash
satmorph demo \
  --target-volume-ratio 1.2 \
  --output outputs/demo-sat-120 \
  --quiet
```

会生成：

```text
outputs/demo-sat-120.npz
outputs/demo-sat-120.vtu
outputs/demo-sat-120.json
```

打开 `.vtu` 可以在 ParaView 中查看形变结果。

运行测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell 中使用：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## 输入数据要求

核心 FEM 输入必须是四面体体网格，或可转换为四面体体网格的 73 标签体素模型。

### 推荐输入：73 标签体素 MAT

当前支持的主要输入是 MATLAB `.mat`，其中包含：

```text
MaterialLabelGrid    三维整数标签体素
Axis0                x 轴坐标或边界
Axis1                y 轴坐标或边界
Axis2                z 轴坐标或边界
```

标签约定：

```text
0  = 人体外背景，不生成四面体
1  = SAT (Subcutaneous Fat)
2  = Skin
3  = Fat
19 = Tooth
52 = Muscle
68 = Bone cancellous
69 = Bone cortical
70 = Bone marrow yellow
```

完整 73 标签说明见：

```text
docs/tissue-labels-and-materials.md
```

### 已转换的 FEM 输入

也可以直接输入：

```text
.npz   本项目格式
.vtu   VTK 四面体网格
.msh   Gmsh 四面体网格
```

最小 `.npz` 格式：

```python
numpy.savez_compressed(
    "model.npz",
    points=points,          # (N, 3)
    tetra=tetra,            # (M, 4)
    cell_tags=cell_tags,    # (M,)
)
```

如果包含下面这些字段，后处理会更好：

```text
source_label
material_id
mechanical_group_id
```

## 完整运行流程

下面以 73 标签人体体素模型为例。

### 1. 转换体素模型为粗 FEM 网格

先用较粗参数验证流程：

```bash
satmorph convert-voxel-mat \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-coarse.npz \
  --surface-output outputs/human-block-surface.npz \
  --report outputs/human-convert.json \
  --stride 20 \
  --surface-stride 4
```

参数含义：

```text
--stride 20          每 20 x 20 x 20 个原始体素合成一个粗 FEM 块
--surface-stride 4   生成更高分辨率的块状体表面
--max-tetrahedra     限制四面体数量，避免模型过大直接爆内存
```

默认求解角色：

```text
SAT  = label 1
SKIN = label 2
BONE = label 19, 68, 69
SOFT = 其他非零标签
```

注意：label 70 黄骨髓现在不再默认作为固定骨，而是作为脂肪样软组织保留。

转换后的 `.npz` 会保留：

```text
cell_tags                 SAT/BONE/SOFT/SKIN 求解角色
cell_data__source_label   原始 1-73 标签
cell_data__material_id
cell_data__mechanical_group_id
```

### 2. 求解 SAT 增减

增加 SAT 到目标体积比例 1.2：

```bash
satmorph solve \
  --input outputs/human-coarse.npz \
  --target-label 1 \
  --bone-tag BONE \
  --target-volume-ratio 1.2 \
  --output outputs/human-sat-120
```

减少 SAT 到目标体积比例 0.8：

```bash
satmorph solve \
  --input outputs/human-coarse.npz \
  --target-label 1 \
  --bone-tag BONE \
  --target-volume-ratio 0.8 \
  --output outputs/human-sat-080
```

输出：

```text
outputs/human-sat-120.npz     原始点、变形点、位移、单元数据
outputs/human-sat-120.vtu     ParaView 可视化粗 FEM 结果
outputs/human-sat-120.json    体积比例、Jacobian、迭代记录
```

也可以继续使用旧接口：

```bash
satmorph solve \
  --input outputs/human-coarse.npz \
  --sat-tag SAT \
  --bone-tag BONE \
  --target-volume-ratio 1.2 \
  --output outputs/human-sat-120
```

但论文复现更推荐 `--target-label 1`，因为它直接对应 73 标签模型里的原始 SAT 标签。

### 3. 提取平滑可视化表面

为了避免只看到方块人体，建议用 marching cubes 提取平滑外表面。

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-visual-surface.vtp \
  --report outputs/human-visual-surface.json \
  --surface-stride 2 \
  --method marching-cubes \
  --pre-smooth-sigma 0.5 \
  --smooth-method taubin \
  --smooth-iterations 20
```

如果没有安装 `scikit-image`，可以用块状 fallback：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-visual-surface.vtp \
  --report outputs/human-visual-surface.json \
  --surface-stride 2 \
  --method blocks \
  --smooth-method taubin \
  --smooth-iterations 20
```

### 4. 把粗 FEM 位移映射到平滑表面

```bash
satmorph map-surface \
  --coarse-result outputs/human-sat-120.npz \
  --surface outputs/human-visual-surface.vtp \
  --output outputs/human-sat-120-visual.vtp \
  --centers-output outputs/human-sat-120-visual-centers.npz \
  --report outputs/human-sat-120-visual-map.json
```

映射后，`human-sat-120-visual.vtp` 已经是变形后的表面，不需要再在 ParaView 里 `Warp By Vector`。

### 5. 单独提取某个组织表面

只看 SAT：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --include-label 1 \
  --output outputs/sat-only.vtp \
  --report outputs/sat-only.json \
  --surface-stride 2 \
  --method marching-cubes \
  --smooth-method taubin \
  --smooth-iterations 20
```

只看皮肤：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --include-label 2 \
  --output outputs/skin-only.vtp \
  --report outputs/skin-only.json \
  --surface-stride 2 \
  --method marching-cubes \
  --smooth-method taubin \
  --smooth-iterations 20
```

只看普通脂肪：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --include-label 3 \
  --output outputs/fat-only.vtp \
  --report outputs/fat-only.json \
  --surface-stride 2 \
  --method marching-cubes \
  --smooth-method taubin \
  --smooth-iterations 20
```

常用标签：

```text
1  SAT
2  Skin
3  Fat
35 Liver
52 Muscle
68 Bone cancellous
69 Bone cortical
70 Bone marrow yellow
```

## ParaView 中看什么

打开 `.vtu` 或 `.vtp` 后，在 `Coloring` 里选择：

```text
region                      看 SAT/BONE/SOFT/SKIN 求解角色
source_label                看原始 1-73 组织
mechanical_group_id         看机械分组
material_id                 看材料编号
growth_lambda               看哪些单元被主动施加 SAT 增长
J_total                     看总体体积变化
J_elastic                   看弹性体积变化
displacement_magnitude      看位移大小
mapped_source_label         映射到表面后的原始组织标签
mapped_mechanical_group_id  映射到表面后的机械组
```

如果选择 `region` 后画面消失，通常是 ParaView 的显示范围或颜色映射状态问题，可以尝试：

```text
1. 点击 Reset Camera
2. Coloring 改回 Solid Color
3. 再重新选择 region
4. 点 Rescale to Data Range
5. 确认左侧对象前面的小眼睛是打开状态
```

## 文件含义

### `.npz`

项目内部数据包，适合继续计算。

常见字段：

```text
points              原始粗网格节点坐标
tetra               四面体连接关系
cell_tags           SAT/BONE/SOFT/SKIN 求解标签
deformed_points     变形后的节点坐标
displacement        每个节点的位移向量
growth_lambda       每个四面体的主动增长因子
j_total             总体积 Jacobian
j_elastic           弹性部分 Jacobian
source_label        原始 73 组织标签
mechanical_group_id 机械分组
material_id         材料编号
```

### `.vtu`

粗四面体 FEM 结果，用 ParaView 检查内部体单元、Jacobian、增长区域和材料分组。

### `.vtp`

三角表面网格，适合展示、截图、导入 Blender 或继续做表面处理。

### `.json`

质量报告和运行记录。重点看：

```text
actual_sat_volume_ratio
minimum_total_jacobian
maximum_displacement
outside_points
outside_triangle_centers
maximum_point_residual
```

## 材料参数

当前材料表分两类：

### 1. Sim4Life/IT'IS 风格材料表

你提供的 `s4l_materials_unified.json` 主要包含：

```text
mass_density_kg_per_m3
conductivity_s_per_m
relative_permittivity
frequency_hz
```

这类数据对电磁仿真和物性追踪有用，但不能直接作为 FEM 超弹性参数。

### 2. FEM 力学材料参数

求解器需要：

```text
young      Young 模量，单位 Pa
poisson    泊松比
```

如果网格有 `mechanical_group_id`，程序会使用内置演示级参数。你也可以用 JSON 覆盖：

```json
{
  "default": {"young": 10000.0, "poisson": 0.45},
  "SAT": {"young": 5000.0, "poisson": 0.45},
  "SKIN": {"young": 50000.0, "poisson": 0.45},
  "SOFT": {"young": 12000.0, "poisson": 0.45},
  "BONE": {"young": 1000000.0, "poisson": 0.30}
}
```

运行：

```bash
satmorph solve \
  --input outputs/human-coarse.npz \
  --target-label 1 \
  --bone-tag BONE \
  --target-volume-ratio 1.2 \
  --materials materials.json \
  --output outputs/human-sat-120-custom-materials
```

## 各向同性假设

当前求解器是各向同性 Neo-Hookean。这个假设对 SAT 体积增减的第一版近似可以接受，因为我们主要控制的是体积增长；但对皮肤、肌肉、肌腱韧带、椎间盘、软骨并不完全真实。

当前处理策略：

```text
SAT              各向同性体积增长
普通脂肪/黄骨髓  脂肪样被动材料，不主动作为 SAT 增长
皮肤             各向同性等效材料，后续应加入纤维方向
肌肉             各向同性等效材料，后续应加入肌纤维方向
韧带/软骨/椎间盘  各向同性等效材料，后续应加入各向异性或纤维增强模型
骨               固定约束或高刚度材料
```

真正各向异性求解需要额外输入纤维方向或结构方向，否则代码无法凭空知道肌肉/皮肤的方向性。

## 常见问题

### 结果还是方块人

`.vtu` 是粗 FEM 结果，本来就会块状。展示时应使用：

```text
extract-visual-surface
map-surface
```

最终看 `human-sat-120-visual.vtp`。

### `outside_points` 很多

说明高分辨率表面和粗 FEM 网格不完全对齐，或粗 FEM 网格没有包住表面。可以尝试：

```text
1. 减小 surface_stride
2. 减小 stride
3. 检查坐标单位是否一致
4. 检查表面和粗网格是否来自同一个姿态
```

### 求解不收敛

可以尝试：

```bash
satmorph solve \
  --input outputs/human-coarse.npz \
  --target-label 1 \
  --bone-tag BONE \
  --target-volume-ratio 1.2 \
  --increments 24 \
  --max-iterations 50 \
  --output outputs/human-sat-120
```

也可以先把目标比例调小，例如 `1.05` 或 `0.95`。

### 模型太大

先增大 `stride`：

```bash
--stride 30
```

确认流程没问题后，再逐步降低到：

```bash
--stride 20
--stride 15
--stride 10
```

## 推荐的当前实验组合

建议先跑三组：

```bash
satmorph solve --input outputs/human-coarse.npz --target-label 1 --bone-tag BONE --target-volume-ratio 0.8 --output outputs/human-sat-080
satmorph solve --input outputs/human-coarse.npz --target-label 1 --bone-tag BONE --target-volume-ratio 1.0 --output outputs/human-sat-100
satmorph solve --input outputs/human-coarse.npz --target-label 1 --bone-tag BONE --target-volume-ratio 1.2 --output outputs/human-sat-120
```

也可以用 `solve-series` 一次跑完：

```bash
satmorph solve-series \
  --input outputs/human-coarse.npz \
  --target-label 1 \
  --bone-tag BONE \
  --ratio 0.8 \
  --ratio 1.0 \
  --ratio 1.2 \
  --output-dir outputs/sat-series \
  --prefix human-sat
```

会生成：

```text
outputs/sat-series/human-sat-080.npz
outputs/sat-series/human-sat-080.vtu
outputs/sat-series/human-sat-080.json
outputs/sat-series/human-sat-100.*
outputs/sat-series/human-sat-120.*
```

然后分别映射到同一个平滑表面：

```bash
satmorph map-surface --coarse-result outputs/human-sat-080.npz --surface outputs/human-visual-surface.vtp --output outputs/human-sat-080-visual.vtp --report outputs/human-sat-080-visual-map.json
satmorph map-surface --coarse-result outputs/human-sat-100.npz --surface outputs/human-visual-surface.vtp --output outputs/human-sat-100-visual.vtp --report outputs/human-sat-100-visual-map.json
satmorph map-surface --coarse-result outputs/human-sat-120.npz --surface outputs/human-visual-surface.vtp --output outputs/human-sat-120-visual.vtp --report outputs/human-sat-120-visual-map.json
```

也可以用 `map-series` 批量映射：

```bash
satmorph map-series \
  --input-dir outputs/sat-series \
  --pattern "*.npz" \
  --surface outputs/human-visual-surface.vtp \
  --output-dir outputs/sat-series-visual \
  --output-suffix=-visual.vtp \
  --report
```

如果当前环境没有安装 `meshio`，先导出不依赖额外写网格库的 `.npz`：

```bash
satmorph map-series \
  --input-dir outputs/sat-series \
  --pattern "*.npz" \
  --surface outputs/human-visual-surface.npz \
  --output-dir outputs/sat-series-visual \
  --output-suffix=-visual.npz \
  --report
```

最后把所有求解 JSON 汇总成 CSV：

```bash
satmorph summarize-series \
  --input-dir outputs/sat-series \
  --pattern "*.json" \
  --output outputs/sat-series-summary.csv
```

这个 CSV 适合直接放进周报或论文复现对比表，重点列包括：

```text
target_volume_ratio
actual_target_volume_ratio
target_volume_error_percent
minimum_total_jacobian
maximum_displacement
source_label_1_volume_ratio
region_SAT_volume_ratio
```

这三组最适合放进周报或 PPT，能直观看到减少、原始、增加 SAT 的效果。

## 高可信 SAT 流程（新增）

下面这套命令把“局部加密、体积审计、外层校正、质量检查和论文图”串在一起。坐标单位为 `m` 时，`--search-distance 0.005` 表示 5 mm。

### 1. 多组织边界感知的自适应转换

```bash
satmorph convert-voxel-mat-adaptive \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-adaptive.npz \
  --report outputs/human-adaptive.json \
  --coarse-stride 20 \
  --refine-stride 5 \
  --refine-all-boundaries \
  --refine-halo-blocks 1
```

自适应模式在多组织混合边界块中加入更密的采样点，再做全局 Delaunay 四面体化，因此没有悬挂节点。只关心 SAT/皮肤边界时，可把 `--refine-all-boundaries` 换成重复的 `--refine-label 1 --refine-label 2`，显著减少点数。它仍属于研究型转换器：四面体标签由中心点回采原始体素确定，正式求解前必须运行 `volume-audit` 和 `quality-report`。

已在当前 `598 x 263 x 1645` 的真实 73 标签 MAT 上做过低成本验证：

| 网格 | 四面体 | 总组织体积 | SAT 误差 | Skin 误差 | Liver 误差 |
|---|---:|---:|---:|---:|---:|
| 原始体素 | - | 0.062866 m³ | - | - | - |
| 规则 `stride=40` | 10,644 | 0.113254 m³ | +160.66% | +443.84% | - |
| 自适应 `40/20` | 47,660 | 0.062608 m³ | -2.47% | -0.45% | -1.88% |

这组结果说明自适应采样明显改善了主体组织体积，但 `Ureter/Urethra`、`Eye Lens/Cornea`、部分脑内小结构在 20 体素细化尺度下仍会消失。正式的 73 标签全保留网格需要继续缩小 `refine_stride`，或针对审计报告中的 `missing_source_labels` 再做一次标签定向加密。

### 2. 对照原始体素逐组织检查体积

```bash
satmorph volume-audit \
  --voxel-mat combined_material_label_model_001_073_1mm.mat \
  --mesh outputs/human-adaptive.npz \
  --output outputs/human-volume-audit.csv \
  --report outputs/human-volume-audit.json
```

CSV 每行对应一个原始标签，重点看 `mesh_vs_voxel_error_percent`。JSON 还包含缺失标签、平均/最大绝对体积误差和参考网格质量。

### 3. 可选：建立器官间无穿透接触

```bash
satmorph build-contact \
  --input outputs/human-adaptive.npz \
  --slave-label 35 \
  --master-label 26 \
  --search-distance 0.005 \
  --penalty 100000 \
  --output outputs/liver-stomach-contact.json
```

当前实现是参考法向的节点-三角面罚接触，适合防止邻近但网格节点不共享的器官表面穿透。对于共享节点的 conforming 组织界面，当前网格本身是绑定界面，不应重复建立接触。它还不是 Sim4Life/Abaqus 级的有限滑移接触；大位移时需要更新接触搜索和法向。

### 4. 外层校正到实际 SAT 体积比

```bash
satmorph calibrate-growth \
  --input outputs/human-adaptive.npz \
  --target-label 1 \
  --bone-tag BONE \
  --desired-volume-ratio 1.20 \
  --calibration-tolerance 0.0025 \
  --max-corrections 4 \
  --contact outputs/liver-stomach-contact.json \
  --output outputs/human-sat-120-calibrated
```

外层循环会重复调用非线性 FEM，并用乘法更新/割线更新调整内部的 `growth_lambda^3`，直到约束后的实际 SAT 体积比接近 1.20。结果 JSON 中：

```text
desired_target_volume_ratio       真正希望达到的 SAT 体积比
target_volume_ratio_unconstrained 最后一次施加的自由生长体积比
actual_target_volume_ratio        FEM 平衡后的实际 SAT 体积比
calibration_iterations            每轮校正历史
contact                           接触激活和穿透统计
```

### 5. 检查变形后网格质量

```bash
satmorph quality-report \
  --result outputs/human-sat-120-calibrated.npz \
  --output outputs/human-sat-120-quality.json
```

至少要求 `negative_or_zero_volume_count = 0`。`mean_ratio_quality` 越接近 1 越好；低于 0.1 的单元应重点检查。最终求解 JSON 的 `minimum_total_jacobian` 也必须大于 0。

### 6. 生成统一对齐的论文级外表面对比图

先对 80%、100%、120% 三个结果运行 `map-surface`，建议输出 `.npz` 以完整保留位移和标签数据，然后：

```bash
satmorph paper-figure \
  --input outputs/human-sat-080-visual.npz \
  --input outputs/human-sat-100-visual.npz \
  --input outputs/human-sat-120-visual.npz \
  --label "SAT 80%" \
  --label "Reference" \
  --label "SAT 120%" \
  --color-by displacement \
  --output outputs/human-sat-paper-comparison.png \
  --report outputs/human-sat-paper-comparison.json \
  --dpi 300
```

所有案例共用模型边界、正交相机、观察角度和色标，因此图片可以直接横向比较。灰色细线是未变形参考表面；使用 `--no-reference` 可关闭。输出也可使用 `.pdf`。

## 第二轮高级功能

### 1. 用审计报告驱动三级局部加密

第一次自适应转换和 `volume-audit` 完成后，可把缺失标签和高误差标签自动送入最细层：

```bash
satmorph convert-voxel-mat-adaptive \
  --input combined_material_label_model_001_073_1mm.mat \
  --output outputs/human-adaptive-40-20-10.npz \
  --report outputs/human-adaptive-40-20-10.json \
  --coarse-stride 40 \
  --refine-stride 20 \
  --fine-stride 10 \
  --refine-all-boundaries \
  --audit-report outputs/human-volume-audit-40-20.json \
  --volume-error-threshold 5
```

`--audit-report` 可以重复使用。程序会累计所有历史报告，关键标签集合只增不减，避免全局 Delaunay 重三角化后已恢复的小标签再次丢失：

```bash
--audit-report outputs/audit-round-1.json \
--audit-report outputs/audit-round-2.json
```

真实模型验证结果：

| 采样 | 四面体 | 平均逐标签误差 | 缺失标签 |
|---|---:|---:|---:|
| 规则 40 | 10,644 | 94.49% | 多个 |
| 自适应 40/20 | 47,660 | 26.70% | 11 |
| 审计三级 40/20/10 | 140,497 | 14.29% | 4 |
| 累计审计 40/20/5 | 817,776 | 9.71% | 3 |

最后三个缺失结构是体积只有约 `8e-9` 到 `1.6e-8 m³` 的微小脑/气道标签。它们小于当前 FEM 的有效机械分辨率；若必须保留，需要 `stride=1` 的局部嵌入网格或专门的多尺度方法。

### 2. 动态有限滑移近似接触

```bash
satmorph build-contact \
  --input outputs/human-adaptive.npz \
  --slave-label 35 \
  --master-label 44 \
  --search-distance 0.02 \
  --penalty 100000 \
  --dynamic \
  --output outputs/liver-spleen-dynamic-contact.json
```

然后在 `solve` 或 `calibrate-growth` 中加入：

```bash
--contact outputs/liver-spleen-dynamic-contact.json
```

动态模式每次装配都会更新最近主三角面、重心投影和当前法向，允许从一个主面滑移到另一个主面。当前切线在一次装配内冻结投影与法向，因此属于准 Newton 罚接触；`search-distance` 和 `penalty` 需要做敏感性分析。

### 3. 生成并使用纤维方向场

```bash
satmorph build-fiber-field \
  --input outputs/human-adaptive.npz \
  --output outputs/human-adaptive-fibers.npz \
  --report outputs/human-adaptive-fibers.json \
  --longitudinal-axis 2
```

皮肤方向取身体表面近似环向，肌肉、肌腱和软骨方向取每个组织的主轴。当前真实网格中有 20,679 个单元获得了有效方向。默认演示材料对 `SKIN`、`MUSCLE`、`TENDON_LIGAMENT` 启用纤维增强；没有 `fiber_direction` 的网格仍退化为原来的各向同性模型。

### 4. 近不可压缩锁死控制

所有求解命令现在支持：

```bash
--bulk-modulus-ratio-cap 100
```

当泊松比非常接近 0.5 时，程序限制 `kappa/mu`，降低线性四面体的体积锁死。使用 `--bulk-modulus-ratio-cap 0` 可关闭。结果 JSON 中的 `locking_control.capped_cell_count` 会记录受影响的单元数。

这是一种工程稳定化手段，不等同于严格的混合 `u-p` 单元。需要高精度压力场或严格不可压缩性时，应使用 FEniCSx/PETSc 的混合有限元实现。

### 5. 多组织 VTP/VTM 可视化

一次读取 MAT 并分别提取组织：

```bash
satmorph extract-tissue-surfaces \
  --input combined_material_label_model_001_073_1mm.mat \
  --output-dir outputs/tissue-surfaces \
  --include-label 1 \
  --include-label 2 \
  --include-label 3 \
  --include-label 11 \
  --include-label 35 \
  --include-label 52 \
  --include-label 68 \
  --include-label 69 \
  --surface-stride 4 \
  --method marching-cubes \
  --suffix .vtp
```

输出目录中的 `tissues.vtm` 可以直接在 ParaView 打开，每个组织都是独立 block。`tissues.json` 保存标签、机械组、颜色和透明度。

把同一 FEM 结果映射到所有组织：

```bash
satmorph map-tissue-surfaces \
  --coarse-result outputs/human-sat-120.npz \
  --manifest outputs/tissue-surfaces/tissues.json \
  --output-dir outputs/tissue-surfaces-sat-120
```

生成不依赖 ParaView 的论文三视图：

```bash
satmorph tissue-figure \
  --manifest outputs/tissue-surfaces/tissues.json \
  --output outputs/tissues-paper.png \
  --dpi 300
```

只显示肺、肝和骨骼：

```bash
satmorph tissue-figure \
  --manifest outputs/tissue-surfaces/tissues.json \
  --include-label 11 \
  --include-label 35 \
  --include-label 68 \
  --include-label 69 \
  --output outputs/organs-paper.png
```

程序会利用肺和肝脏的相对位置自动判断纵轴是否需要翻转。

### 6. 自动形态学验证

对映射后的外表面计算面积、封闭体积、位移、双向 Hausdorff/Chamfer 距离和腰围曲线：

```bash
satmorph surface-metrics \
  --input outputs/human-sat-120-visual.npz \
  --output outputs/human-sat-120-metrics.json \
  --profile-csv outputs/human-sat-120-circumference.csv \
  --longitudinal-axis 2 \
  --slice-count 41
```

如果已有映射后的皮肤外表面和 SAT 内边界表面，可计算最近表面厚度：

```bash
satmorph sat-thickness \
  --outer outputs/skin-120.npz \
  --inner outputs/sat-inner-120.npz \
  --output outputs/sat-thickness-120.json
```

这里的厚度是最近表面欧氏距离，不是严格沿皮肤法向的射线厚度；复杂褶皱区域需要进一步实现法向射线与 SAT 内表面的相交算法。

### 7. 性能优化

- 线搜索只需要能量和残差时，所有基础 Neo-Hookean 单元使用 NumPy 批量装配。
- 刚度矩阵局部块改为单次 `einsum`，不再执行 16 个节点对循环。
- 外层体积校正使用上一轮位移和生长比作为下一轮初值。
- 动态接触只对指定从面节点进行 KD-tree 搜索。

约 80 万四面体的网格可以完成转换和审计，但当前纯 Python 全 Newton 刚度装配仍不适合直接求解。正式大网格建议先使用 14 万单元版本做参数验证，再迁移到 PETSc/FEniCSx。

## 开发者入口

主要代码文件：

```text
src/satmorph/voxel_convert.py     体素 MAT 到粗 FEM 网格
src/satmorph/solver.py            FEM 求解器
src/satmorph/material.py          Neo-Hookean 材料模型
src/satmorph/surface_map.py       粗网格位移到高分辨率表面映射
src/satmorph/visual_surface.py    marching cubes / smoothing 表面提取
src/satmorph/tissue_groups.py     73 标签、机械组、各向同性说明
src/satmorph/audit.py             逐标签体积审计和四面体质量
src/satmorph/calibration.py       目标体积外层校正
src/satmorph/contact.py           节点-三角面无穿透罚接触
src/satmorph/adaptive_voxel.py    边界感知自适应采样/局部加密
src/satmorph/fiber.py             皮肤/肌肉/肌腱方向场
src/satmorph/metrics.py           腰围、体积、表面距离和 SAT 厚度
src/satmorph/tissue_surface.py    独立组织 VTP/VTM 提取与批量映射
src/satmorph/paper_figure.py      统一视角、尺度和色标的论文图
src/satmorph/material_library.py  73 标签文献力学参数与来源注册表
src/satmorph/physical_properties.py 标签到密度/电磁参数映射及质量核算
src/satmorph/remesh.py            标签保持的共形 edge-star 四面体二分
src/satmorph/adaptive_growth.py   分阶段有限生长、重网格和历史量转移
src/satmorph/cli.py               命令行入口
```

测试：

```text
tests/test_material.py
tests/test_solver.py
tests/test_voxel_convert.py
tests/test_surface_map.py
tests/test_visual_surface.py
tests/test_tissue_groups.py
tests/test_audit.py
tests/test_contact_calibration.py
tests/test_adaptive_voxel.py
tests/test_metrics.py
tests/test_tissue_surface.py
tests/test_material_remesh_mass.py
```

## 当前最值得继续做的三件事

1. 用 14 万四面体版本正式跑 80%/100%/120% SAT 梯级，生成体积、腰围、Hausdorff、厚度和 Jacobian 对比表。
2. 用论文或实验数据标定纤维刚度、组织 Young 模量和泊松比，并记录参数来源和不确定性范围。
3. 将全 Newton 装配和线性求解迁移到 FEniCSx/PETSc，实现真正混合 `u-p` 单元与百万级网格求解。按本轮要求暂不实现刚体骨架约束。

## 文献参数、动态加密、质量与电磁标签

### 1. 生成73标签参数表

项目已经整理出可追溯的准静态力学参数，并可与移交材料中的密度和电磁参数合并：

```bash
satmorph material-table \
  --physical-materials "C:/path/to/s4l_materials_unified.json" \
  --output docs/mechanical-parameters-73.csv \
  --json-output docs/mechanical-parameters-73.json
```

现成结果位于 `docs/mechanical-parameters-73.csv/.json`，来源和适用边界见
`docs/mechanical-parameters-sources.md`。`young_pa` 是当前求解器的准静态有效值；
`reference_low_pa/reference_high_pa` 是文献量级。骨、牙和肌腱在求解值中有意软化，
不能把它们当成临床材料参数。

### 2. 给网格重新附着密度和电磁参数

```bash
satmorph attach-physical-properties \
  --input outputs/human-adaptive-audit-40-20-10.npz \
  --materials "C:/path/to/s4l_materials_unified.json" \
  --output outputs/human-physical.npz \
  --vtu-output outputs/human-physical.vtu \
  --report outputs/human-physical.json
```

程序只通过不可变的 `source_label` 查找材料，生成：

```text
mass_density_kg_per_m3
conductivity_s_per_m
relative_permittivity
em_frequency_hz
```

网格加密后可以再次运行该命令。器官标签不会转换，电磁材料会按原标签重新赋值。

### 3. 对已有SAT求解结果增加四面体

```bash
satmorph refine-result \
  --input outputs/human-sat-120.npz \
  --target-label 1 \
  --max-edges 1000 \
  --interface-mode propagate \
  --physical-materials "C:/path/to/s4l_materials_unified.json" \
  --output outputs/human-sat-120-remeshed.npz \
  --vtu-output outputs/human-sat-120-remeshed.vtu \
  --report outputs/human-sat-120-remeshed.json
```

算法选取SAT单元的最长边，并在该边的整个四面体星域同步二分。这样没有悬挂节点：

- SAT父单元的子单元仍为 `source_label=1`；
- 界面邻居可能为了共形被同步细分，但继承原器官标签；
- `material_id`、`mechanical_group_id`、纤维方向和电磁字段均继承父单元；
- 每种组织的几何体积在二分前后守恒；
- `material_reference_volume` 作为广延量按子单元保守分配。

`--interface-mode interior-only` 只细分星域全部属于目标组织的内部边，可避免增加器官单元，
但SAT层很薄时可能找不到足够的内部边。

### 4. 对减脂结果安全减少四面体

```bash
satmorph coarsen-result \
  --input outputs/human-sat-080.npz \
  --target-label 1 \
  --max-collapses 500 \
  --max-local-volume-drift 0.01 \
  --output outputs/human-sat-080-coarsened.npz \
  --vtu-output outputs/human-sat-080-coarsened.vtu \
  --report outputs/human-sat-080-coarsened.json
```

边塌缩必须同时通过端点标签邻域一致、边界类型、四面体 link condition、局部逐标签体积漂移、
重复单元和正 Jacobian 检查。删除单元的 `material_reference_volume` 只在同标签局部单元中
保守重分配，`source_label` 永不改写。严格检查可能使很薄或分辨率不足的SAT层没有可接受候选边，
这属于安全拒绝，不应通过放宽标签检查强行塌缩。

### 5. 分步增长/减脂并自动重网格

```bash
satmorph solve-remesh \
  --input outputs/human-physical.npz \
  --target-label 1 \
  --bone-tag BONE \
  --target-growth-volume-ratio 1.20 \
  --stages 4 \
  --max-edges-per-stage 1000 \
  --max-collapses-per-stage 500 \
  --remesh-mode auto \
  --interface-mode propagate \
  --output-dir outputs/human-sat-120-remesh
```

`--remesh-mode auto` 在增脂时采用共形 edge-star 二分，在减脂时采用标签安全边塌缩，
目标比为1时只传递状态。单元数量变化只调整离散分辨率；真实增减质量来自 `J_growth` 和密度。

重网格后会保留累计质量生长 `accumulated_growth_J`，并把累计弹性变形梯度存入
`elastic_history_F`。下一阶段使用
`F_elastic = F_incremental @ elastic_history_F / growth_lambda_incremental`，
因此不会再把所有组织当成无应力状态。二分时子单元精确继承父状态；塌缩时在同标签局部区域
按参考材料体积投影，报告会记录守恒误差。接触乘子和摩擦历史尚不传递，重网格后需要重建接触。

### 6. 体重报告

```bash
satmorph mass-report \
  --input outputs/human-sat-120.npz \
  --materials "C:/path/to/s4l_materials_unified.json" \
  --length-unit m \
  --output outputs/human-sat-120-mass.json
```

报告同时给出：

- `initial_mass_kg`：初始各标签体积乘密度；
- `growth_accounted_mass_kg`：按 `rho * V_reference * J_growth` 计算的生物增长质量；
- `geometric_constant_density_mass_kg`：变形后几何体积乘密度，仅用于几何核查。

弹性压缩 `J_elastic` 不应被解释为组织质量消失。对于粗体素网格，体重误差首先受原始标签体积误差影响，
必须结合 `volume-audit` 使用，不能只看 `mass-report` 的小数位。
