# SAT Morphing FEM

基于 73 组织标签人体模型的皮下脂肪组织（SAT）有限元形变研究工具。

本项目从带组织标签的人体体素模型出发，生成四面体有限元网格，对指定组织施加有限增长或收缩，并将粗网格位移映射到高分辨率人体表面。当前最成熟的应用是模拟 SAT 增脂/减脂及其对人体外形的影响。

```text
73 标签人体体素模型
        ↓
四面体 FEM 网格
        ↓
组织增长/收缩求解
        ↓
体积与网格质量检查
        ↓
高分辨率表面位移映射
        ↓
ParaView / Blender / 论文图与定量指标
```

> 本项目是研究型原型，不是 Sim4Life 源码，也不能替代临床级或商业级多物理场仿真平台。

---

## 目录

1. [项目能做什么](#1-项目能做什么)
2. [安装](#2-安装)
3. [五分钟快速验证](#3-五分钟快速验证)
4. [输入数据](#4-输入数据)
5. [推荐运行 pipeline](#5-推荐运行-pipeline)
6. [如何判断结果是否可信](#6-如何判断结果是否可信)
7. [批量运行 80% / 100% / 120% SAT](#7-批量运行-80--100--120-sat)
8. [ParaView 可视化](#8-paraview-可视化)
9. [输出文件与关键字段](#9-输出文件与关键字段)
10. [高级功能入口](#10-高级功能入口)
11. [常见问题](#11-常见问题)
12. [代码结构与开发测试](#12-代码结构与开发测试)
13. [当前限制](#13-当前限制)

---

## 1. 项目能做什么

### 1.1 核心功能

- 读取含 73 种组织标签的 MATLAB 体素模型；
- 将体素模型转换为带 `source_label` 的四面体 FEM 网格；
- 支持规则粗化和边界感知的自适应局部加密；
- 按原始组织标签选择目标组织，例如：
  - `--target-label 1`：SAT；
  - `--target-label 35`：Liver；
- 对目标组织施加各向同性增长或收缩；
- 使用可压缩 Neo-Hookean 超弹性模型、Newton 迭代和线搜索求解；
- 通过外层校正控制 FEM 平衡后的实际目标组织体积；
- 检查逐组织体积误差、Jacobian、位移和四面体质量；
- 将粗 FEM 位移映射到 marching cubes 提取的高分辨率表面；
- 输出 ParaView、Blender、统计分析和论文绘图所需文件。

### 1.2 推荐使用方式

第一次使用时，只需要掌握一条主流程：

```text
convert-voxel-mat-adaptive
        ↓
volume-audit
        ↓
calibrate-growth
        ↓
quality-report
        ↓
extract-visual-surface
        ↓
map-surface
        ↓
surface-metrics / ParaView
```

接触、纤维方向、重网格、质量核算和多组织表面都属于可选模块，不应在基础流程尚未跑通时全部启用。

---

## 2. 安装

### 2.1 推荐环境

推荐使用：

- Linux；或
- Windows 11 + WSL2 Ubuntu。

Windows PowerShell 可以运行小型和中型模型，但真实人体大模型更推荐 WSL/Linux。

### 2.2 创建虚拟环境

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[visual,preprocess,mat]'
```

只运行核心 FEM：

```bash
python -m pip install -e .
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[visual,preprocess,mat]"
```

### 2.3 检查安装

```bash
satmorph --help
```

运行全部测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

---

## 3. 五分钟快速验证

先运行内置合成模型，确认命令行、求解器和输出功能正常：

```bash
mkdir -p outputs

satmorph demo \
  --target-volume-ratio 1.2 \
  --output outputs/demo-sat-120 \
  --quiet
```

预期生成：

```text
outputs/demo-sat-120.npz
outputs/demo-sat-120.vtu
outputs/demo-sat-120.json
```

验收：

- 命令无异常退出；
- 三个输出文件均存在；
- ParaView 可以打开 `.vtu`；
- JSON 中 `minimum_total_jacobian > 0`；
- 目标区域体积相对初始状态增加。

如果合成示例失败，不要直接运行真实人体模型，应先检查安装和测试结果。

---

## 4. 输入数据

### 4.1 推荐输入：73 标签体素 MAT

MAT 文件应包含：

```text
MaterialLabelGrid    三维整数组织标签体素
Axis0                第 0 维坐标或体素边界
Axis1                第 1 维坐标或体素边界
Axis2                第 2 维坐标或体素边界
```

背景标签为 `0`，不会生成四面体。

常用标签：

| 标签 | 组织 |
|---:|---|
| 1 | SAT / Subcutaneous Fat |
| 2 | Skin |
| 3 | Fat |
| 11 | Lung |
| 19 | Tooth |
| 35 | Liver |
| 52 | Muscle |
| 68 | Bone cancellous |
| 69 | Bone cortical |
| 70 | Bone marrow yellow |

完整标签说明：

```text
docs/tissue-labels-and-materials.md
```

### 4.2 已转换的有限元输入

也可以直接输入：

```text
.npz    本项目内部格式
.vtu    VTK 四面体网格
.msh    Gmsh 四面体网格
```

最小 `.npz` 结构：

```python
numpy.savez_compressed(
    "model.npz",
    points=points,          # (N, 3) 节点坐标
    tetra=tetra,            # (M, 4) 四面体连接关系
    cell_tags=cell_tags,    # (M,) 求解角色
)
```

推荐保留：

```text
source_label
material_id
mechanical_group_id
```

`source_label` 是组织身份的主键。后续的目标组织选择、材料赋值、体积审计、重网格和结果解释都依赖它，不应被改写。

### 4.3 坐标单位

接触距离、表面指标和质量计算必须与输入坐标单位一致。

例如坐标单位为米时：

```text
--search-distance 0.005
```

表示 5 mm。

---

## 5. 推荐运行 pipeline

下面以：

```text
combined_material_label_model_001_073_1mm.mat
```

为输入，模拟 SAT 达到初始体积的 120%。

```bash
INPUT_MAT="combined_material_label_model_001_073_1mm.mat"
mkdir -p outputs
```

### Step 1：生成边界感知 FEM 网格

推荐使用自适应转换：

```bash
satmorph convert-voxel-mat-adaptive \
  --input "$INPUT_MAT" \
  --output outputs/human-adaptive.npz \
  --report outputs/human-adaptive.json \
  --coarse-stride 40 \
  --refine-stride 20 \
  --refine-all-boundaries \
  --refine-halo-blocks 1
```

参数含义：

| 参数 | 含义 |
|---|---|
| `--coarse-stride` | 人体内部低变化区域的粗采样步长 |
| `--refine-stride` | 多组织边界附近的细采样步长 |
| `--refine-all-boundaries` | 对所有组织边界进行局部加密 |
| `--refine-halo-blocks` | 在边界块周围扩展细化范围 |

输出：

```text
outputs/human-adaptive.npz
outputs/human-adaptive.json
```

只想低成本调试时，可以使用规则转换：

```bash
satmorph convert-voxel-mat \
  --input "$INPUT_MAT" \
  --output outputs/human-coarse.npz \
  --surface-output outputs/human-block-surface.npz \
  --report outputs/human-coarse.json \
  --stride 20 \
  --surface-stride 4
```

规则转换更快，但组织边界和小体积标签误差通常更大，不建议直接用于最终定量结果。

### Step 2：逐组织体积审计

```bash
satmorph volume-audit \
  --voxel-mat "$INPUT_MAT" \
  --mesh outputs/human-adaptive.npz \
  --output outputs/human-volume-audit.csv \
  --report outputs/human-volume-audit.json
```

重点检查：

```text
missing_source_labels
mesh_vs_voxel_error_percent
mean_absolute_volume_error_percent
maximum_absolute_volume_error_percent
negative_or_zero_volume_count
```

如果 SAT、Skin、Bone 或研究关注的器官明显缺失或体积误差过大，应先重新生成网格，不要直接进入正式求解。

### Step 3：可选——附着物理属性

已有密度和电磁材料表时：

```bash
satmorph attach-physical-properties \
  --input outputs/human-adaptive.npz \
  --materials "C:/path/to/s4l_materials_unified.json" \
  --output outputs/human-physical.npz \
  --vtu-output outputs/human-physical.vtu \
  --report outputs/human-physical.json
```

生成的字段包括：

```text
mass_density_kg_per_m3
conductivity_s_per_m
relative_permittivity
em_frequency_hz
```

没有材料表时，继续使用 `outputs/human-adaptive.npz`。

### Step 4：求解并校正到实际 SAT 体积比

正式结果推荐使用 `calibrate-growth`：

```bash
satmorph calibrate-growth \
  --input outputs/human-adaptive.npz \
  --target-label 1 \
  --bone-tag BONE \
  --desired-volume-ratio 1.20 \
  --calibration-tolerance 0.0025 \
  --max-corrections 4 \
  --bulk-modulus-ratio-cap 100 \
  --output outputs/human-sat-120-calibrated
```

输出：

```text
outputs/human-sat-120-calibrated.npz
outputs/human-sat-120-calibrated.vtu
outputs/human-sat-120-calibrated.json
```

关键概念：

```text
desired_target_volume_ratio
    希望 FEM 平衡后真正达到的体积比

target_volume_ratio_unconstrained
    最终施加的自由增长体积比

actual_target_volume_ratio
    FEM 平衡后的实际目标组织体积比
```

快速调试时可以直接运行：

```bash
satmorph solve \
  --input outputs/human-adaptive.npz \
  --target-label 1 \
  --bone-tag BONE \
  --target-volume-ratio 1.20 \
  --increments 24 \
  --max-iterations 50 \
  --bulk-modulus-ratio-cap 100 \
  --output outputs/human-sat-120
```

`solve` 中的目标比例是施加的自由增长比例，不保证等于 FEM 平衡后的实际体积比。

### Step 5：检查变形后网格质量

```bash
satmorph quality-report \
  --result outputs/human-sat-120-calibrated.npz \
  --output outputs/human-sat-120-quality.json
```

最低要求：

```text
negative_or_zero_volume_count = 0
minimum_total_jacobian > 0
```

同时检查：

```text
mean_ratio_quality
minimum_mean_ratio_quality
maximum_edge_ratio
maximum_displacement
```

`mean_ratio_quality` 越接近 1 越好。低于 0.1 的单元需要重点定位，不应只依据平均值判断网格安全性。

### Step 6：提取高分辨率人体表面

粗 FEM `.vtu` 用于检查内部单元，不适合直接展示人体外观。建议从原始 MAT 单独提取平滑表面：

```bash
satmorph extract-visual-surface \
  --input "$INPUT_MAT" \
  --output outputs/human-visual-surface.vtp \
  --report outputs/human-visual-surface.json \
  --surface-stride 2 \
  --method marching-cubes \
  --pre-smooth-sigma 0.5 \
  --smooth-method taubin \
  --smooth-iterations 20
```

没有 `scikit-image` 时可以使用块状 fallback：

```bash
satmorph extract-visual-surface \
  --input "$INPUT_MAT" \
  --output outputs/human-visual-surface.vtp \
  --report outputs/human-visual-surface.json \
  --surface-stride 2 \
  --method blocks \
  --smooth-method taubin \
  --smooth-iterations 20
```

### Step 7：把 FEM 位移映射到高分辨率表面

```bash
satmorph map-surface \
  --coarse-result outputs/human-sat-120-calibrated.npz \
  --surface outputs/human-visual-surface.vtp \
  --output outputs/human-sat-120-visual.vtp \
  --centers-output outputs/human-sat-120-visual-centers.npz \
  --report outputs/human-sat-120-visual-map.json
```

映射后的：

```text
outputs/human-sat-120-visual.vtp
```

已经包含变形后的表面坐标，通常不需要在 ParaView 中再次使用 `Warp By Vector`。

需要继续计算表面指标时，建议将映射结果同时保存为 `.npz`。

### Step 8：计算形态学指标

```bash
satmorph surface-metrics \
  --input outputs/human-sat-120-visual.npz \
  --output outputs/human-sat-120-metrics.json \
  --profile-csv outputs/human-sat-120-circumference.csv \
  --longitudinal-axis 2 \
  --slice-count 41
```

可输出：

- 表面积；
- 封闭体积；
- 位移统计；
- 双向 Hausdorff 距离；
- Chamfer 距离；
- 沿身体纵轴的截面周长曲线。

---

## 6. 如何判断结果是否可信

### 6.1 网格转换阶段

检查 `human-volume-audit.json/.csv`：

- SAT 是否存在；
- Skin、Bone 和关注器官是否缺失；
- SAT 体积误差是否可接受；
- 小组织是否因采样步长过大而消失；
- 初始网格是否存在非正体积单元。

### 6.2 非线性求解阶段

检查求解 JSON：

- 是否达到收敛条件；
- 是否存在失败的加载增量；
- `minimum_total_jacobian` 是否大于 0；
- `maximum_displacement` 是否符合预期量级；
- `actual_target_volume_ratio` 是否接近目标；
- 外层校正是否稳定逼近，而不是振荡或发散。

### 6.3 表面映射阶段

检查：

```text
outside_points
outside_triangle_centers
maximum_point_residual
```

`outside_points` 较多通常说明：

- FEM 网格没有完整包住表面；
- 表面和网格坐标单位不一致；
- 两者不是来自同一模型姿态；
- FEM 网格过粗；
- 表面平滑将部分点推到 FEM 外部。

### 6.4 100% 对照组

`SAT 100%` 是重要的数值对照。理想情况下：

- 位移应接近 0；
- 目标组织实际体积比应接近 1；
- 不应产生明显表面形变；
- 不应出现低 Jacobian 或负体积单元。

如果 100% 对照也发生明显变形，应优先检查边界条件、初始应力、材料历史和坐标映射。

---

## 7. 批量运行 80% / 100% / 120% SAT

### 7.1 批量求解

```bash
satmorph solve-series \
  --input outputs/human-adaptive.npz \
  --target-label 1 \
  --bone-tag BONE \
  --ratio 0.8 \
  --ratio 1.0 \
  --ratio 1.2 \
  --output-dir outputs/sat-series \
  --prefix human-sat
```

预期输出：

```text
outputs/sat-series/human-sat-080.*
outputs/sat-series/human-sat-100.*
outputs/sat-series/human-sat-120.*
```

`solve-series` 适合快速批量比较。需要严格控制实际体积比时，应分别使用 `calibrate-growth`。

### 7.2 批量映射表面

```bash
satmorph map-series \
  --input-dir outputs/sat-series \
  --pattern "*.npz" \
  --surface outputs/human-visual-surface.vtp \
  --output-dir outputs/sat-series-visual \
  --output-suffix=-visual.vtp \
  --report
```

### 7.3 汇总求解结果

```bash
satmorph summarize-series \
  --input-dir outputs/sat-series \
  --pattern "*.json" \
  --output outputs/sat-series-summary.csv
```

重点列：

```text
target_volume_ratio
actual_target_volume_ratio
target_volume_error_percent
minimum_total_jacobian
maximum_displacement
source_label_1_volume_ratio
region_SAT_volume_ratio
```

### 7.4 生成统一比较图

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

---

## 8. ParaView 可视化

### 8.1 应该打开哪个文件

| 文件 | 用途 |
|---|---|
| `*.vtu` | 检查粗 FEM 内部单元、Jacobian、材料和增长区域 |
| `*-visual.vtp` | 查看平滑的最终人体外表面 |
| `tissues.vtm` | 同时查看多个独立组织 block |

### 8.2 常用着色字段

```text
region                      SAT/BONE/SOFT/SKIN 求解角色
source_label                原始 1-73 组织标签
mechanical_group_id         力学分组
material_id                 材料编号
growth_lambda               主动增长因子
J_total                     总体积 Jacobian
J_elastic                   弹性体积 Jacobian
displacement_magnitude      位移大小
mapped_source_label         映射到表面后的组织标签
mapped_mechanical_group_id  映射到表面后的机械组
fiber_direction             三分量纤维方向
```

### 8.3 选择字段后模型消失

依次尝试：

1. 点击 `Reset Camera`；
2. 将 `Coloring` 改回 `Solid Color`；
3. 再选择目标字段；
4. 点击 `Rescale to Data Range`；
5. 确认对象前的小眼睛已打开；
6. 检查当前选择的是 `Point Data` 还是 `Cell Data`。

---

## 9. 输出文件与关键字段

### 9.1 文件类型

| 后缀 | 用途 |
|---|---|
| `.npz` | 项目内部数据包，适合继续计算 |
| `.vtu` | 四面体体网格，适合 ParaView 检查 FEM 内部结果 |
| `.vtp` | 三角表面网格，适合人体展示和表面处理 |
| `.vtm` | 多组织 block 集合 |
| `.json` | 配置、求解历史、质量与映射报告 |
| `.csv` | 逐组织审计、批量汇总和周长曲线 |

### 9.2 常见 `.npz` 字段

```text
points                       原始节点坐标
tetra                        四面体连接关系
cell_tags                    SAT/BONE/SOFT/SKIN 求解角色
source_label                 原始组织标签
material_id                  材料编号
mechanical_group_id          力学分组
deformed_points              变形后节点坐标
displacement                 节点位移向量
growth_lambda                当前增长因子
accumulated_growth_J          累计生长 Jacobian
elastic_history_F            累计弹性历史
j_total                      总变形 Jacobian
j_elastic                    弹性 Jacobian
fiber_direction              单元纤维方向
material_reference_volume    参考材料体积
```

---

## 10. 高级功能入口

基础 pipeline 稳定后，再按需求启用以下功能。

### 10.1 审计驱动的三级局部加密

```bash
satmorph convert-voxel-mat-adaptive \
  --input "$INPUT_MAT" \
  --output outputs/human-adaptive-40-20-10.npz \
  --report outputs/human-adaptive-40-20-10.json \
  --coarse-stride 40 \
  --refine-stride 20 \
  --fine-stride 10 \
  --refine-all-boundaries \
  --audit-report outputs/human-volume-audit.json \
  --volume-error-threshold 5
```

用途：恢复缺失标签并降低高误差组织的体积误差。

### 10.2 纤维方向场

```bash
satmorph build-fiber-field \
  --input outputs/human-adaptive.npz \
  --output outputs/human-adaptive-fibers.npz \
  --report outputs/human-adaptive-fibers.json \
  --longitudinal-axis 2
```

用途：为皮肤、肌肉和肌腱等组织启用纤维增强项。当前方向是规则/PCA 估计，不能替代真实 DTI。

### 10.3 器官接触

```bash
satmorph build-contact \
  --input outputs/human-adaptive.npz \
  --slave-label 35 \
  --master-label 44 \
  --search-distance 0.02 \
  --penalty 100000 \
  --dynamic \
  --output outputs/liver-spleen-contact.json
```

求解时加入：

```bash
--contact outputs/liver-spleen-contact.json
```

共享节点的 conforming 界面已经是绑定界面，不应重复建立接触。重网格后需要重建接触。

### 10.4 多组织表面

```bash
satmorph extract-tissue-surfaces \
  --input "$INPUT_MAT" \
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

映射到同一 FEM 结果：

```bash
satmorph map-tissue-surfaces \
  --coarse-result outputs/human-sat-120-calibrated.npz \
  --manifest outputs/tissue-surfaces/tissues.json \
  --output-dir outputs/tissue-surfaces-sat-120
```

### 10.5 SAT 厚度

```bash
satmorph sat-thickness \
  --outer outputs/skin-120.npz \
  --inner outputs/sat-inner-120.npz \
  --output outputs/sat-thickness-120.json
```

当前厚度为最近表面欧氏距离，不是严格沿皮肤法向的射线厚度。

### 10.6 分阶段增长和自动重网格

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

`auto` 模式：

- 增脂：共形 edge-star 二分；
- 减脂：标签安全边塌缩；
- 目标比为 1：仅传递状态。

跨阶段保存 `accumulated_growth_J`、`elastic_history_F` 和 `material_reference_volume`。

### 10.7 对已有结果单独加密或粗化

加密：

```bash
satmorph refine-result \
  --input outputs/human-sat-120-calibrated.npz \
  --target-label 1 \
  --max-edges 1000 \
  --interface-mode propagate \
  --output outputs/human-sat-120-remeshed.npz \
  --report outputs/human-sat-120-remeshed.json
```

粗化：

```bash
satmorph coarsen-result \
  --input outputs/human-sat-080.npz \
  --target-label 1 \
  --max-collapses 500 \
  --max-local-volume-drift 0.01 \
  --output outputs/human-sat-080-coarsened.npz \
  --report outputs/human-sat-080-coarsened.json
```

### 10.8 材料表和质量报告

生成 73 标签材料表：

```bash
satmorph material-table \
  --physical-materials "C:/path/to/s4l_materials_unified.json" \
  --output docs/mechanical-parameters-73.csv \
  --json-output docs/mechanical-parameters-73.json
```

质量报告：

```bash
satmorph mass-report \
  --input outputs/human-sat-120-calibrated.npz \
  --materials "C:/path/to/s4l_materials_unified.json" \
  --length-unit m \
  --output outputs/human-sat-120-mass.json
```

质量报告必须结合 `volume-audit` 一起解释。弹性压缩 `J_elastic` 不代表组织质量消失。

---

## 11. 常见问题

### 11.1 结果还是方块人

你打开的是粗 FEM `.vtu`。展示时应运行：

```text
extract-visual-surface
map-surface
```

然后查看 `*-visual.vtp`。

### 11.2 `outside_points` 很多

检查：

1. 表面和 FEM 网格是否来自同一输入；
2. 坐标单位是否一致；
3. FEM 网格是否过粗；
4. 表面平滑是否过强；
5. `surface_stride`、`coarse_stride` 和 `refine_stride` 是否合适。

### 11.3 求解不收敛

按顺序尝试：

1. 将目标比例从 `1.20` 改为 `1.05`；
2. 增加 `--increments`；
3. 增加 `--max-iterations`；
4. 检查初始网格是否存在低质量或负体积单元；
5. 降低过硬材料的 Young 模量；
6. 启用 `--bulk-modulus-ratio-cap 100`；
7. 暂时关闭接触；
8. 检查固定骨区域是否过约束；
9. 使用 `solve-remesh` 将大变形拆成多个阶段。

### 11.4 模型太大或内存不足

调试阶段优先：

- 增大 `coarse_stride`；
- 增大 `refine_stride`；
- 只细化 SAT/Skin 边界；
- 限制四面体总数；
- 先在约 10 万至 15 万四面体上完成参数验证。

### 11.5 实际体积比达不到目标

- `solve` 不保证最终实际体积比等于输入比例；
- 正式控制体积时使用 `calibrate-growth`；
- 检查皮肤刚度、骨约束和接触是否限制了目标组织；
- 检查外层校正历史是否单调逼近。

### 11.6 小器官标签消失

- 减小 `refine_stride`；
- 启用 `fine_stride`；
- 使用 `volume-audit` 定位缺失标签；
- 将审计报告传入下一轮自适应转换；
- 对极小结构考虑局部 `stride=1` 或专门多尺度网格。

---

## 12. 代码结构与开发测试

### 12.1 主要源码

```text
src/satmorph/
├── cli.py                   命令行入口
├── voxel_convert.py         规则体素 MAT 到 FEM 网格
├── adaptive_voxel.py        边界感知自适应采样和局部加密
├── solver.py                非线性 FEM 求解器
├── material.py              Neo-Hookean 和纤维增强材料
├── material_library.py      73 标签力学参数和来源
├── physical_properties.py   密度、电磁属性和质量核算
├── tissue_groups.py         组织标签、机械组和求解角色
├── audit.py                 逐标签体积审计和网格质量
├── calibration.py           实际体积外层校正
├── contact.py               节点—三角面罚接触
├── fiber.py                 纤维方向场
├── visual_surface.py        marching cubes 和表面平滑
├── surface_map.py           粗 FEM 位移到表面映射
├── tissue_surface.py        多组织 VTP/VTM
├── metrics.py               表面、腰围、距离和 SAT 厚度
├── paper_figure.py          统一视角和色标的论文图
├── remesh.py                共形细分和标签安全粗化
└── adaptive_growth.py       分阶段增长、重网格和状态转移
```

### 12.2 测试

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

### 12.3 修改代码后的最低回归要求

1. `satmorph demo` 能运行；
2. 全部单元测试通过；
3. `source_label` 数量和逐标签体积没有异常漂移；
4. 100% 对照组位移接近 0；
5. 80% / 100% / 120% 实际 SAT 体积比顺序正确；
6. 所有结果 `minimum_total_jacobian > 0`；
7. 表面映射的 `outside_points` 没有异常增加；
8. 重网格前后逐标签参考材料体积近似守恒。

### 12.4 建议归档的运行信息

```text
输入文件哈希
Git commit
完整命令
材料 JSON
网格转换报告
体积审计报告
求解 JSON
质量报告
表面映射报告
结果汇总 CSV
```

---

## 13. 当前限制

- 内置材料参数主要用于数值演示，不能直接作为临床材料参数；
- 规则/PCA 纤维方向不能替代 DTI 或真实解剖纤维图谱；
- 动态接触使用准 Newton 近似，尚未包含严格几何一致切线和摩擦历史转移；
- 当前锁死控制不等同于严格混合 `u-p` 单元；
- 纯 Python Newton 装配不适合直接求解百万级四面体；
- 尚未实现完整的器官姿态控制、刚体骨架约束、GUI 和多物理场耦合；
- 在材料、边界条件和网格收敛未完成标定前，结果应解释为研究型形变趋势，而不是人体真实生理变化的定量预测。

当前建议优先完成：

1. 在经过审计的中等规模网格上稳定运行 SAT 80% / 100% / 120%；
2. 建立体积、Jacobian、网格质量、腰围和表面距离的自动验收表；
3. 用文献或实验数据标定材料和边界条件；
4. 将大规模求解迁移到 PETSc/FEniCSx，并进一步实现混合 `u-p` 单元。
