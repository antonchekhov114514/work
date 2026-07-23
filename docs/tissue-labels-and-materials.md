# 73 组织标签、材料表和 SAT 变形说明

## 三类不是三种组织

本项目里曾经把 73 种组织压缩成 `SAT / BONE / SOFT / SKIN`，这是为了让 FEM 求解流程先跑通。现在代码保留两层信息：

| 层级 | 字段 | 用途 |
|---|---|---|
| 求解角色 | `region` / `cell_tags` | 决定哪些单元主动增长、哪些节点固定、哪些单元被动变形 |
| 原始标签 | `source_label` | 保留 1-73 的原始组织编号，方便追踪具体组织 |
| 机械组 | `mechanical_group_id` | 把组织分成脂肪、皮肤、肌肉、器官、骨、流体等更接近力学行为的组 |
| 材料编号 | `material_id` | 与材料表中的 tissue/material id 对应，方便后续接 Sim4Life/IT'IS 数据 |

因此，“三类”更准确地说是三个求解角色：

- `SAT`：主动施加体积增长或收缩的皮下脂肪。
- `BONE`：固定锚点，主要用于去除整体刚体漂移。
- `SOFT/SKIN`：随 SAT 被动变形的组织。

## SAT 下属哪些组织

严格 SAT 只包含：

```text
1 = SAT (Subcutaneous Fat)
```

不建议默认并入 SAT 的脂肪样组织：

```text
3  = Fat
70 = Bone Marrow (Yellow)
```

原因是 label 3 更像普通脂肪/内脏脂肪或其他非皮下脂肪，label 70 是黄骨髓。它们可以有脂肪样材料参数，但不应该在“增加/减少皮下脂肪”任务里自动跟着主动增长。

## 现在的机械分组

代码中的主要机械组包括：

```text
SAT_FAT
VISCERAL_FAT
MARROW_YELLOW
SKIN
MUSCLE
ORGAN_SOFT
LUNG_AIRWAY
FLUID_BLOOD
AIR_LUMEN
TENDON_LIGAMENT
CNS_NERVE
EYE
CARTILAGE_DISC
BONE_CANCELLOUS
BONE_CORTICAL
TOOTH
```

这些分组写在 `src/satmorph/tissue_groups.py`。默认求解时，如果网格包含 `mechanical_group_id`，程序会按这些组给不同的演示级 Young 模量和泊松比，而不是把所有软组织当成同一个材料。

## 各向同性假设

当前求解器仍然使用各向同性 Neo-Hookean 模型。对本项目的目标来说：

- SAT 主动体积变化：可以作为第一版近似，因为我们主要控制体积增长 `F_in = lambda I`。
- 皮肤：真实情况下常有方向性，尤其与胶原纤维方向有关，严格说不应长期当成各向同性。
- 肌肉、心肌、舌肌、膈肌：明显受肌纤维方向影响，真实力学更接近横观各向同性或各向异性。
- 肌腱、韧带、硬脑膜、椎间盘、半月板、软骨：方向性和纤维增强效应更强。

所以代码现在的策略是：SAT 增长仍用各向同性体积增长；其他组织先用各向同性等效材料参数跑通几何变形，但在 `isotropy_assumption` 中标出哪些组织属于各向异性近似。后续如果能拿到肌肉/皮肤/韧带纤维方向，再加入各向异性本构会更合理。

## 如何分别可视化组织

转换体素模型后，粗 FEM 结果 `.vtu` 里可以在 ParaView 的 `Coloring` 中选择：

```text
region                 看 SAT/BONE/SOFT/SKIN 求解角色
source_label           看原始 1-73 组织编号
mechanical_group_id    看机械分组
material_id            看材料编号
growth_lambda          看哪些单元被施加了 SAT 增长
J_total                看总体体积变化
J_elastic              看弹性体积变化
```

如果要单独提取某个组织的平滑表面：

```bash
satmorph extract-visual-surface \
  --input combined_material_label_model_001_073_1mm.mat \
  --include-label 1 \
  --output outputs/sat-only.vtp \
  --report outputs/sat-only.json
```

常用标签：

```text
1  SAT
2  Skin
3  Fat
52 Muscle
68 Bone cancellous
69 Bone cortical
70 Yellow marrow
```

映射位移后，输出的 `.vtp` 会带有 `mapped_source_label`、`mapped_mechanical_group_id`、`mapped_region` 等点数据，可以继续在 ParaView 中按这些字段上色。
