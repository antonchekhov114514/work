# 73标签力学参数表：来源与使用边界

## 文件

- `mechanical-parameters-73.csv`：便于Excel查看的73行参数表。
- `mechanical-parameters-73.json`：包含相同参数、来源注册表和程序可读元数据。

表中 `young_pa` 是当前准静态SAT形变求解器采用的有效杨氏模量，
`reference_low_pa/reference_high_pa` 是文献量级或工程模型范围。两者不能混为一谈：
骨、牙、肌腱等组织在表中被有意软化，避免当前一阶四面体纯Python求解器出现极端条件数。

## 主要来源

| source key | 主要用途 | 链接 |
|---|---|---|
| `itis_v4_2` | 质量密度、电导率、介电常数；不提供机械弹性 | https://itis.swiss/virtual-population/tissue-properties/downloads/database-v4-2 |
| `alkhouli_2013` | 人体SAT与网膜/内脏脂肪的初始、终末模量和松弛 | https://doi.org/10.1152/ajpendo.00111.2013 |
| `sun_2023` | 体内脂肪和肌肉的应变相关剪切模量 | https://pubmed.ncbi.nlm.nih.gov/37276651/ |
| `ni_annaidh_2012` | 人体皮肤各向异性拉伸实验 | https://doi.org/10.1016/j.jmbbm.2011.08.016 |
| `piper_adult_2023` | 静态全身人体有限元模型的软组织、皮肤和骨工程参数 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10267746/ |
| `singh_chanda_2021` | 全身人体软组织力学性质综述 | https://doi.org/10.1088/1748-605X/ac2b7a |
| `brain_review_2025` | 人脑灰质、白质、区域与加载速率差异 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12045392/ |
| `pydi_2023` | 人肺实质准静态和动态压缩 | https://doi.org/10.1007/s10237-023-01751-0 |
| `joshi_2024` | 新鲜人体胰腺压痕模量 | https://pubmed.ncbi.nlm.nih.gov/38255197/ |
| `bladder_review_2024` | 膀胱、SAT、皮肤的组织仿体目标范围 | https://doi.org/10.1002/advs.202400271 |
| `kim_2013` | 松质骨、软骨、韧带、皮肤、SAT人体模型材料卡 | https://doi.org/10.1002/oby.20355 |
| `gao_2010` | 肝组织非线性本构建模 | https://doi.org/10.1007/s10439-009-9812-0 |

## 生长与重网格依据

- Rodriguez、Hoger 和 McCulloch 的有限生长框架将总变形分解为弹性部分和生长部分，
  即 `F = Fe Fg`。本项目中的 `growth_lambda`/`J_growth` 用来描述材料增加或减少，
  而不是通过修改标签或单元数量伪造质量变化：
  https://doi.org/10.1016/0021-9290(94)90021-3
- Kennaway 和 Coen 的体积生长有限元工作指出，大幅差异生长会造成单元尺寸和形状恶化，
  因而需要动态细分；其四面体示例同样沿共享边同步分裂以保持连续网格：
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6544983/
- CGAL 的多材料四面体重网格算法以单元 subdomain index 保留材料域和界面拓扑，并组合
  edge split、collapse、flip 和 relocation。当前Python实现包含共形 edge-star bisection，
  以及带标签邻域、link condition、边界、局部体积和Jacobian检查的保守edge collapse；
  edge flip、目标尺寸场和边界重投影仍属于后续工作：
  https://doc.cgal.org/latest/Tetrahedral_remeshing/

因此，增加SAT四面体数量本身不增加体重。体重变化按
`density * material_reference_volume * J_growth` 计算；细分只把父单元的参考材料体积
保守分配给子单元。器官单元即使为了界面共形被同步细分，也始终继承原 `source_label`。

## 参数解释

- `young_pa`、`poisson`：当前可压缩Neo-Hookean模型的输入。
- `fiber_stiffness`：皮肤、肌肉、肌腱、椎间盘等方向增强项的有效刚度。
- `model`：当前参数应搭配的近似模型，而不是宣称真实组织就是该模型。
- `confidence`：`high/medium/low/placeholder`，其中 `placeholder` 表示该标签本质上不是结构固体。
- `source_keys`：分号分隔的来源键，可在JSON的 `sources` 中追溯完整链接。
- `mass_density_kg_per_m3`、`conductivity_s_per_m`、`relative_permittivity`：来自项目移交的
  `s4l_materials_unified.json`，其组织映射基于IT'IS/Sim4Life材料名称。

## 不能直接当固体的标签

空气、管腔、血液、尿液、胆汁、脑脊液、眼玻璃体目前使用低剪切刚度占位材料，
目的是维持共形体网格和完成SAT形变。高可信器官力学应将它们替换为压力腔、流体、
流固耦合或多孔介质模型。占位模量不能解释为这些流体的真实杨氏模量。

## 重新生成

```bash
satmorph material-table \
  --physical-materials "C:/path/to/s4l_materials_unified.json" \
  --output docs/mechanical-parameters-73.csv \
  --json-output docs/mechanical-parameters-73.json
```

这套参数用于研究型准静态形变，不是患者特异、临床诊断或损伤预测材料卡。
