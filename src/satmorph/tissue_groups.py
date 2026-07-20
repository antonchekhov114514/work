from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AtlasLabel:
    label_id: int
    atlas_name: str
    material_id: str
    mechanical_group: str
    solver_role: str
    isotropy_assumption: str


MECHANICAL_GROUPS: dict[str, int] = {
    "OUTSIDE": 0,
    "SAT_FAT": 1,
    "VISCERAL_FAT": 2,
    "SKIN": 3,
    "MUSCLE": 4,
    "ORGAN_SOFT": 5,
    "LUNG_AIRWAY": 6,
    "FLUID_BLOOD": 7,
    "AIR_LUMEN": 8,
    "TENDON_LIGAMENT": 9,
    "CNS_NERVE": 10,
    "EYE": 11,
    "CARTILAGE_DISC": 12,
    "BONE_CANCELLOUS": 13,
    "BONE_CORTICAL": 14,
    "TOOTH": 15,
    "MARROW_YELLOW": 16,
}

MECHANICAL_GROUP_NAMES = {value: key for key, value in MECHANICAL_GROUPS.items()}

DEFAULT_MECHANICAL_PARAMETERS: dict[str, dict[str, float]] = {
    "SAT_FAT": {"young": 5_000.0, "poisson": 0.45},
    "VISCERAL_FAT": {"young": 5_000.0, "poisson": 0.45},
    "SKIN": {"young": 50_000.0, "poisson": 0.45},
    "MUSCLE": {"young": 15_000.0, "poisson": 0.45},
    "ORGAN_SOFT": {"young": 12_000.0, "poisson": 0.45},
    "LUNG_AIRWAY": {"young": 5_000.0, "poisson": 0.40},
    "FLUID_BLOOD": {"young": 2_000.0, "poisson": 0.49},
    "AIR_LUMEN": {"young": 1_000.0, "poisson": 0.20},
    "TENDON_LIGAMENT": {"young": 500_000.0, "poisson": 0.40},
    "CNS_NERVE": {"young": 3_000.0, "poisson": 0.45},
    "EYE": {"young": 20_000.0, "poisson": 0.45},
    "CARTILAGE_DISC": {"young": 300_000.0, "poisson": 0.40},
    "BONE_CANCELLOUS": {"young": 1_000_000.0, "poisson": 0.30},
    "BONE_CORTICAL": {"young": 10_000_000.0, "poisson": 0.30},
    "TOOTH": {"young": 10_000_000.0, "poisson": 0.30},
    "MARROW_YELLOW": {"young": 8_000.0, "poisson": 0.45},
}

_ANISOTROPIC = "anisotropic_in_literature; isotropic_solver_approximation"
_ISOTROPIC = "reasonable_isotropic_first_approximation"
_LUMEN = "not_a_structural_solid; isotropic_placeholder"

ATLAS_LABELS: dict[int, AtlasLabel] = {
    1: AtlasLabel(1, "SAT (Subcutaneous Fat)", "sat_subcutaneous_fat", "SAT_FAT", "SAT", _ISOTROPIC),
    2: AtlasLabel(2, "Skin", "skin", "SKIN", "SKIN", _ANISOTROPIC),
    3: AtlasLabel(3, "Fat", "fat", "VISCERAL_FAT", "SOFT", _ISOTROPIC),
    4: AtlasLabel(4, "Kidney (Cortex)", "kidney_cortex", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    5: AtlasLabel(5, "Trachea Lumen", "trachea_lumen", "AIR_LUMEN", "SOFT", _LUMEN),
    6: AtlasLabel(6, "Kidney (Medulla)", "kidney_medulla", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    7: AtlasLabel(7, "Ureter\\Urethra", "ureter_urethra", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    8: AtlasLabel(8, "Bronchi", "bronchi", "LUNG_AIRWAY", "SOFT", _ISOTROPIC),
    9: AtlasLabel(9, "Urine", "urine", "FLUID_BLOOD", "SOFT", _LUMEN),
    10: AtlasLabel(10, "Urinary Bladder Wall", "urinary_bladder_wall", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    11: AtlasLabel(11, "Lung", "lung", "LUNG_AIRWAY", "SOFT", _ISOTROPIC),
    12: AtlasLabel(12, "Larynx", "larynx", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    13: AtlasLabel(13, "Mucous Membrane", "mucous_membrane", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    14: AtlasLabel(14, "Trachea", "trachea", "LUNG_AIRWAY", "SOFT", _ISOTROPIC),
    15: AtlasLabel(15, "Bronchi lumen", "bronchi_lumen", "AIR_LUMEN", "SOFT", _LUMEN),
    16: AtlasLabel(16, "Air", "air", "AIR_LUMEN", "SOFT", _LUMEN),
    17: AtlasLabel(17, "Diaphragm", "diaphragm", "MUSCLE", "SOFT", _ANISOTROPIC),
    18: AtlasLabel(18, "Tongue", "tongue", "MUSCLE", "SOFT", _ANISOTROPIC),
    19: AtlasLabel(19, "Tooth", "tooth", "TOOTH", "BONE", _ISOTROPIC),
    20: AtlasLabel(20, "Tendon\\Ligament", "tendon_ligament", "TENDON_LIGAMENT", "SOFT", _ANISOTROPIC),
    21: AtlasLabel(21, "Seminal vesicle", "seminal_vesicle", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    22: AtlasLabel(22, "Ductus Deferens", "ductus_deferens", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    23: AtlasLabel(23, "Large Intestine", "large_intestine", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    24: AtlasLabel(24, "Cerebrospinal Fluid", "cerebrospinal_fluid", "FLUID_BLOOD", "SOFT", _LUMEN),
    25: AtlasLabel(25, "Small Intestine Lumen", "small_intestine_lumen", "AIR_LUMEN", "SOFT", _LUMEN),
    26: AtlasLabel(26, "Stomach", "stomach", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    27: AtlasLabel(27, "Prostate", "prostate", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    28: AtlasLabel(28, "Esophagus Lumen", "esophagus_lumen", "AIR_LUMEN", "SOFT", _LUMEN),
    29: AtlasLabel(29, "Small Intestine", "small_intestine", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    30: AtlasLabel(30, "Epididymis", "epididymis", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    31: AtlasLabel(31, "Testis", "testis", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    32: AtlasLabel(32, "Bile", "bile", "FLUID_BLOOD", "SOFT", _LUMEN),
    33: AtlasLabel(33, "Dura", "dura", "TENDON_LIGAMENT", "SOFT", _ANISOTROPIC),
    34: AtlasLabel(34, "Gallbladder", "gallbladder", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    35: AtlasLabel(35, "Liver", "liver", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    36: AtlasLabel(36, "Stomach Lumen", "stomach_lumen", "AIR_LUMEN", "SOFT", _LUMEN),
    37: AtlasLabel(37, "Esophagus", "esophagus", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    38: AtlasLabel(38, "Large Intestine Lumen", "large_intestine_lumen", "AIR_LUMEN", "SOFT", _LUMEN),
    39: AtlasLabel(39, "Pancreas", "pancreas", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    40: AtlasLabel(40, "Penis", "penis", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    41: AtlasLabel(41, "Lymphnode", "lymphnode", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    42: AtlasLabel(42, "Adrenal Gland", "adrenal_gland", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    43: AtlasLabel(43, "Salivary Gland", "salivary_gland", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    44: AtlasLabel(44, "Spleen", "spleen", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    45: AtlasLabel(45, "Thyroid Gland", "thyroid_gland", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    46: AtlasLabel(46, "Brain (Grey Matter)", "brain_grey_matter", "CNS_NERVE", "SOFT", _ISOTROPIC),
    47: AtlasLabel(47, "Eye (Sclera)", "eye_sclera", "EYE", "SOFT", _ANISOTROPIC),
    48: AtlasLabel(48, "Eye (Vitreous Humor)", "eye_vitreous_humor", "EYE", "SOFT", _LUMEN),
    49: AtlasLabel(49, "Eye (Cornea)", "eye_cornea", "EYE", "SOFT", _ANISOTROPIC),
    50: AtlasLabel(50, "Nerve", "nerve", "CNS_NERVE", "SOFT", _ANISOTROPIC),
    51: AtlasLabel(51, "Eye (Lens)", "eye_lens", "EYE", "SOFT", _ISOTROPIC),
    52: AtlasLabel(52, "Muscle", "muscle", "MUSCLE", "SOFT", _ANISOTROPIC),
    53: AtlasLabel(53, "Blood", "blood", "FLUID_BLOOD", "SOFT", _LUMEN),
    54: AtlasLabel(54, "Heart Lumen", "heart_lumen", "FLUID_BLOOD", "SOFT", _LUMEN),
    55: AtlasLabel(55, "Heart Muscle", "heart_muscle", "MUSCLE", "SOFT", _ANISOTROPIC),
    56: AtlasLabel(56, "Cerebellum", "cerebellum", "CNS_NERVE", "SOFT", _ISOTROPIC),
    57: AtlasLabel(57, "Commissura Posterior", "commissura_posterior", "CNS_NERVE", "SOFT", _ISOTROPIC),
    58: AtlasLabel(58, "Brain (White Matter)", "brain_white_matter", "CNS_NERVE", "SOFT", _ANISOTROPIC),
    59: AtlasLabel(59, "Hippocampus", "hippocampus", "CNS_NERVE", "SOFT", _ISOTROPIC),
    60: AtlasLabel(60, "Pons", "pons", "CNS_NERVE", "SOFT", _ISOTROPIC),
    61: AtlasLabel(61, "Hypophysis", "hypophysis", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    62: AtlasLabel(62, "Thalamus", "thalamus", "CNS_NERVE", "SOFT", _ISOTROPIC),
    63: AtlasLabel(63, "Pineal Body", "pineal_body", "ORGAN_SOFT", "SOFT", _ISOTROPIC),
    64: AtlasLabel(64, "Medulla Oblongata", "medulla_oblongata", "CNS_NERVE", "SOFT", _ISOTROPIC),
    65: AtlasLabel(65, "Midbrain", "midbrain", "CNS_NERVE", "SOFT", _ISOTROPIC),
    66: AtlasLabel(66, "Hypothalamus", "hypothalamus", "CNS_NERVE", "SOFT", _ISOTROPIC),
    67: AtlasLabel(67, "Commissura Anterior", "commissura_anterior", "CNS_NERVE", "SOFT", _ISOTROPIC),
    68: AtlasLabel(68, "Bone (Cancellous)", "bone_cancellous", "BONE_CANCELLOUS", "BONE", _ISOTROPIC),
    69: AtlasLabel(69, "Bone (Cortical)", "bone_cortical", "BONE_CORTICAL", "BONE", _ISOTROPIC),
    70: AtlasLabel(70, "Bone Marrow (Yellow)", "bone_marrow_yellow", "MARROW_YELLOW", "SOFT", _ISOTROPIC),
    71: AtlasLabel(71, "Intervertebral Disc", "intervertebral_disc", "CARTILAGE_DISC", "SOFT", _ANISOTROPIC),
    72: AtlasLabel(72, "Meniscus", "meniscus", "CARTILAGE_DISC", "SOFT", _ANISOTROPIC),
    73: AtlasLabel(73, "Cartilage", "cartilage", "CARTILAGE_DISC", "SOFT", _ANISOTROPIC),
}


def labels_to_solver_roles(source_labels: np.ndarray) -> np.ndarray:
    from .demo import BONE, SAT, SKIN, SOFT

    labels = np.asarray(source_labels, dtype=np.int64)
    out = np.zeros(labels.shape, dtype=np.uint8)
    for label_id, entry in ATLAS_LABELS.items():
        if entry.solver_role == "SAT":
            out[labels == label_id] = SAT
        elif entry.solver_role == "SKIN":
            out[labels == label_id] = SKIN
        elif entry.solver_role == "BONE":
            out[labels == label_id] = BONE
        elif entry.solver_role == "SOFT":
            out[labels == label_id] = SOFT
    out[(labels != 0) & (out == 0)] = SOFT
    return out


def labels_to_mechanical_group_ids(source_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(source_labels, dtype=np.int64)
    out = np.zeros(labels.shape, dtype=np.int16)
    for label_id, entry in ATLAS_LABELS.items():
        out[labels == label_id] = MECHANICAL_GROUPS[entry.mechanical_group]
    return out


def labels_to_material_ids(source_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(source_labels, dtype=np.int64)
    out = np.zeros(labels.shape, dtype=np.int16)
    for label_id in ATLAS_LABELS:
        out[labels == label_id] = label_id
    return out


def labels_for_role(role: str) -> list[int]:
    role = role.upper()
    return [label_id for label_id, entry in ATLAS_LABELS.items() if entry.solver_role == role]


def labels_for_groups(groups: Iterable[str]) -> list[int]:
    wanted = {group.upper() for group in groups}
    return [
        label_id
        for label_id, entry in ATLAS_LABELS.items()
        if entry.mechanical_group.upper() in wanted
    ]


def atlas_report() -> list[dict[str, object]]:
    return [
        {
            "label_id": entry.label_id,
            "atlas_name": entry.atlas_name,
            "material_id": entry.material_id,
            "mechanical_group": entry.mechanical_group,
            "mechanical_group_id": MECHANICAL_GROUPS[entry.mechanical_group],
            "solver_role": entry.solver_role,
            "isotropy_assumption": entry.isotropy_assumption,
        }
        for entry in ATLAS_LABELS.values()
    ]
