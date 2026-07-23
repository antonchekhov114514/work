from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from .tissue_groups import ATLAS_LABELS


SOURCE_REGISTRY: dict[str, dict[str, str]] = {
    "itis_v4_2": {
        "citation": "IT'IS Foundation, Tissue Properties Database V4.2 (2024)",
        "url": "https://itis.swiss/virtual-population/tissue-properties/downloads/database-v4-2",
        "use": "mass density and electromagnetic properties; not mechanical elasticity",
    },
    "alkhouli_2013": {
        "citation": "Alkhouli et al., Am J Physiol Endocrinol Metab 305:E1427-E1435 (2013)",
        "url": "https://doi.org/10.1152/ajpendo.00111.2013",
        "use": "human subcutaneous and omental adipose tensile/relaxation moduli",
    },
    "sun_2023": {
        "citation": "Sun et al., J Mech Behav Biomed Mater 143:105891 (2023)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37276651/",
        "use": "in-vivo strain-dependent shear modulus of human fat and muscle",
    },
    "ni_annaidh_2012": {
        "citation": "Ni Annaidh et al., J Mech Behav Biomed Mater 5:139-148 (2012)",
        "url": "https://doi.org/10.1016/j.jmbbm.2011.08.016",
        "use": "anisotropic tensile response of excised human skin",
    },
    "piper_adult_2023": {
        "citation": "Beillas et al., Front Bioeng Biotechnol 11:1170768 (2023)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10267746/",
        "use": "validated static full-body engineering priors for soft tissue, skin and bone",
    },
    "singh_chanda_2021": {
        "citation": "Singh and Chanda, Biomed Mater 16:062004 (2021)",
        "url": "https://doi.org/10.1088/1748-605X/ac2b7a",
        "use": "whole-body human soft-tissue mechanical-property review",
    },
    "brain_review_2025": {
        "citation": "Mechanical Characterization of Brain Tissue review (2025)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12045392/",
        "use": "human brain regional, rate and constitutive-model variability",
    },
    "pydi_2023": {
        "citation": "Pydi et al., Biomech Model Mechanobiol 22:2083-2096 (2023)",
        "url": "https://doi.org/10.1007/s10237-023-01751-0",
        "use": "human lung parenchyma quasi-static and dynamic compression",
    },
    "joshi_2024": {
        "citation": "Capretti et al., Bioengineering 11:89 (2024)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38255197/",
        "use": "fresh human pancreas indentation modulus",
    },
    "bladder_review_2024": {
        "citation": "Fernandez et al., Advanced Science 11:2400271 (2024)",
        "url": "https://doi.org/10.1002/advs.202400271",
        "use": "human bladder, SAT and skin modulus ranges for tissue phantoms",
    },
    "kim_2013": {
        "citation": "Kim et al., Obesity 21:1459-1466 (2013)",
        "url": "https://doi.org/10.1002/oby.20355",
        "use": "trabecular bone, cartilage, ligament, skin and SAT engineering material cards",
    },
    "gao_2010": {
        "citation": "Gao et al., Ann Biomed Eng 38:505-516 (2010)",
        "url": "https://doi.org/10.1007/s10439-009-9812-0",
        "use": "nonlinear constitutive modeling of liver tissue",
    },
}


# These are quasi-static effective parameters for the current compressible
# Neo-Hookean P1-tetra solver. Reference ranges describe literature scale, while
# solver values may be softened to keep a whole-body morphing solve tractable.
GROUP_PRESETS: dict[str, dict[str, object]] = {
    "SAT_FAT": {
        "young_pa": 5_000.0,
        "poisson": 0.49,
        "reference_low_pa": 1_600.0,
        "reference_high_pa": 24_000.0,
        "model": "near-incompressible Neo-Hookean effective fit",
        "confidence": "high",
        "sources": ["alkhouli_2013", "sun_2023"],
        "note": "SAT initial modulus is low and increases strongly with strain.",
    },
    "VISCERAL_FAT": {
        "young_pa": 8_000.0,
        "poisson": 0.49,
        "reference_low_pa": 2_900.0,
        "reference_high_pa": 32_000.0,
        "model": "near-incompressible Neo-Hookean effective fit",
        "confidence": "high",
        "sources": ["alkhouli_2013"],
        "note": "Omental/visceral fat is generally stiffer than paired SAT.",
    },
    "SKIN": {
        "young_pa": 635_000.0,
        "poisson": 0.45,
        "fiber_stiffness": 1_000_000.0,
        "reference_low_pa": 420_000.0,
        "reference_high_pa": 2_000_000.0,
        "model": "fiber-reinforced Neo-Hookean effective fit",
        "confidence": "medium",
        "sources": ["ni_annaidh_2012", "piper_adult_2023", "kim_2013"],
        "note": "Skin is anisotropic; this modulus targets low-rate whole-body morphing, not failure.",
    },
    "MUSCLE": {
        "young_pa": 20_000.0,
        "poisson": 0.49,
        "fiber_stiffness": 40_000.0,
        "reference_low_pa": 3_000.0,
        "reference_high_pa": 100_000.0,
        "model": "passive fiber-reinforced Neo-Hookean effective fit",
        "confidence": "medium",
        "sources": ["sun_2023", "piper_adult_2023", "singh_chanda_2021"],
        "note": "Passive muscle only; no active contraction is represented.",
    },
    "ORGAN_SOFT": {
        "young_pa": 12_000.0,
        "poisson": 0.49,
        "reference_low_pa": 1_000.0,
        "reference_high_pa": 100_000.0,
        "model": "near-incompressible Neo-Hookean surrogate",
        "confidence": "low",
        "sources": ["singh_chanda_2021", "piper_adult_2023"],
        "note": "Fallback only; label-specific overrides are preferred for major organs.",
    },
    "LUNG_AIRWAY": {
        "young_pa": 43_000.0,
        "poisson": 0.35,
        "reference_low_pa": 3_000.0,
        "reference_high_pa": 153_000.0,
        "model": "compressible nonlinear-solid approximation",
        "confidence": "medium",
        "sources": ["pydi_2023"],
        "note": "Depends strongly on inflation and strain rate; 43 kPa is a quasi-static compression scale.",
    },
    "FLUID_BLOOD": {
        "young_pa": 500.0,
        "poisson": 0.49,
        "reference_low_pa": 0.0,
        "reference_high_pa": 2_000.0,
        "model": "low-shear solid placeholder",
        "confidence": "placeholder",
        "sources": ["piper_adult_2023"],
        "note": "A fluid/cavity formulation is physically preferable; this value only stabilizes the solid mesh.",
    },
    "AIR_LUMEN": {
        "young_pa": 50.0,
        "poisson": 0.20,
        "reference_low_pa": 0.0,
        "reference_high_pa": 100.0,
        "model": "very-soft solid placeholder",
        "confidence": "placeholder",
        "sources": [],
        "note": "Air is not a structural solid; use pressure/cavity elements in a higher-fidelity model.",
    },
    "TENDON_LIGAMENT": {
        "young_pa": 5_000_000.0,
        "poisson": 0.40,
        "fiber_stiffness": 50_000_000.0,
        "reference_low_pa": 100_000_000.0,
        "reference_high_pa": 1_000_000_000.0,
        "model": "softened fiber-reinforced Neo-Hookean",
        "confidence": "medium",
        "sources": ["kim_2013", "singh_chanda_2021"],
        "note": "Solver value is deliberately softened; literature tensile moduli are much higher.",
    },
    "CNS_NERVE": {
        "young_pa": 1_500.0,
        "poisson": 0.49,
        "reference_low_pa": 100.0,
        "reference_high_pa": 10_000.0,
        "model": "near-incompressible Neo-Hookean effective fit",
        "confidence": "medium",
        "sources": ["brain_review_2025"],
        "note": "White matter and peripheral nerve anisotropy are simplified.",
    },
    "EYE": {
        "young_pa": 100_000.0,
        "poisson": 0.45,
        "fiber_stiffness": 250_000.0,
        "reference_low_pa": 10_000.0,
        "reference_high_pa": 2_000_000.0,
        "model": "label-dependent eye-tissue surrogate",
        "confidence": "low",
        "sources": ["singh_chanda_2021"],
        "note": "Sclera, cornea, lens and vitreous should ultimately use separate models.",
    },
    "CARTILAGE_DISC": {
        "young_pa": 500_000.0,
        "poisson": 0.45,
        "fiber_stiffness": 2_000_000.0,
        "reference_low_pa": 300_000.0,
        "reference_high_pa": 10_000_000.0,
        "model": "softened fiber-reinforced Neo-Hookean",
        "confidence": "medium",
        "sources": ["kim_2013", "singh_chanda_2021"],
        "note": "Disc, meniscus and articular cartilage are distinct biphasic/fiber tissues.",
    },
    "BONE_CANCELLOUS": {
        "young_pa": 5_000_000.0,
        "poisson": 0.30,
        "reference_low_pa": 44_800_000.0,
        "reference_high_pa": 1_000_000_000.0,
        "model": "softened isotropic elastic approximation",
        "confidence": "medium",
        "sources": ["kim_2013"],
        "note": "Reference stiffness depends strongly on apparent density and anatomical site.",
    },
    "BONE_CORTICAL": {
        "young_pa": 10_000_000.0,
        "poisson": 0.30,
        "reference_low_pa": 6_000_000_000.0,
        "reference_high_pa": 20_000_000_000.0,
        "model": "softened isotropic elastic approximation",
        "confidence": "medium",
        "sources": ["piper_adult_2023", "singh_chanda_2021"],
        "note": "Bones are normally fixed/rigid in SAT morphing; solver value avoids extreme conditioning.",
    },
    "TOOTH": {
        "young_pa": 10_000_000.0,
        "poisson": 0.30,
        "reference_low_pa": 10_000_000_000.0,
        "reference_high_pa": 80_000_000_000.0,
        "model": "softened isotropic elastic approximation",
        "confidence": "low",
        "sources": ["singh_chanda_2021"],
        "note": "Dental tissues are rigid for the present whole-body morphing purpose.",
    },
    "MARROW_YELLOW": {
        "young_pa": 8_000.0,
        "poisson": 0.49,
        "reference_low_pa": 1_000.0,
        "reference_high_pa": 50_000.0,
        "model": "fat-like Neo-Hookean surrogate",
        "confidence": "low",
        "sources": ["alkhouli_2013", "singh_chanda_2021"],
        "note": "Yellow marrow remains label 70 and is never treated as SAT growth.",
    },
}


LABEL_OVERRIDES: dict[int, dict[str, object]] = {
    10: {
        "young_pa": 100_000.0,
        "reference_low_pa": 70_000.0,
        "reference_high_pa": 430_000.0,
        "sources": ["bladder_review_2024"],
        "confidence": "medium",
        "note": "Human bladder wall range; empty/full state remains unmodeled.",
    },
    24: {
        "young_pa": 100.0,
        "model": "low-shear CSF placeholder",
        "note": "Use fluid/poroelastic treatment for brain mechanics.",
    },
    35: {
        "young_pa": 6_000.0,
        "reference_low_pa": 2_000.0,
        "reference_high_pa": 20_000.0,
        "sources": ["gao_2010", "singh_chanda_2021"],
        "confidence": "medium",
        "note": "Quasi-static healthy-liver effective value; perfusion and viscoelasticity are omitted.",
    },
    39: {
        "young_pa": 5_000.0,
        "reference_low_pa": 1_400.0,
        "reference_high_pa": 148_000.0,
        "sources": ["joshi_2024", "singh_chanda_2021"],
        "confidence": "low",
        "note": "Published pancreas values vary by orders of magnitude with test scale and condition.",
    },
    46: {"young_pa": 1_000.0, "sources": ["brain_review_2025"]},
    48: {
        "young_pa": 100.0,
        "fiber_stiffness": 0.0,
        "model": "low-shear vitreous placeholder",
    },
    55: {
        "young_pa": 30_000.0,
        "fiber_stiffness": 80_000.0,
        "note": "Passive myocardium surrogate; no active cardiac contraction.",
    },
    58: {"young_pa": 1_500.0, "sources": ["brain_review_2025"]},
    71: {"young_pa": 500_000.0, "fiber_stiffness": 2_000_000.0},
    72: {"young_pa": 2_000_000.0, "fiber_stiffness": 10_000_000.0},
    73: {"young_pa": 1_000_000.0, "fiber_stiffness": 2_000_000.0},
}


def mechanical_record(label_id: int) -> dict[str, object]:
    entry = ATLAS_LABELS[int(label_id)]
    record = dict(GROUP_PRESETS[entry.mechanical_group])
    record.update(LABEL_OVERRIDES.get(int(label_id), {}))
    record.update(
        {
            "label_id": int(label_id),
            "tissue": entry.atlas_name,
            "material_id": entry.material_id,
            "mechanical_group": entry.mechanical_group,
            "isotropy_assumption": entry.isotropy_assumption,
        }
    )
    record.setdefault("fiber_stiffness", 0.0)
    record.setdefault("sources", [])
    return record


def mechanical_parameters_for_label(label_id: int) -> dict[str, float | bool]:
    record = mechanical_record(label_id)
    return {
        "young": float(record["young_pa"]),
        "poisson": float(record["poisson"]),
        "fiber_stiffness": float(record.get("fiber_stiffness", 0.0)),
        "fiber_tension_only": True,
    }


def load_physical_materials(path: str | Path) -> dict[str, dict[str, object]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    materials = raw.get("materials")
    if not isinstance(materials, list):
        raise ValueError("physical material JSON must contain a materials list")
    return {str(item["material_id"]): item for item in materials}


def material_table(
    physical_materials: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_id in sorted(ATLAS_LABELS):
        row = mechanical_record(label_id)
        physical = None if physical_materials is None else physical_materials.get(str(row["material_id"]))
        properties = {} if physical is None else dict(physical.get("properties", {}))
        electric = dict(properties.get("electric", {}))
        row.update(
            {
                "mass_density_kg_per_m3": properties.get("mass_density_kg_per_m3"),
                "em_frequency_hz": electric.get("frequency_hz"),
                "conductivity_s_per_m": electric.get("conductivity_s_per_m"),
                "relative_permittivity": electric.get("relative_permittivity"),
                "source_keys": ";".join(str(value) for value in row.pop("sources")),
            }
        )
        rows.append(row)
    return rows


def write_material_table(
    rows: list[dict[str, object]],
    csv_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> None:
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["label_id", "tissue"]
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if json_path is not None:
        output = Path(json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": "satmorph-mechanical-library-v1",
                    "profile": "quasi-static whole-body SAT morphing",
                    "warning": (
                        "Effective research parameters, not patient-specific or clinical. "
                        "Fluid and lumen labels use solid placeholders."
                    ),
                    "sources": SOURCE_REGISTRY,
                    "labels": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
