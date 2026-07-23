import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from satmorph.demo import BONE, SAT, SKIN, SOFT, layered_torso_mesh
from satmorph.adaptive_growth import solve_incremental_with_remeshing
from satmorph.cli import command_mass_report
from satmorph.io import save_mesh_vtu, save_npz
from satmorph.material_library import SOURCE_REGISTRY, material_table, mechanical_record
from satmorph.mesh import TetMesh
from satmorph.physical_properties import attach_physical_properties, mass_report
from satmorph.remesh import coarsen_marked_tetrahedra, refine_marked_tetrahedra
from satmorph.solver import MorphResult, SolverOptions


def labeled_demo() -> TetMesh:
    base = layered_torso_mesh(3, 2)
    labels = np.full(base.n_cells, 35, dtype=np.int16)
    labels[base.cell_tags == SAT] = 1
    labels[base.cell_tags == SKIN] = 2
    labels[base.cell_tags == BONE] = 68
    return TetMesh(
        base.points,
        base.tetra,
        base.cell_tags,
        base.tag_names,
        {
            "source_label": labels,
            "material_id": labels.copy(),
            "mechanical_group_id": np.zeros(base.n_cells, dtype=np.int16),
        },
    )


class MaterialRemeshMassTests(unittest.TestCase):
    def test_material_table_covers_all_73_labels_and_sources(self):
        rows = material_table()
        self.assertEqual(len(rows), 73)
        self.assertEqual({int(row["label_id"]) for row in rows}, set(range(1, 74)))
        for label in range(1, 74):
            record = mechanical_record(label)
            self.assertGreater(float(record["young_pa"]), 0.0)
            self.assertLess(float(record["poisson"]), 0.5)
            for source in record["sources"]:
                self.assertIn(source, SOURCE_REGISTRY)

    def test_edge_star_refinement_preserves_labels_and_label_volumes(self):
        mesh = labeled_demo()
        marked = mesh.cell_data["source_label"] == 1
        before_volume = {
            int(label): float(mesh.cell_volumes()[mesh.cell_data["source_label"] == label].sum())
            for label in np.unique(mesh.cell_data["source_label"])
        }
        refined = refine_marked_tetrahedra(mesh, marked, max_edges=5)
        self.assertGreater(refined.mesh.n_cells, mesh.n_cells)
        self.assertGreater(
            np.count_nonzero(refined.mesh.cell_data["source_label"] == 1),
            np.count_nonzero(marked),
        )
        np.testing.assert_array_equal(
            refined.mesh.cell_data["source_label"],
            mesh.cell_data["source_label"][refined.parent_cells],
        )
        self.assertTrue(refined.report["label_set_preserved"])
        self.assertLess(refined.report["maximum_label_volume_drift_relative"], 1.0e-12)
        for label, volume in before_volume.items():
            mask = refined.mesh.cell_data["source_label"] == label
            self.assertAlmostEqual(float(refined.mesh.cell_volumes()[mask].sum()), volume, places=12)

    def test_extensive_reference_volume_is_conserved(self):
        mesh = labeled_demo()
        marked = mesh.cell_data["source_label"] == 1
        reference = mesh.cell_volumes()
        refined = refine_marked_tetrahedra(
            mesh,
            marked,
            max_edges=4,
            extensive_parent_cell_data={"material_reference_volume": reference},
        )
        self.assertAlmostEqual(
            float(refined.mesh.cell_data["material_reference_volume"].sum()),
            float(reference.sum()),
            places=12,
        )

    def test_label_safe_coarsening_reduces_cells_and_conserves_material(self):
        mesh = labeled_demo()
        marked = mesh.cell_data["source_label"] == 1
        reference = mesh.cell_volumes()
        history = np.repeat(
            np.diag([1.03, 0.98, 1.01])[None, :, :], mesh.n_cells, axis=0
        )
        coarsened = coarsen_marked_tetrahedra(
            mesh,
            marked,
            max_collapses=4,
            parent_cell_data={"elastic_history_F": history},
            extensive_parent_cell_data={"material_reference_volume": reference},
            project_parent_cell_data=("elastic_history_F",),
        )
        self.assertEqual(coarsened.report["selected_collapses"], 4)
        self.assertLess(coarsened.mesh.n_points, mesh.n_points)
        self.assertLess(coarsened.mesh.n_cells, mesh.n_cells)
        self.assertTrue(coarsened.report["label_set_preserved"])
        np.testing.assert_array_equal(
            np.unique(coarsened.mesh.cell_data["source_label"]),
            np.unique(mesh.cell_data["source_label"]),
        )
        self.assertAlmostEqual(
            float(coarsened.mesh.cell_data["material_reference_volume"].sum()),
            float(reference.sum()),
            places=12,
        )
        self.assertGreater(
            float(np.linalg.det(coarsened.mesh.cell_data["elastic_history_F"]).min()),
            0.0,
        )

    def test_physical_mapping_and_growth_mass(self):
        mesh = labeled_demo()
        records = []
        densities = {1: 900.0, 2: 1100.0, 35: 1050.0, 68: 1200.0}
        from satmorph.tissue_groups import ATLAS_LABELS

        for label, density in densities.items():
            entry = ATLAS_LABELS[label]
            records.append(
                {
                    "material_id": entry.material_id,
                    "database_catalog": "test",
                    "database_name": entry.atlas_name,
                    "properties": {
                        "mass_density_kg_per_m3": density,
                        "electric": {
                            "frequency_hz": 316.0,
                            "conductivity_s_per_m": 0.1 + label * 1.0e-3,
                            "relative_permittivity": 100.0 + label,
                        },
                    },
                }
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "materials.json"
            path.write_text(json.dumps({"materials": records}), encoding="utf-8")
            mapped, report = attach_physical_properties(mesh, path)
        self.assertEqual(
            report["source_labels"], len(np.unique(mesh.cell_data["source_label"]))
        )
        self.assertTrue(np.all(np.isfinite(mapped.cell_data["mass_density_kg_per_m3"])))
        growth = np.ones(mesh.n_cells)
        growth[mesh.cell_data["source_label"] == 1] = 1.2
        result = mass_report(mapped, growth_j=growth)
        sat = next(row for row in result["labels"] if row["label_id"] == 1)
        self.assertAlmostEqual(
            sat["growth_accounted_mass_kg"] / sat["initial_mass_kg"], 1.2, places=12
        )

    def test_mass_report_combines_previous_and_current_growth(self):
        mesh = labeled_demo()
        mesh.cell_data["mass_density_kg_per_m3"] = np.full(mesh.n_cells, 1000.0)
        mesh.cell_data["accumulated_growth_J"] = np.full(mesh.n_cells, 1.2)
        current_growth_j = 1.1
        result = MorphResult(
            points=mesh.points.copy(),
            displacement=np.zeros_like(mesh.points),
            growth_lambda=np.full(mesh.n_cells, current_growth_j ** (1.0 / 3.0)),
            j_total=np.ones(mesh.n_cells),
            j_elastic=np.ones(mesh.n_cells),
            target_reference_volume=1.0,
            target_current_volume=1.0,
            target_volume_ratio=1.0,
        )
        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "stage-result.npz"
            report_path = Path(directory) / "mass.json"
            save_npz(result_path, mesh, result)
            command_mass_report(
                Namespace(
                    input=str(result_path),
                    cell_data=None,
                    materials=None,
                    length_unit="m",
                    output=str(report_path),
                )
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            report["summary"]["growth_accounted_mass_kg"]
            / report["summary"]["initial_mass_kg"],
            1.2 * current_growth_j,
            places=12,
        )

    def test_elastic_history_vtu_has_nine_components(self):
        mesh = labeled_demo()
        history = np.repeat(np.eye(3)[None, :, :], mesh.n_cells, axis=0)
        mesh.cell_data["elastic_history_F"] = history
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.vtu"
            save_mesh_vtu(path, mesh)
            text = path.read_text(encoding="utf-8")
        self.assertIn('Name="elastic_history_F" NumberOfComponents="9"', text)

    def test_two_stage_shrink_coarsens_and_continues_with_history(self):
        mesh = labeled_demo()
        with TemporaryDirectory() as directory:
            summary = solve_incremental_with_remeshing(
                mesh,
                target_labels=[1],
                bone_tags=[BONE],
                target_growth_volume_ratio=0.99,
                stages=2,
                max_edges_per_stage=0,
                interface_mode="propagate",
                output_dir=directory,
                options=SolverOptions(
                    increments=2,
                    max_iterations=30,
                    relative_tolerance=1.0e-7,
                    absolute_tolerance=1.0e-8,
                    verbose=False,
                ),
                max_collapses_per_stage=1,
                remesh_mode="auto",
                max_local_volume_drift=0.02,
            )
        self.assertLess(summary["tetrahedra_after"], summary["tetrahedra_before"])
        self.assertLess(summary["target_cells_after"], summary["target_cells_before"])
        self.assertTrue(summary["source_labels_preserved"])
        self.assertEqual(
            [stage["remesh_mode"] for stage in summary["stages_detail"]],
            ["coarsen", "coarsen"],
        )
        self.assertTrue(
            all(stage["selected_operations"] == 1 for stage in summary["stages_detail"])
        )


if __name__ == "__main__":
    unittest.main()
