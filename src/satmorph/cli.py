from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .adaptive_voxel import convert_voxel_mat_adaptive
from .adaptive_growth import (
    coarsen_morph_result,
    literature_cell_materials,
    remesh_morph_result,
    solve_incremental_with_remeshing,
)
from .audit import build_volume_audit, tetra_quality, write_volume_audit
from .calibration import calibrate_target_volume
from .contact import (
    build_label_dynamic_contact,
    build_label_contact_constraints,
    load_contact_constraints,
    save_dynamic_contact,
    save_contact_constraints,
)
from .demo import BONE, SAT, SKIN, SOFT, layered_torso_mesh
from .fiber import save_mesh_with_fibers
from .io import load_mesh, load_result_npz, save_mesh_vtu, save_npz, save_result_bundle
from .material_library import load_physical_materials, material_table, write_material_table
from .mat_convert import convert_mat, describe_arrays, load_mat_arrays
from .metrics import mapped_surface_metrics, sat_thickness_metrics, write_metrics
from .preprocess import repair_surface
from .paper_figure import render_surface_comparison, render_tissue_bundle
from .physical_properties import attach_physical_properties, mass_report, write_json_report
from .solver import Material, SolverOptions, morph_sat, morph_target_region
from .study import summarize_result_jsons, write_summary_csv
from .surface_map import load_surface, map_surface, save_center_result, save_surface_result
from .tissue_groups import (
    DEFAULT_MECHANICAL_PARAMETERS,
    MECHANICAL_GROUP_NAMES,
    mesh_domain_report,
)
from .tissue_surface import extract_tissue_surface_bundle, map_tissue_surface_bundle
from .visual_surface import extract_visual_surface_from_voxel_mat
from .voxel_convert import convert_voxel_mat


def _options(args: argparse.Namespace) -> SolverOptions:
    cap = getattr(args, "bulk_modulus_ratio_cap", 100.0)
    return SolverOptions(
        increments=args.increments,
        max_iterations=args.max_iterations,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        stagnation_tolerance_factor=getattr(
            args, "stagnation_tolerance_factor", 5.0
        ),
        bulk_modulus_ratio_cap=None if cap is not None and cap <= 0.0 else cap,
        verbose=not args.quiet,
    )


def _target_ratio(args: argparse.Namespace) -> float:
    value = args.lambda_sat if args.lambda_sat is not None else args.lambda_target
    if value is not None:
        return float(value) ** 3
    return float(args.target_volume_ratio)


def _load_materials(path: str | None, mesh) -> tuple[Material, dict[int, Material]]:
    default = Material(young=10_000.0, poisson=0.45)
    by_tag: dict[int, Material] = {}
    if path is None:
        return default, by_tag
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "default" in raw:
        default = Material(**raw.pop("default"))
    for key, values in raw.items():
        by_tag[mesh.resolve_tag(key)] = Material(**values)
    return default, by_tag


def _default_cell_materials(mesh) -> np.ndarray | None:
    if "source_label" in mesh.cell_data:
        return literature_cell_materials(mesh)
    group_ids = mesh.cell_data.get("mechanical_group_id")
    if group_ids is None:
        return None
    materials = np.empty(mesh.n_cells, dtype=object)
    fallback = Material(10_000.0, 0.45)
    for group_id, group_name in MECHANICAL_GROUP_NAMES.items():
        params = DEFAULT_MECHANICAL_PARAMETERS.get(group_name)
        material = fallback if params is None else Material(**params)
        materials[np.asarray(group_ids, dtype=np.int64) == int(group_id)] = material
    unset = np.fromiter((value is None for value in materials), dtype=bool, count=len(materials))
    materials[unset] = fallback
    return materials


def _print_outputs(paths, result) -> None:
    print("\nMorphing completed")
    print(f"  target region                     : {result.target_name}")
    print(f"  target unconstrained volume ratio : {result.target_volume_ratio:.6f}")
    print(f"  actual target volume ratio        : {result.actual_volume_ratio:.6f}")
    print(f"  minimum element Jacobian          : {result.j_total.min():.6e}")
    for path in paths:
        print(f"  wrote: {path}")


def command_demo(args: argparse.Namespace) -> None:
    mesh = layered_torso_mesh(args.resolution)
    sat_cells = mesh.cell_tags == SAT
    fixed_nodes = mesh.nodes_for_tags([BONE])
    materials = {
        BONE: Material(young=10_000.0, poisson=0.45),
        SOFT: Material(young=12_000.0, poisson=0.45),
        SAT: Material(young=5_000.0, poisson=0.45),
        SKIN: Material(young=30_000.0, poisson=0.45),
    }
    result = morph_sat(
        mesh,
        sat_cells,
        fixed_nodes,
        _target_ratio(args),
        materials=materials,
        options=_options(args),
    )
    _print_outputs(save_result_bundle(args.output, mesh, result), result)


def command_solve(args: argparse.Namespace) -> None:
    mesh = load_mesh(args.input, args.cell_data)
    bone_tags = [mesh.resolve_tag(tag) for tag in args.bone_tag]
    target_cells, target_name, target_tag = _select_target_cells(mesh, args)
    fixed_nodes = mesh.nodes_for_tags(bone_tags)
    if args.fixed_nodes:
        extra = np.loadtxt(args.fixed_nodes, dtype=np.int64, ndmin=1)
        fixed_nodes = np.unique(np.concatenate((fixed_nodes, extra)))
    default, materials = _load_materials(args.materials, mesh)
    if target_tag is not None and target_tag not in materials:
        materials[target_tag] = Material(args.young_sat, args.poisson_sat)
    cell_materials = None if args.materials else _default_cell_materials(mesh)
    contact = load_contact_constraints(args.contact, mesh) if getattr(args, "contact", None) else None
    result = morph_target_region(
        mesh,
        target_cells,
        fixed_nodes,
        _target_ratio(args),
        materials=materials,
        default_material=default,
        cell_materials=cell_materials,
        options=_options(args),
        target_name=target_name,
        contact=contact,
    )
    _print_outputs(save_result_bundle(args.output, mesh, result), result)


def command_solve_series(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nRunning morphing series")
    for ratio in args.ratio:
        child = argparse.Namespace(**vars(args))
        child.target_volume_ratio = float(ratio)
        child.lambda_sat = None
        child.lambda_target = None
        child.output = str(output_dir / f"{args.prefix}-{_ratio_suffix(float(ratio))}")
        command_solve(child)


def _ratio_suffix(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}"


def _select_target_cells(mesh, args: argparse.Namespace) -> tuple[np.ndarray, str, int | None]:
    if args.sat_tag is not None:
        tag = mesh.resolve_tag(args.sat_tag)
        return mesh.cell_tags == tag, str(args.sat_tag), tag
    if args.target_label:
        if "source_label" not in mesh.cell_data:
            raise ValueError("--target-label requires a mesh converted with source_label cell data")
        labels = np.asarray(args.target_label, dtype=np.int64)
        cells = np.isin(np.asarray(mesh.cell_data["source_label"], dtype=np.int64), labels)
        return cells, "source_label:" + ",".join(str(int(value)) for value in labels), None
    raise ValueError("one of --sat-tag or --target-label is required")


def command_repair(args: argparse.Namespace) -> None:
    repair_surface(args.input, args.output)
    print(f"wrote repaired surface: {args.output}")


def command_map_surface(args: argparse.Namespace) -> None:
    mesh, displacement, _ = load_result_npz(args.coarse_result)
    surface = load_surface(args.surface)
    result = map_surface(
        mesh,
        displacement,
        surface,
        candidate_cells=args.candidate_cells,
        outside_mode=args.outside_mode,
        tolerance=args.tolerance,
        map_centers=not args.no_centers,
    )
    save_surface_result(args.output, surface, result)
    paths = [Path(args.output)]
    if args.centers_output:
        save_center_result(args.centers_output, result)
        paths.append(Path(args.centers_output))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(result.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        paths.append(Path(args.report))
    summary = result.summary()
    print("\nSurface mapping completed")
    print(f"  mapped surface points       : {summary['mapped_points']}")
    print(f"  outside surface points      : {summary['outside_points']}")
    print(f"  maximum point residual      : {summary['maximum_point_residual']:.6e}")
    if "mapped_triangle_centers" in summary:
        print(f"  mapped triangle centers     : {summary['mapped_triangle_centers']}")
        print(f"  outside triangle centers    : {summary['outside_triangle_centers']}")
        print(f"  maximum center residual     : {summary['maximum_center_residual']:.6e}")
    for path in paths:
        print(f"  wrote: {path}")


def command_map_series(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface = load_surface(args.surface)
    result_paths = sorted(input_dir.glob(args.pattern))
    if not result_paths:
        raise ValueError(f"no result files matched {input_dir / args.pattern}")
    print("\nSurface mapping series")
    for coarse_result in result_paths:
        mesh, displacement, _ = load_result_npz(coarse_result)
        mapped = map_surface(
            mesh,
            displacement,
            surface,
            candidate_cells=args.candidate_cells,
            outside_mode=args.outside_mode,
            tolerance=args.tolerance,
            map_centers=not args.no_centers,
        )
        output = output_dir / f"{coarse_result.stem}{args.output_suffix}"
        save_surface_result(output, surface, mapped)
        print(f"  {coarse_result.name} -> {output}")
        if args.report:
            report = output_dir / f"{coarse_result.stem}-map.json"
            report.write_text(
                json.dumps(mapped.summary(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"    report -> {report}")


def command_summarize_series(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise ValueError(f"no JSON files matched {input_dir / args.pattern}")
    rows = summarize_result_jsons(paths)
    write_summary_csv(args.output, rows)
    print("\nSeries summary completed")
    print(f"  cases: {len(rows)}")
    print(f"  wrote: {args.output}")


def command_calibrate(args: argparse.Namespace) -> None:
    mesh = load_mesh(args.input, args.cell_data)
    bone_tags = [mesh.resolve_tag(tag) for tag in args.bone_tag]
    target_cells, target_name, target_tag = _select_target_cells(mesh, args)
    fixed_nodes = mesh.nodes_for_tags(bone_tags)
    if args.fixed_nodes:
        fixed_nodes = np.unique(
            np.concatenate((fixed_nodes, np.loadtxt(args.fixed_nodes, dtype=np.int64, ndmin=1)))
        )
    default, materials = _load_materials(args.materials, mesh)
    if target_tag is not None and target_tag not in materials:
        materials[target_tag] = Material(args.young_sat, args.poisson_sat)
    cell_materials = None if args.materials else _default_cell_materials(mesh)
    contact = load_contact_constraints(args.contact, mesh) if args.contact else None
    result = calibrate_target_volume(
        mesh,
        target_cells,
        fixed_nodes,
        args.desired_volume_ratio,
        materials=materials,
        default_material=default,
        cell_materials=cell_materials,
        options=_options(args),
        target_name=target_name,
        tolerance=args.calibration_tolerance,
        max_corrections=args.max_corrections,
        relaxation=args.calibration_relaxation,
        ratio_bounds=(args.minimum_growth_ratio, args.maximum_growth_ratio),
        max_solver_retries=args.adaptive_increment_retries,
        increment_retry_factor=args.increment_retry_factor,
        contact=contact,
    )
    _print_outputs(save_result_bundle(args.output, mesh, result), result)
    print(f"  desired constrained volume ratio  : {args.desired_volume_ratio:.6f}")
    print(f"  outer calibration solves           : {len(result.calibration_records)}")


def command_volume_audit(args: argparse.Namespace) -> None:
    report = build_volume_audit(
        args.voxel_mat,
        args.mesh,
        result_path=args.result,
        variable=args.variable,
        axis_keys=tuple(args.axis_key),
        axis_unit=args.axis_unit,
        output_unit=args.output_unit,
        voxel_size_mm=args.voxel_size_mm,
    )
    write_volume_audit(report, json_path=args.report, csv_path=args.output)
    summary = report["summary"]
    print("\nVolume audit completed")
    print(f"  mean absolute label error : {summary['mean_absolute_label_volume_error_percent']:.3f}%")
    print(f"  maximum label error       : {summary['maximum_absolute_label_volume_error_percent']:.3f}%")
    print(f"  wrote: {args.output}")
    if args.report:
        print(f"  wrote: {args.report}")


def command_quality_report(args: argparse.Namespace) -> None:
    if args.result:
        mesh, _, deformed = load_result_npz(args.result)
        report = {
            "input": args.result,
            "reference": tetra_quality(mesh),
            "deformed": tetra_quality(mesh, deformed),
        }
    else:
        mesh = load_mesh(args.input, args.cell_data)
        report = {"input": args.input, "reference": tetra_quality(mesh)}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMesh quality report completed")
    print(f"  wrote: {target}")


def command_material_table(args: argparse.Namespace) -> None:
    physical = load_physical_materials(args.physical_materials) if args.physical_materials else None
    rows = material_table(physical)
    write_material_table(rows, args.output, json_path=args.json_output)
    print("\nMaterial table completed")
    print(f"  labels: {len(rows)}")
    print(f"  wrote: {args.output}")
    if args.json_output:
        print(f"  wrote: {args.json_output}")


def command_mesh_domain_table(args: argparse.Namespace) -> None:
    rows = mesh_domain_report()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mesh_domain_tag",
        "mesh_domain",
        "source_labels",
        "mesh_policy",
        "interface_policy",
        "description",
    ]
    if output.suffix.lower() == ".json":
        output.write_text(
            json.dumps(
                {
                    "schema": "satmorph-mesh-domain-policy-v1",
                    "purpose": (
                        "A meshing/interface hierarchy layered above immutable 73-label "
                        "source_label data and below label-specific material assignment."
                    ),
                    "domains": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        with output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                item = dict(row)
                item["source_labels"] = ";".join(str(value) for value in item["source_labels"])
                writer.writerow({field: item[field] for field in fields})
    print("\nMesh-domain policy table completed")
    print(f"  domains: {len(rows)}")
    print(f"  wrote: {output}")


def command_attach_properties(args: argparse.Namespace) -> None:
    mesh = load_mesh(args.input, args.cell_data)
    output, report = attach_physical_properties(mesh, args.materials)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_npz(target, output)
    vtu = Path(args.vtu_output) if args.vtu_output else target.with_suffix(".vtu")
    save_mesh_vtu(vtu, output)
    if args.report:
        write_json_report(report, args.report)
    print("\nPhysical/EM property mapping completed")
    print(f"  source labels: {report['source_labels']}")
    print(f"  wrote: {target}")
    print(f"  wrote: {vtu}")
    if args.report:
        print(f"  wrote: {args.report}")


def command_mass_report(args: argparse.Namespace) -> None:
    path = Path(args.input)
    with np.load(path, allow_pickle=False) as data:
        is_result = "deformed_points" in data.files and "growth_lambda" in data.files
        growth_lambda = np.asarray(data["growth_lambda"], dtype=float) if is_result else None
    if is_result:
        mesh, _, current_points = load_result_npz(path)
    else:
        mesh = load_mesh(path, args.cell_data)
        current_points = None
    if args.materials:
        mesh, _ = attach_physical_properties(mesh, args.materials)
    material_reference = mesh.cell_data.get("material_reference_volume")
    accumulated_growth = mesh.cell_data.get("accumulated_growth_J")
    if accumulated_growth is None:
        accumulated_growth = np.ones(mesh.n_cells, dtype=float)
    else:
        accumulated_growth = np.asarray(accumulated_growth, dtype=float)
    growth_j = accumulated_growth
    if growth_lambda is not None:
        growth_j = growth_j * growth_lambda**3
    report = mass_report(
        mesh,
        current_points=current_points,
        growth_j=growth_j,
        material_reference_volume=material_reference,
        length_unit=args.length_unit,
    )
    write_json_report(report, args.output)
    summary = report["summary"]
    print("\nMass report completed")
    print(f"  initial mass             : {summary['initial_mass_kg']:.6f} kg")
    print(f"  growth-accounted mass    : {summary['growth_accounted_mass_kg']:.6f} kg")
    print(f"  growth mass change       : {summary['growth_mass_change_kg']:.6f} kg")
    print(f"  wrote: {args.output}")


def command_refine_result(args: argparse.Namespace) -> None:
    mesh, _, deformed = load_result_npz(args.input)
    with np.load(args.input, allow_pickle=False) as data:
        growth = np.asarray(data["growth_lambda"], dtype=float)
        j_total = np.asarray(data["j_total"], dtype=float)
        j_elastic = np.asarray(data["j_elastic"], dtype=float)
    refined, report = remesh_morph_result(
        mesh,
        deformed,
        growth,
        j_total,
        j_elastic,
        target_labels=args.target_label or [1],
        max_edges=args.max_edges,
        interface_mode=args.interface_mode,
    )
    if args.physical_materials:
        refined, property_report = attach_physical_properties(refined, args.physical_materials)
        report["physical_property_mapping"] = property_report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output, refined)
    vtu = Path(args.vtu_output) if args.vtu_output else output.with_suffix(".vtu")
    save_mesh_vtu(vtu, refined)
    write_json_report(report, args.report)
    print("\nLabel-preserving result remeshing completed")
    print(f"  selected edges      : {report['selected_edges']}")
    print(f"  tetrahedra before   : {report['tetrahedra_before']}")
    print(f"  tetrahedra after    : {report['tetrahedra_after']}")
    print(f"  labels preserved    : {report['label_set_preserved']}")
    print(f"  wrote: {output}")
    print(f"  wrote: {vtu}")
    print(f"  wrote: {args.report}")


def command_coarsen_result(args: argparse.Namespace) -> None:
    mesh, _, deformed = load_result_npz(args.input)
    with np.load(args.input, allow_pickle=False) as data:
        growth = np.asarray(data["growth_lambda"], dtype=float)
        j_total = np.asarray(data["j_total"], dtype=float)
        j_elastic = np.asarray(data["j_elastic"], dtype=float)
    coarsened, report = coarsen_morph_result(
        mesh,
        deformed,
        growth,
        j_total,
        j_elastic,
        target_labels=args.target_label or [1],
        max_collapses=args.max_collapses,
        max_local_volume_drift=args.max_local_volume_drift,
    )
    if args.physical_materials:
        coarsened, property_report = attach_physical_properties(
            coarsened, args.physical_materials
        )
        report["physical_property_mapping"] = property_report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output, coarsened)
    vtu = Path(args.vtu_output) if args.vtu_output else output.with_suffix(".vtu")
    save_mesh_vtu(vtu, coarsened)
    write_json_report(report, args.report)
    print("\nLabel-preserving result coarsening completed")
    print(f"  selected collapses : {report['selected_collapses']}")
    print(f"  tetrahedra before  : {report['tetrahedra_before']}")
    print(f"  tetrahedra after   : {report['tetrahedra_after']}")
    print(f"  labels preserved   : {report['label_set_preserved']}")
    print(f"  wrote: {output}")
    print(f"  wrote: {vtu}")
    print(f"  wrote: {args.report}")


def command_solve_remesh(args: argparse.Namespace) -> None:
    mesh = load_mesh(args.input, args.cell_data)
    if args.physical_materials:
        mesh, _ = attach_physical_properties(mesh, args.physical_materials)
    bone_tags = [mesh.resolve_tag(value) for value in args.bone_tag]
    summary = solve_incremental_with_remeshing(
        mesh,
        target_labels=args.target_label or [1],
        bone_tags=bone_tags,
        target_growth_volume_ratio=args.target_growth_volume_ratio,
        stages=args.stages,
        max_edges_per_stage=args.max_edges_per_stage,
        interface_mode=args.interface_mode,
        output_dir=args.output_dir,
        options=_options(args),
        max_collapses_per_stage=args.max_collapses_per_stage,
        remesh_mode=args.remesh_mode,
        max_local_volume_drift=args.max_local_volume_drift,
    )
    print("\nIncremental growth/remeshing completed")
    print(f"  tetrahedra before   : {summary['tetrahedra_before']}")
    print(f"  tetrahedra after    : {summary['tetrahedra_after']}")
    print(f"  target cells before : {summary['target_cells_before']}")
    print(f"  target cells after  : {summary['target_cells_after']}")
    print(f"  labels preserved    : {summary['source_labels_preserved']}")
    print(f"  wrote: {summary['summary_path']}")


def command_build_contact(args: argparse.Namespace) -> None:
    mesh = load_mesh(args.input, args.cell_data)
    metadata = {
        "mesh": args.input,
        "slave_labels": args.slave_label,
        "master_labels": args.master_label,
        "search_distance": args.search_distance,
    }
    if args.dynamic:
        constraints = build_label_dynamic_contact(
            mesh,
            args.slave_label,
            args.master_label,
            search_distance=args.search_distance,
            penalty=args.penalty,
            candidates=args.candidates,
        )
        save_dynamic_contact(args.output, constraints, metadata=metadata)
    else:
        constraints = build_label_contact_constraints(
            mesh,
            args.slave_label,
            args.master_label,
            search_distance=args.search_distance,
            penalty=args.penalty,
            candidates=args.candidates,
            max_constraints=args.max_constraints,
        )
        save_contact_constraints(args.output, constraints, metadata=metadata)
    print("\nContact constraints completed")
    print(f"  constraints: {constraints.count}")
    print(f"  wrote: {args.output}")


def command_adaptive_convert(args: argparse.Namespace) -> None:
    report = convert_voxel_mat_adaptive(
        args.input,
        args.output,
        report_path=args.report,
        variable=args.variable,
        axis_keys=tuple(args.axis_key),
        axis_unit=args.axis_unit,
        output_unit=args.output_unit,
        voxel_size_mm=args.voxel_size_mm,
        coarse_stride=args.coarse_stride,
        refine_stride=args.refine_stride,
        fine_stride=args.fine_stride,
        refine_labels=None if args.refine_all_boundaries else args.refine_label,
        preserve_labels=args.preserve_label or (),
        audit_report=args.audit_report,
        volume_error_threshold=args.volume_error_threshold,
        refine_halo_blocks=args.refine_halo_blocks,
        sat_labels=args.sat_label,
        skin_labels=args.skin_label,
        bone_labels=args.bone_label,
        max_points=args.max_points,
        max_tetrahedra=args.max_tetrahedra,
    )
    print("\nAdaptive voxel conversion completed")
    print(f"  points      : {report['mesh']['points']}")
    print(f"  tetrahedra  : {report['mesh']['tetrahedra']}")
    print(f"  wrote: {args.output}")


def command_paper_figure(args: argparse.Namespace) -> None:
    report = render_surface_comparison(
        args.input,
        args.output,
        views=tuple(args.view),
        color_by=args.color_by,
        dpi=args.dpi,
        max_triangles=args.max_triangles,
        include_reference=not args.no_reference,
        labels=args.label,
    )
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nPaper figure completed")
    print(f"  wrote: {args.output}")


def command_build_fibers(args: argparse.Namespace) -> None:
    report = save_mesh_with_fibers(
        args.input,
        args.output,
        longitudinal_axis=args.longitudinal_axis,
    )
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAnatomical fiber field completed")
    print(f"  cells with fibers: {report['cells_with_fibers']}")
    print(f"  wrote: {args.output}")


def command_surface_metrics(args: argparse.Namespace) -> None:
    report = mapped_surface_metrics(
        args.input,
        longitudinal_axis=args.longitudinal_axis,
        slice_count=args.slice_count,
        slice_range=(args.slice_range[0], args.slice_range[1]),
    )
    write_metrics(report, args.output, profile_csv=args.profile_csv)
    print("\nSurface validation metrics completed")
    print(f"  waist change: {report['waist']['change_percent']:.3f}%")
    print(f"  Hausdorff distance: {report['surface_distance']['hausdorff']:.6g}")
    print(f"  wrote: {args.output}")


def command_thickness_metrics(args: argparse.Namespace) -> None:
    report = sat_thickness_metrics(args.outer, args.inner)
    write_metrics(report, args.output)
    print("\nSAT thickness metrics completed")
    print(f"  mean change: {report['mean_change_percent']:.3f}%")
    print(f"  wrote: {args.output}")


def command_extract_tissues(args: argparse.Namespace) -> None:
    report = extract_tissue_surface_bundle(
        args.input,
        args.output_dir,
        include_labels=args.include_label,
        variable=args.variable,
        axis_keys=tuple(args.axis_key),
        axis_unit=args.axis_unit,
        output_unit=args.output_unit,
        voxel_size_mm=args.voxel_size_mm,
        surface_stride=args.surface_stride,
        pre_smooth_sigma=args.pre_smooth_sigma,
        smooth_iterations=args.smooth_iterations,
        suffix=args.suffix,
        method=args.method,
    )
    print("\nTissue surface bundle completed")
    print(f"  tissues: {len(report['surfaces'])}")
    print(f"  wrote: {Path(args.output_dir) / 'tissues.json'}")


def command_map_tissues(args: argparse.Namespace) -> None:
    report = map_tissue_surface_bundle(
        args.coarse_result,
        args.manifest,
        args.output_dir,
        candidate_cells=args.candidate_cells,
        outside_mode=args.outside_mode,
    )
    print("\nTissue surface mapping completed")
    print(f"  tissues: {len(report['surfaces'])}")
    print(f"  wrote: {Path(args.output_dir) / 'tissues-deformed.json'}")


def command_tissue_figure(args: argparse.Namespace) -> None:
    report = render_tissue_bundle(
        args.manifest,
        args.output,
        views=tuple(args.view),
        dpi=args.dpi,
        max_triangles_per_tissue=args.max_triangles_per_tissue,
        include_labels=args.include_label,
    )
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTissue figure completed")
    print(f"  tissues: {report['tissue_count']}")
    print(f"  wrote: {args.output}")


def command_convert_mat(args: argparse.Namespace) -> None:
    arrays = load_mat_arrays(args.input)
    if args.list_variables:
        print(json.dumps(describe_arrays(arrays), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        raise ValueError("--output is required unless --list-variables is used")
    tag_names = None
    if args.tag_names:
        tag_names = json.loads(Path(args.tag_names).read_text(encoding="utf-8"))
    index_base = "one" if args.one_based else "zero" if args.zero_based else "auto"
    report = convert_mat(
        args.input,
        args.output,
        points_key=args.points_key,
        tetra_key=args.tetra_key,
        tags_key=args.tags_key,
        surface_points_key=args.surface_points_key,
        surface_faces_key=args.surface_faces_key,
        surface_output=args.surface_output,
        report_path=args.report,
        index_base=index_base,
        tag_names=tag_names,
    )
    print("\nMAT conversion completed")
    print(f"  mesh points       : {report['mesh']['points']}")
    print(f"  tetrahedra        : {report['mesh']['tetrahedra']}")
    print(f"  wrote: {args.output}")
    if args.surface_output and "surface" in report:
        print(f"  wrote: {args.surface_output}")
    if args.report:
        print(f"  wrote: {args.report}")


def command_convert_voxel_mat(args: argparse.Namespace) -> None:
    report = convert_voxel_mat(
        args.input,
        args.output,
        surface_output=args.surface_output,
        report_path=args.report,
        variable=args.variable,
        axis_keys=tuple(args.axis_key),
        axis_unit=args.axis_unit,
        output_unit=args.output_unit,
        voxel_size_mm=args.voxel_size_mm,
        stride=args.stride,
        surface_stride=args.surface_stride,
        sat_labels=args.sat_label,
        skin_labels=args.skin_label,
        bone_labels=args.bone_label,
        max_tetrahedra=args.max_tetrahedra,
        envelope_fraction=args.envelope_fraction,
        skin_fraction=args.skin_fraction,
    )
    mesh = report["mesh"]
    print("\nVoxel MAT conversion completed")
    print(f"  source grid       : {tuple(report['source_grid_shape'])}")
    print(f"  mesh points       : {mesh['points']}")
    print(f"  tetrahedra        : {mesh['tetrahedra']}")
    print(f"  wrote: {args.output}")
    if args.surface_output:
        surface = report["surface"]
        print(f"  surface triangles : {surface['triangles']}")
        print(f"  wrote: {args.surface_output}")
    if args.report:
        print(f"  wrote: {args.report}")


def command_extract_visual_surface(args: argparse.Namespace) -> None:
    report = extract_visual_surface_from_voxel_mat(
        args.input,
        args.output,
        report_path=args.report,
        variable=args.variable,
        axis_keys=tuple(args.axis_key),
        axis_unit=args.axis_unit,
        output_unit=args.output_unit,
        voxel_size_mm=args.voxel_size_mm,
        surface_stride=args.surface_stride,
        method=args.method,
        pre_smooth_sigma=args.pre_smooth_sigma,
        smooth_method=args.smooth_method,
        smooth_iterations=args.smooth_iterations,
        laplacian_lambda=args.laplacian_lambda,
        taubin_lambda=args.taubin_lambda,
        taubin_mu=args.taubin_mu,
        include_labels=args.include_label,
    )
    surface = report["surface"]
    raw = report["raw_surface"]
    print("\nVisual surface extraction completed")
    print(f"  method            : {report['method']}")
    print(f"  raw points        : {raw['points']}")
    print(f"  raw triangles     : {raw['triangles']}")
    print(f"  smoothed points   : {surface['points']}")
    print(f"  smoothed triangles: {surface['triangles']}")
    print(f"  wrote: {args.output}")
    if args.report:
        print(f"  wrote: {args.report}")


def _add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-volume-ratio",
        type=float,
        help="unconstrained target region volume ratio (lambda^3)",
    )
    target.add_argument(
        "--lambda-sat",
        type=float,
        help="backward-compatible prescribed SAT linear growth factor lambda",
    )
    target.add_argument(
        "--lambda-target",
        type=float,
        help="prescribed isotropic target-region linear growth factor lambda",
    )
    parser.add_argument("--increments", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--bulk-modulus-ratio-cap",
        type=float,
        default=100.0,
        help="cap kappa/mu to reduce linear-tetra locking; use <=0 to disable",
    )
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satmorph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a layered synthetic torso verification")
    _add_solver_arguments(demo)
    demo.add_argument("--resolution", type=int, default=4)
    demo.add_argument("--output", default="demo-sat-morph")
    demo.set_defaults(func=command_demo)

    solve = subparsers.add_parser(
        "solve",
        help="morph SAT or another selected anatomical region in a tetrahedral mesh",
    )
    _add_solver_arguments(solve)
    solve.add_argument("--input", required=True)
    solve.add_argument("--cell-data", default=None)
    selector = solve.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--sat-tag",
        default=None,
        help="target solver-region tag, usually SAT; kept for existing workflows",
    )
    selector.add_argument(
        "--target-label",
        action="append",
        type=int,
        default=None,
        help="target original 73-atlas source label; repeat to combine labels",
    )
    solve.add_argument("--bone-tag", action="append", required=True)
    solve.add_argument("--fixed-nodes", default=None)
    solve.add_argument("--materials", default=None)
    solve.add_argument("--young-sat", type=float, default=5_000.0)
    solve.add_argument("--poisson-sat", type=float, default=0.45)
    solve.add_argument("--contact", default=None, help="optional contact-constraint JSON")
    solve.add_argument("--output", required=True)
    solve.set_defaults(func=command_solve)

    series = subparsers.add_parser(
        "solve-series",
        help="run the same target-region morphing problem for several volume ratios",
    )
    series.add_argument("--input", required=True)
    series.add_argument("--cell-data", default=None)
    series_selector = series.add_mutually_exclusive_group(required=True)
    series_selector.add_argument(
        "--sat-tag",
        default=None,
        help="target solver-region tag, usually SAT; kept for existing workflows",
    )
    series_selector.add_argument(
        "--target-label",
        action="append",
        type=int,
        default=None,
        help="target original 73-atlas source label; repeat to combine labels",
    )
    series.add_argument("--bone-tag", action="append", required=True)
    series.add_argument("--fixed-nodes", default=None)
    series.add_argument("--materials", default=None)
    series.add_argument("--young-sat", type=float, default=5_000.0)
    series.add_argument("--poisson-sat", type=float, default=0.45)
    series.add_argument("--contact", default=None, help="optional contact-constraint JSON")
    series.add_argument("--ratio", action="append", type=float, required=True)
    series.add_argument("--output-dir", required=True)
    series.add_argument("--prefix", default="target")
    series.add_argument("--increments", type=int, default=12)
    series.add_argument("--max-iterations", type=int, default=30)
    series.add_argument("--relative-tolerance", type=float, default=1.0e-7)
    series.add_argument("--absolute-tolerance", type=float, default=1.0e-8)
    series.add_argument("--bulk-modulus-ratio-cap", type=float, default=100.0)
    series.add_argument("--quiet", action="store_true")
    series.set_defaults(func=command_solve_series)

    calibrate = subparsers.add_parser(
        "calibrate-growth",
        help="outer-loop correct growth until the constrained target volume is reached",
    )
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--cell-data", default=None)
    calibrate_selector = calibrate.add_mutually_exclusive_group(required=True)
    calibrate_selector.add_argument("--sat-tag", default=None)
    calibrate_selector.add_argument("--target-label", action="append", type=int, default=None)
    calibrate.add_argument("--bone-tag", action="append", required=True)
    calibrate.add_argument("--fixed-nodes", default=None)
    calibrate.add_argument("--materials", default=None)
    calibrate.add_argument("--young-sat", type=float, default=5_000.0)
    calibrate.add_argument("--poisson-sat", type=float, default=0.45)
    calibrate.add_argument("--contact", default=None)
    calibrate.add_argument("--desired-volume-ratio", type=float, required=True)
    calibrate.add_argument("--calibration-tolerance", type=float, default=2.5e-3)
    calibrate.add_argument("--max-corrections", type=int, default=4)
    calibrate.add_argument("--calibration-relaxation", type=float, default=0.8)
    calibrate.add_argument("--minimum-growth-ratio", type=float, default=0.1)
    calibrate.add_argument("--maximum-growth-ratio", type=float, default=4.0)
    calibrate.add_argument("--increments", type=int, default=12)
    calibrate.add_argument("--max-iterations", type=int, default=30)
    calibrate.add_argument("--relative-tolerance", type=float, default=1.0e-7)
    calibrate.add_argument("--absolute-tolerance", type=float, default=1.0e-8)
    calibrate.add_argument(
        "--stagnation-tolerance-factor",
        type=float,
        default=5.0,
        help=(
            "accept a positive-J state when line search stalls within this "
            "multiple of the nonlinear convergence tolerance"
        ),
    )
    calibrate.add_argument(
        "--adaptive-increment-retries",
        type=int,
        default=3,
        help="retry a failed inner solve with more load increments",
    )
    calibrate.add_argument(
        "--increment-retry-factor",
        type=int,
        default=2,
        help="multiply load increments by this factor after each retry",
    )
    calibrate.add_argument("--bulk-modulus-ratio-cap", type=float, default=100.0)
    calibrate.add_argument("--quiet", action="store_true")
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(func=command_calibrate)

    contact = subparsers.add_parser(
        "build-contact",
        help="build node-to-triangle no-penetration constraints between source labels",
    )
    contact.add_argument("--input", required=True)
    contact.add_argument("--cell-data", default=None)
    contact.add_argument("--slave-label", action="append", type=int, required=True)
    contact.add_argument("--master-label", action="append", type=int, required=True)
    contact.add_argument("--search-distance", type=float, required=True)
    contact.add_argument("--penalty", type=float, default=1.0e5)
    contact.add_argument("--candidates", type=int, default=12)
    contact.add_argument("--max-constraints", type=int, default=100_000)
    contact.add_argument(
        "--dynamic",
        action="store_true",
        help="update closest master face, projection, and normal at every assembly",
    )
    contact.add_argument("--output", required=True)
    contact.set_defaults(func=command_build_contact)

    repair = subparsers.add_parser("repair-surface", help="repair a triangle mesh with PyMeshFix")
    repair.add_argument("--input", required=True)
    repair.add_argument("--output", required=True)
    repair.set_defaults(func=command_repair)

    mapper = subparsers.add_parser(
        "map-surface",
        help="map a coarse tetrahedral displacement result to an independent high-resolution surface",
    )
    mapper.add_argument(
        "--coarse-result",
        required=True,
        help="NPZ bundle produced by 'satmorph solve' or 'satmorph demo'",
    )
    mapper.add_argument("--surface", required=True, help="surface mesh to deform")
    mapper.add_argument("--output", required=True, help="deformed surface mesh or NPZ")
    mapper.add_argument(
        "--centers-output",
        default=None,
        help="optional NPZ file with mapped triangle-center coordinates",
    )
    mapper.add_argument(
        "--report",
        default=None,
        help="optional JSON quality report for the mapping step",
    )
    mapper.add_argument(
        "--candidate-cells",
        type=int,
        default=64,
        help="number of nearby tetrahedra checked for each query point",
    )
    mapper.add_argument(
        "--outside-mode",
        choices=("clamp", "linear", "fail"),
        default="clamp",
        help="how to handle surface points outside the coarse tetrahedral mesh",
    )
    mapper.add_argument("--tolerance", type=float, default=1.0e-8)
    mapper.add_argument(
        "--no-centers",
        action="store_true",
        help="skip triangle-center mapping",
    )
    mapper.set_defaults(func=command_map_surface)

    map_series = subparsers.add_parser(
        "map-series",
        help="map every coarse NPZ result in a directory to the same surface",
    )
    map_series.add_argument("--input-dir", required=True)
    map_series.add_argument("--pattern", default="*.npz")
    map_series.add_argument("--surface", required=True)
    map_series.add_argument("--output-dir", required=True)
    map_series.add_argument("--output-suffix", default="-visual.vtp")
    map_series.add_argument("--report", action="store_true")
    map_series.add_argument("--candidate-cells", type=int, default=64)
    map_series.add_argument(
        "--outside-mode",
        choices=("clamp", "linear", "fail"),
        default="clamp",
    )
    map_series.add_argument("--tolerance", type=float, default=1.0e-8)
    map_series.add_argument("--no-centers", action="store_true")
    map_series.set_defaults(func=command_map_series)

    summarize = subparsers.add_parser(
        "summarize-series",
        help="collect result JSON files into one CSV table",
    )
    summarize.add_argument("--input-dir", required=True)
    summarize.add_argument("--pattern", default="*.json")
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=command_summarize_series)

    audit = subparsers.add_parser(
        "volume-audit",
        help="compare per-label original voxel, reference mesh, and deformed volumes",
    )
    audit.add_argument("--voxel-mat", required=True)
    audit.add_argument("--mesh", required=True)
    audit.add_argument("--result", default=None)
    audit.add_argument("--output", required=True, help="CSV table")
    audit.add_argument("--report", default=None, help="JSON report")
    audit.add_argument("--variable", default="MaterialLabelGrid")
    audit.add_argument("--axis-key", action="append", default=None)
    audit.add_argument("--axis-unit", choices=("m", "mm"), default="m")
    audit.add_argument("--output-unit", choices=("m", "mm"), default="m")
    audit.add_argument("--voxel-size-mm", type=float, default=1.0)
    audit.set_defaults(func=command_volume_audit)

    quality = subparsers.add_parser(
        "quality-report",
        help="report tetra mean-ratio quality and signed-volume checks",
    )
    quality_source = quality.add_mutually_exclusive_group(required=True)
    quality_source.add_argument("--input", default=None)
    quality_source.add_argument("--result", default=None)
    quality.add_argument("--cell-data", default=None)
    quality.add_argument("--output", required=True)
    quality.set_defaults(func=command_quality_report)

    material_table_parser = subparsers.add_parser(
        "material-table",
        help="write the literature-informed 73-label mechanical and physical property table",
    )
    material_table_parser.add_argument("--physical-materials", default=None)
    material_table_parser.add_argument("--output", required=True, help="CSV table")
    material_table_parser.add_argument("--json-output", default=None)
    material_table_parser.set_defaults(func=command_material_table)

    domain_table_parser = subparsers.add_parser(
        "mesh-domain-table",
        help="write the 19-domain meshing/interface policy layered over the 73 source labels",
    )
    domain_table_parser.add_argument("--output", required=True, help="CSV or JSON table")
    domain_table_parser.set_defaults(func=command_mesh_domain_table)

    attach = subparsers.add_parser(
        "attach-physical-properties",
        help="attach density and EM fields to every tetrahedron using immutable source_label",
    )
    attach.add_argument("--input", required=True)
    attach.add_argument("--cell-data", default=None)
    attach.add_argument("--materials", required=True, help="s4l_materials_unified.json")
    attach.add_argument("--output", required=True, help="output NPZ mesh")
    attach.add_argument("--vtu-output", default=None)
    attach.add_argument("--report", default=None)
    attach.set_defaults(func=command_attach_properties)

    mass = subparsers.add_parser(
        "mass-report",
        help="compute initial, deformed-geometric, and growth-accounted body mass",
    )
    mass.add_argument("--input", required=True, help="mesh or solve-result NPZ")
    mass.add_argument("--cell-data", default=None)
    mass.add_argument("--materials", default=None, help="optional s4l_materials_unified.json")
    mass.add_argument("--length-unit", choices=("m", "mm"), default="m")
    mass.add_argument("--output", required=True)
    mass.set_defaults(func=command_mass_report)

    refine_result = subparsers.add_parser(
        "refine-result",
        help="conformingly add tetrahedra to a deformed target tissue without changing labels",
    )
    refine_result.add_argument("--input", required=True, help="solve-result NPZ")
    refine_result.add_argument("--target-label", action="append", type=int, default=None)
    refine_result.add_argument("--max-edges", type=int, default=1_000)
    refine_result.add_argument(
        "--interface-mode", choices=("propagate", "interior-only"), default="propagate"
    )
    refine_result.add_argument("--physical-materials", default=None)
    refine_result.add_argument("--output", required=True, help="remeshed NPZ")
    refine_result.add_argument("--vtu-output", default=None)
    refine_result.add_argument("--report", required=True)
    refine_result.set_defaults(func=command_refine_result)

    coarsen_result = subparsers.add_parser(
        "coarsen-result",
        help="safely collapse target-tissue edges without changing source labels",
    )
    coarsen_result.add_argument("--input", required=True, help="solve-result NPZ")
    coarsen_result.add_argument("--target-label", action="append", type=int, default=None)
    coarsen_result.add_argument("--max-collapses", type=int, default=500)
    coarsen_result.add_argument("--max-local-volume-drift", type=float, default=0.01)
    coarsen_result.add_argument("--physical-materials", default=None)
    coarsen_result.add_argument("--output", required=True, help="coarsened NPZ")
    coarsen_result.add_argument("--vtu-output", default=None)
    coarsen_result.add_argument("--report", required=True)
    coarsen_result.set_defaults(func=command_coarsen_result)

    solve_remesh = subparsers.add_parser(
        "solve-remesh",
        help="incrementally grow/shrink SAT with label-preserving remeshing after each stage",
    )
    solve_remesh.add_argument("--input", required=True)
    solve_remesh.add_argument("--cell-data", default=None)
    solve_remesh.add_argument("--target-label", action="append", type=int, default=None)
    solve_remesh.add_argument("--bone-tag", action="append", required=True)
    solve_remesh.add_argument("--target-growth-volume-ratio", type=float, required=True)
    solve_remesh.add_argument("--stages", type=int, default=3)
    solve_remesh.add_argument("--max-edges-per-stage", type=int, default=1_000)
    solve_remesh.add_argument("--max-collapses-per-stage", type=int, default=500)
    solve_remesh.add_argument(
        "--remesh-mode",
        choices=("auto", "refine", "coarsen", "none"),
        default="auto",
    )
    solve_remesh.add_argument("--max-local-volume-drift", type=float, default=0.01)
    solve_remesh.add_argument(
        "--interface-mode", choices=("propagate", "interior-only"), default="propagate"
    )
    solve_remesh.add_argument("--physical-materials", default=None)
    solve_remesh.add_argument("--output-dir", required=True)
    solve_remesh.add_argument("--increments", type=int, default=6)
    solve_remesh.add_argument("--max-iterations", type=int, default=30)
    solve_remesh.add_argument("--relative-tolerance", type=float, default=1.0e-7)
    solve_remesh.add_argument("--absolute-tolerance", type=float, default=1.0e-8)
    solve_remesh.add_argument("--bulk-modulus-ratio-cap", type=float, default=100.0)
    solve_remesh.add_argument("--quiet", action="store_true")
    solve_remesh.set_defaults(func=command_solve_remesh)

    fibers = subparsers.add_parser(
        "build-fiber-field",
        help="add approximate skin/muscle/tendon fiber directions to a labeled mesh",
    )
    fibers.add_argument("--input", required=True)
    fibers.add_argument("--output", required=True)
    fibers.add_argument("--report", default=None)
    fibers.add_argument("--longitudinal-axis", type=int, choices=(0, 1, 2), default=2)
    fibers.set_defaults(func=command_build_fibers)

    metrics = subparsers.add_parser(
        "surface-metrics",
        help="measure area, enclosed volume, waist profile, displacement, and surface distances",
    )
    metrics.add_argument("--input", required=True, help="mapped surface NPZ")
    metrics.add_argument("--output", required=True)
    metrics.add_argument("--profile-csv", default=None)
    metrics.add_argument("--longitudinal-axis", type=int, choices=(0, 1, 2), default=2)
    metrics.add_argument("--slice-count", type=int, default=31)
    metrics.add_argument("--slice-range", type=float, nargs=2, default=(0.35, 0.65))
    metrics.set_defaults(func=command_surface_metrics)

    thickness = subparsers.add_parser(
        "sat-thickness",
        help="compare nearest-surface SAT thickness between mapped outer and inner surfaces",
    )
    thickness.add_argument("--outer", required=True)
    thickness.add_argument("--inner", required=True)
    thickness.add_argument("--output", required=True)
    thickness.set_defaults(func=command_thickness_metrics)

    tissues = subparsers.add_parser(
        "extract-tissue-surfaces",
        help="extract independent labeled tissue surfaces and a ParaView VTM collection",
    )
    tissues.add_argument("--input", required=True)
    tissues.add_argument("--output-dir", required=True)
    tissues.add_argument("--include-label", action="append", type=int, default=None)
    tissues.add_argument("--variable", default="MaterialLabelGrid")
    tissues.add_argument("--axis-key", action="append", default=None)
    tissues.add_argument("--axis-unit", choices=("m", "mm"), default="m")
    tissues.add_argument("--output-unit", choices=("m", "mm"), default="m")
    tissues.add_argument("--voxel-size-mm", type=float, default=1.0)
    tissues.add_argument("--surface-stride", type=int, default=2)
    tissues.add_argument("--method", choices=("marching-cubes", "blocks"), default="marching-cubes")
    tissues.add_argument("--pre-smooth-sigma", type=float, default=0.0)
    tissues.add_argument("--smooth-iterations", type=int, default=15)
    tissues.add_argument("--suffix", choices=(".vtp", ".npz"), default=".vtp")
    tissues.set_defaults(func=command_extract_tissues)

    map_tissues = subparsers.add_parser(
        "map-tissue-surfaces",
        help="map one FEM result to every surface in a tissue bundle",
    )
    map_tissues.add_argument("--coarse-result", required=True)
    map_tissues.add_argument("--manifest", required=True)
    map_tissues.add_argument("--output-dir", required=True)
    map_tissues.add_argument("--candidate-cells", type=int, default=64)
    map_tissues.add_argument("--outside-mode", choices=("clamp", "linear", "fail"), default="clamp")
    map_tissues.set_defaults(func=command_map_tissues)

    tissue_figure = subparsers.add_parser(
        "tissue-figure",
        help="render aligned publication-style views of a tissue surface bundle",
    )
    tissue_figure.add_argument("--manifest", required=True)
    tissue_figure.add_argument("--output", required=True)
    tissue_figure.add_argument("--report", default=None)
    tissue_figure.add_argument(
        "--view", action="append", choices=("front", "side", "back", "oblique"), default=None
    )
    tissue_figure.add_argument("--dpi", type=int, default=300)
    tissue_figure.add_argument("--max-triangles-per-tissue", type=int, default=20_000)
    tissue_figure.add_argument("--include-label", action="append", type=int, default=None)
    tissue_figure.set_defaults(func=command_tissue_figure)

    converter = subparsers.add_parser(
        "convert-mat", help="convert MATLAB mesh arrays to satmorph NPZ inputs"
    )
    converter.add_argument("--input", required=True)
    converter.add_argument("--output", default=None)
    converter.add_argument("--surface-output", default=None)
    converter.add_argument("--report", default=None)
    converter.add_argument("--list-variables", action="store_true")
    converter.add_argument("--points-key", default=None)
    converter.add_argument("--tetra-key", default=None)
    converter.add_argument("--tags-key", default=None)
    converter.add_argument("--surface-points-key", default=None)
    converter.add_argument("--surface-faces-key", default=None)
    converter.add_argument("--tag-names", default=None)
    base = converter.add_mutually_exclusive_group()
    base.add_argument("--one-based", action="store_true")
    base.add_argument("--zero-based", action="store_true")
    converter.set_defaults(func=command_convert_mat)

    voxel = subparsers.add_parser(
        "convert-voxel-mat",
        help="convert a labeled MATLAB voxel atlas to a coarse tetra mesh and body surface",
    )
    voxel.add_argument("--input", required=True)
    voxel.add_argument("--output", required=True)
    voxel.add_argument("--surface-output", default=None)
    voxel.add_argument("--report", default=None)
    voxel.add_argument("--variable", default="MaterialLabelGrid")
    voxel.add_argument(
        "--axis-key", action="append", default=None,
        help="MAT axis variable; provide exactly three times (default: Axis0/Axis1/Axis2)",
    )
    voxel.add_argument("--axis-unit", choices=("m", "mm"), default="m")
    voxel.add_argument("--output-unit", choices=("m", "mm"), default="m")
    voxel.add_argument("--voxel-size-mm", type=float, default=1.0)
    voxel.add_argument("--stride", type=int, default=20)
    voxel.add_argument("--surface-stride", type=int, default=4)
    voxel.add_argument("--max-tetrahedra", type=int, default=500_000)
    voxel.add_argument("--envelope-fraction", type=float, default=0.25)
    voxel.add_argument("--skin-fraction", type=float, default=0.25)
    voxel.add_argument("--sat-label", action="append", type=int, default=None)
    voxel.add_argument("--skin-label", action="append", type=int, default=None)
    voxel.add_argument("--bone-label", action="append", type=int, default=None)
    voxel.set_defaults(func=command_convert_voxel_mat)

    adaptive = subparsers.add_parser(
        "convert-voxel-mat-adaptive",
        help="boundary-aware adaptive Delaunay conversion of a labeled voxel atlas",
    )
    adaptive.add_argument("--input", required=True)
    adaptive.add_argument("--output", required=True)
    adaptive.add_argument("--report", default=None)
    adaptive.add_argument("--variable", default="MaterialLabelGrid")
    adaptive.add_argument("--axis-key", action="append", default=None)
    adaptive.add_argument("--axis-unit", choices=("m", "mm"), default="m")
    adaptive.add_argument("--output-unit", choices=("m", "mm"), default="m")
    adaptive.add_argument("--voxel-size-mm", type=float, default=1.0)
    adaptive.add_argument("--coarse-stride", type=int, default=20)
    adaptive.add_argument("--refine-stride", type=int, default=5)
    adaptive.add_argument("--fine-stride", type=int, default=None)
    adaptive.add_argument("--refine-label", action="append", type=int, default=None)
    adaptive.add_argument(
        "--refine-all-boundaries",
        action="store_true",
        help="refine every mixed tissue/body block instead of selected labels only",
    )
    adaptive.add_argument("--refine-halo-blocks", type=int, default=1)
    adaptive.add_argument(
        "--audit-report",
        action="append",
        default=None,
        help="previous volume-audit JSON used to select missing/high-error labels",
    )
    adaptive.add_argument("--volume-error-threshold", type=float, default=5.0)
    adaptive.add_argument("--preserve-label", action="append", type=int, default=None)
    adaptive.add_argument("--sat-label", action="append", type=int, default=None)
    adaptive.add_argument("--skin-label", action="append", type=int, default=None)
    adaptive.add_argument("--bone-label", action="append", type=int, default=None)
    adaptive.add_argument("--max-points", type=int, default=250_000)
    adaptive.add_argument("--max-tetrahedra", type=int, default=1_500_000)
    adaptive.set_defaults(func=command_adaptive_convert)

    visual = subparsers.add_parser(
        "extract-visual-surface",
        help="extract a smoother visualization surface from a labeled voxel MAT atlas",
    )
    visual.add_argument("--input", required=True)
    visual.add_argument("--output", required=True)
    visual.add_argument("--report", default=None)
    visual.add_argument("--variable", default="MaterialLabelGrid")
    visual.add_argument(
        "--axis-key",
        action="append",
        default=None,
        help="MAT axis variable; provide exactly three times (default: Axis0/Axis1/Axis2)",
    )
    visual.add_argument("--axis-unit", choices=("m", "mm"), default="m")
    visual.add_argument("--output-unit", choices=("m", "mm"), default="m")
    visual.add_argument("--voxel-size-mm", type=float, default=1.0)
    visual.add_argument("--surface-stride", type=int, default=2)
    visual.add_argument(
        "--method",
        choices=("marching-cubes", "blocks"),
        default="marching-cubes",
        help="surface extraction method; blocks needs no scikit-image but remains blockier",
    )
    visual.add_argument(
        "--pre-smooth-sigma",
        type=float,
        default=0.0,
        help="optional Gaussian smoothing of the occupancy field before marching cubes",
    )
    visual.add_argument(
        "--smooth-method",
        choices=("none", "laplacian", "taubin"),
        default="taubin",
    )
    visual.add_argument("--smooth-iterations", type=int, default=20)
    visual.add_argument("--laplacian-lambda", type=float, default=0.35)
    visual.add_argument("--taubin-lambda", type=float, default=0.5)
    visual.add_argument("--taubin-mu", type=float, default=-0.53)
    visual.add_argument(
        "--include-label",
        action="append",
        type=int,
        default=None,
        help="extract only the selected source label; repeat for multiple labels",
    )
    visual.set_defaults(func=command_extract_visual_surface)

    figure = subparsers.add_parser(
        "paper-figure",
        help="render aligned publication-style views from mapped surface NPZ cases",
    )
    figure.add_argument("--input", action="append", required=True)
    figure.add_argument("--output", required=True)
    figure.add_argument("--report", default=None)
    figure.add_argument("--label", action="append", default=None)
    figure.add_argument(
        "--view",
        action="append",
        choices=("front", "side", "back", "oblique"),
        default=None,
    )
    figure.add_argument("--color-by", default="displacement")
    figure.add_argument("--dpi", type=int, default=300)
    figure.add_argument("--max-triangles", type=int, default=45_000)
    figure.add_argument("--no-reference", action="store_true")
    figure.set_defaults(func=command_paper_figure)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "convert-voxel-mat":
        args.axis_key = args.axis_key or ["Axis0", "Axis1", "Axis2"]
        if len(args.axis_key) != 3:
            raise ValueError("--axis-key must be provided exactly three times")
        args.sat_label = args.sat_label or [1]
        args.skin_label = args.skin_label or [2]
        args.bone_label = args.bone_label or [19, 68, 69]
    if args.command == "convert-voxel-mat-adaptive":
        args.axis_key = args.axis_key or ["Axis0", "Axis1", "Axis2"]
        if len(args.axis_key) != 3:
            raise ValueError("--axis-key must be provided exactly three times")
        if not args.refine_all_boundaries:
            args.refine_label = args.refine_label or [1, 2]
        args.sat_label = args.sat_label or [1]
        args.skin_label = args.skin_label or [2]
        args.bone_label = args.bone_label or [19, 68, 69]
    if args.command == "volume-audit":
        args.axis_key = args.axis_key or ["Axis0", "Axis1", "Axis2"]
        if len(args.axis_key) != 3:
            raise ValueError("--axis-key must be provided exactly three times")
    if args.command == "extract-visual-surface":
        args.axis_key = args.axis_key or ["Axis0", "Axis1", "Axis2"]
        if len(args.axis_key) != 3:
            raise ValueError("--axis-key must be provided exactly three times")
    if args.command == "extract-tissue-surfaces":
        args.axis_key = args.axis_key or ["Axis0", "Axis1", "Axis2"]
        if len(args.axis_key) != 3:
            raise ValueError("--axis-key must be provided exactly three times")
    if args.command == "paper-figure":
        args.view = args.view or ["front", "side", "oblique"]
    if args.command == "tissue-figure":
        args.view = args.view or ["front", "side", "oblique"]
    args.func(args)


if __name__ == "__main__":
    main()
