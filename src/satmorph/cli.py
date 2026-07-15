from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .demo import BONE, SAT, SKIN, SOFT, layered_torso_mesh
from .io import load_mesh, load_result_npz, save_result_bundle
from .mat_convert import convert_mat, describe_arrays, load_mat_arrays
from .preprocess import repair_surface
from .solver import Material, SolverOptions, morph_sat
from .surface_map import load_surface, map_surface, save_center_result, save_surface_result
from .voxel_convert import convert_voxel_mat


def _options(args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(
        increments=args.increments,
        max_iterations=args.max_iterations,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        verbose=not args.quiet,
    )


def _target_ratio(args: argparse.Namespace) -> float:
    if args.lambda_sat is not None:
        return float(args.lambda_sat) ** 3
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


def _print_outputs(paths, result) -> None:
    print("\nSAT morphing completed")
    print(f"  target unconstrained volume ratio : {result.target_volume_ratio:.6f}")
    print(f"  actual SAT volume ratio           : {result.actual_volume_ratio:.6f}")
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
    sat_tag = mesh.resolve_tag(args.sat_tag)
    bone_tags = [mesh.resolve_tag(tag) for tag in args.bone_tag]
    sat_cells = mesh.cell_tags == sat_tag
    fixed_nodes = mesh.nodes_for_tags(bone_tags)
    if args.fixed_nodes:
        extra = np.loadtxt(args.fixed_nodes, dtype=np.int64, ndmin=1)
        fixed_nodes = np.unique(np.concatenate((fixed_nodes, extra)))
    default, materials = _load_materials(args.materials, mesh)
    if sat_tag not in materials:
        materials[sat_tag] = Material(args.young_sat, args.poisson_sat)
    result = morph_sat(
        mesh,
        sat_cells,
        fixed_nodes,
        _target_ratio(args),
        materials=materials,
        default_material=default,
        options=_options(args),
    )
    _print_outputs(save_result_bundle(args.output, mesh, result), result)


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


def _add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-volume-ratio",
        type=float,
        help="unconstrained target SAT volume ratio (lambda^3)",
    )
    target.add_argument(
        "--lambda-sat",
        type=float,
        help="prescribed isotropic linear growth factor lambda",
    )
    parser.add_argument("--increments", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satmorph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a layered synthetic torso verification")
    _add_solver_arguments(demo)
    demo.add_argument("--resolution", type=int, default=4)
    demo.add_argument("--output", default="demo-sat-morph")
    demo.set_defaults(func=command_demo)

    solve = subparsers.add_parser("solve", help="morph SAT in a tetrahedral anatomical mesh")
    _add_solver_arguments(solve)
    solve.add_argument("--input", required=True)
    solve.add_argument("--cell-data", default=None)
    solve.add_argument("--sat-tag", required=True)
    solve.add_argument("--bone-tag", action="append", required=True)
    solve.add_argument("--fixed-nodes", default=None)
    solve.add_argument("--materials", default=None)
    solve.add_argument("--young-sat", type=float, default=5_000.0)
    solve.add_argument("--poisson-sat", type=float, default=0.45)
    solve.add_argument("--output", required=True)
    solve.set_defaults(func=command_solve)

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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "convert-voxel-mat":
        args.axis_key = args.axis_key or ["Axis0", "Axis1", "Axis2"]
        if len(args.axis_key) != 3:
            raise ValueError("--axis-key must be provided exactly three times")
        args.sat_label = args.sat_label or [1]
        args.skin_label = args.skin_label or [2]
        args.bone_label = args.bone_label or [19, 68, 69, 70]
    args.func(args)


if __name__ == "__main__":
    main()
