from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

from .mesh import TetMesh
from .solver import MorphResult


def load_mesh(path: str | Path, cell_data_name: str | None = None) -> TetMesh:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            points = data["points"]
            tetra = data["tetra"]
            tags = data["cell_tags"]
            raw_names = data.get("tag_names_json")
            names = json.loads(str(raw_names.item())) if raw_names is not None else {}
            raw_cell_data_names = data.get("cell_data_names_json")
            cell_data_names = (
                json.loads(str(raw_cell_data_names.item()))
                if raw_cell_data_names is not None
                else []
            )
            cell_data = {
                str(name): data[f"cell_data__{name}"]
                for name in cell_data_names
                if f"cell_data__{name}" in data
            }
        return TetMesh(points, tetra, tags, names, cell_data)

    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio is required to read non-NPZ meshes") from exc

    source = meshio.read(path)
    tetra_blocks: list[np.ndarray] = []
    tag_blocks: list[np.ndarray] = []
    candidate_names = [cell_data_name] if cell_data_name else [
        "gmsh:physical",
        "region",
        "cell_tags",
        "material",
    ]
    chosen = next((name for name in candidate_names if name and name in source.cell_data), None)

    for block_index, block in enumerate(source.cells):
        if block.type not in {"tetra", "tetra10"}:
            continue
        tetra_blocks.append(np.asarray(block.data[:, :4], dtype=np.int64))
        if chosen is None:
            tag_blocks.append(np.zeros(len(block.data), dtype=np.int64))
        else:
            tag_blocks.append(np.asarray(source.cell_data[chosen][block_index], dtype=np.int64))
    if not tetra_blocks:
        raise ValueError(f"no tetrahedral cells found in {path}")

    names: dict[str, int] = {}
    for name, value in source.field_data.items():
        raw = np.asarray(value).ravel()
        if len(raw) >= 2 and int(raw[1]) == 3:
            names[str(name)] = int(raw[0])
    extra_cell_data: dict[str, np.ndarray] = {}
    for data_name, blocks in source.cell_data.items():
        if data_name == chosen:
            continue
        selected_blocks: list[np.ndarray] = []
        for block_index, block in enumerate(source.cells):
            if block.type in {"tetra", "tetra10"} and block_index < len(blocks):
                values = np.asarray(blocks[block_index])
                if len(values) == len(block.data):
                    selected_blocks.append(values)
        if selected_blocks:
            combined = np.concatenate(selected_blocks)
            if str(data_name) == "elastic_history_F" and combined.shape[1:] == (9,):
                combined = combined.reshape((-1, 3, 3))
            extra_cell_data[str(data_name)] = combined

    return TetMesh(
        np.asarray(source.points[:, :3], dtype=float),
        np.vstack(tetra_blocks),
        np.concatenate(tag_blocks),
        names,
        extra_cell_data,
    )


def load_result_npz(path: str | Path) -> tuple[TetMesh, np.ndarray, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"points", "tetra", "cell_tags", "displacement", "deformed_points"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing required result fields: {missing}")
        raw_names = data.get("tag_names_json")
        names = json.loads(str(raw_names.item())) if raw_names is not None else {}
        raw_cell_data_names = data.get("cell_data_names_json")
        cell_data_names = (
            json.loads(str(raw_cell_data_names.item()))
            if raw_cell_data_names is not None
            else []
        )
        cell_data = {
            str(name): data[f"cell_data__{name}"]
            for name in cell_data_names
            if f"cell_data__{name}" in data
        }
        mesh = TetMesh(data["points"], data["tetra"], data["cell_tags"], names, cell_data)
        displacement = np.asarray(data["displacement"], dtype=float)
        deformed_points = np.asarray(data["deformed_points"], dtype=float)
    if displacement.shape != mesh.points.shape:
        raise ValueError("result displacement shape does not match mesh points")
    if deformed_points.shape != mesh.points.shape:
        raise ValueError("result deformed_points shape does not match mesh points")
    return mesh, displacement, deformed_points


def save_npz(path: str | Path, mesh: TetMesh, result: MorphResult | None = None) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "points": mesh.points,
        "tetra": mesh.tetra,
        "cell_tags": mesh.cell_tags,
        "tag_names_json": np.asarray(json.dumps(mesh.tag_names)),
        "cell_data_names_json": np.asarray(json.dumps(list(mesh.cell_data))),
    }
    for name, values in mesh.cell_data.items():
        payload[f"cell_data__{name}"] = np.asarray(values)
    if result is not None:
        payload.update(
            {
                "deformed_points": result.points,
                "displacement": result.displacement,
                "growth_lambda": result.growth_lambda,
                "j_total": result.j_total,
                "j_elastic": result.j_elastic,
            }
        )
    np.savez_compressed(path, **payload)


def _ascii(array: np.ndarray) -> str:
    flat = np.asarray(array).ravel()
    if np.issubdtype(flat.dtype, np.integer):
        return " ".join(str(int(value)) for value in flat)
    return " ".join(f"{float(value):.16g}" for value in flat)


def _write_vtu(
    path: str | Path,
    points: np.ndarray,
    tetra: np.ndarray,
    point_data: dict[str, np.ndarray],
    cell_data: dict[str, np.ndarray],
) -> None:
    """Write a dependency-free ASCII tetrahedral VTU."""
    path = Path(path)
    points = np.asarray(points, dtype=float)
    connectivity = np.asarray(tetra, dtype=np.int64)
    n_cells = len(connectivity)
    offsets = np.arange(1, n_cells + 1, dtype=np.int64) * 4
    types = np.full(n_cells, 10, dtype=np.uint8)  # VTK_TETRA

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{n_cells}">',
        "      <PointData>",
    ]
    for name, values in point_data.items():
        values = np.asarray(values)
        components = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
        lines.append(
            f'        <DataArray type="Float64" Name="{escape(name)}" '
            f'NumberOfComponents="{components}" format="ascii">{_ascii(values)}</DataArray>'
        )
    lines.extend(["      </PointData>", "      <CellData>"])
    for name, values in cell_data.items():
        vtk_type = "Int64" if np.issubdtype(np.asarray(values).dtype, np.integer) else "Float64"
        values = np.asarray(values)
        components = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
        lines.append(
            f'        <DataArray type="{vtk_type}" Name="{escape(name)}" '
            f'NumberOfComponents="{components}" format="ascii">{_ascii(values)}</DataArray>'
        )
    lines.extend(
        [
            "      </CellData>",
            "      <Points>",
            f'        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{_ascii(points)}</DataArray>',
            "      </Points>",
            "      <Cells>",
            f'        <DataArray type="Int64" Name="connectivity" format="ascii">{_ascii(connectivity)}</DataArray>',
            f'        <DataArray type="Int64" Name="offsets" format="ascii">{_ascii(offsets)}</DataArray>',
            f'        <DataArray type="UInt8" Name="types" format="ascii">{_ascii(types)}</DataArray>',
            "      </Cells>",
            "    </Piece>",
            "  </UnstructuredGrid>",
            "</VTKFile>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_mesh_vtu(path: str | Path, mesh: TetMesh) -> None:
    cell_data = {"region": mesh.cell_tags}
    for name, values in mesh.cell_data.items():
        if name not in cell_data:
            cell_data[name] = np.asarray(values)
    _write_vtu(path, mesh.points, mesh.tetra, {}, cell_data)


def save_vtu(path: str | Path, mesh: TetMesh, result: MorphResult) -> None:
    """Write a dependency-free ASCII VTU result for inspection in ParaView."""
    point_data = {
        "displacement": result.displacement,
        "displacement_magnitude": np.linalg.norm(result.displacement, axis=1),
    }
    cell_data = {
        "region": mesh.cell_tags,
        "growth_lambda": result.growth_lambda,
        "J_total": result.j_total,
        "J_elastic": result.j_elastic,
    }
    for name, values in mesh.cell_data.items():
        if name not in cell_data:
            cell_data[name] = np.asarray(values)
    _write_vtu(path, result.points, mesh.tetra, point_data, cell_data)


def save_result_bundle(base_path: str | Path, mesh: TetMesh, result: MorphResult) -> tuple[Path, Path, Path]:
    base = Path(base_path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_suffix(".npz")
    vtu_path = base.with_suffix(".vtu")
    json_path = base.with_suffix(".json")
    save_npz(npz_path, mesh, result)
    save_vtu(vtu_path, mesh, result)
    json_path.write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return npz_path, vtu_path, json_path
