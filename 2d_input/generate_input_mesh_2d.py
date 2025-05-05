#!/usr/bin/env python3
"""
generate_input_mesh_2d.py  ·  2‑D test‑data generator

Default output ( --format msh )
  template.msh / template.png
  target.msh   / target.png / target.ply

Optional MEDIT ASCII ( --format mesh )
  template.mesh / template.png
  target.mesh   / target.png / target.ply


  # Gmsh ASCII (3‑column coords, triangle elements)
python generate_input_mesh_2d.py --resolution 80 --pc-resolution 1500 --a 1.3 --b 1.0

# MEDIT ASCII (Dimension 2 header, 2‑column coords)
python generate_input_mesh_2d.py --format mesh --resolution 80 --a 1.3 --b 1.0

"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import gmsh
import meshio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from template_io import save_pointcloud_ply  # helper one level up
# ───────────────────────────── geometry builders ───────────────────────────
def build_template_ellipse(res: int, a: float, b: float) -> None:
    gmsh.model.add("template")
    pts = [
        gmsh.model.occ.addPoint(a * np.cos(t), b * np.sin(t), 0, meshSize=0)
        for t in np.linspace(0, 2 * np.pi, res, endpoint=False)
    ]
    spl = gmsh.model.occ.addSpline(pts + [pts[0]])
    loop = gmsh.model.occ.addCurveLoop([spl])
    gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    lc = 2 * np.pi * max(a, b) / res
    for dim, tag in gmsh.model.getBoundary([(1, spl)]):
        gmsh.model.mesh.setSize([(dim, tag)], lc)
    gmsh.model.mesh.generate(2)


def build_target_rectangle(res: int, a: float, b: float) -> None:
    gmsh.model.add("target")
    p = [
        gmsh.model.occ.addPoint(*xy, 0, meshSize=0)
        for xy in [(-a, -b), (a, -b), (a, b), (-a, b)]
    ]
    lines = [gmsh.model.occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.occ.addCurveLoop(lines)
    gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    seg = max(1, round(res / 4))
    gmsh.model.mesh.setSize([(1, lines[0]), (1, lines[2])], 2 * a / seg)
    gmsh.model.mesh.setSize([(1, lines[1]), (1, lines[3])], 2 * b / seg)
    gmsh.model.mesh.generate(2)

# ──────────────────────────── mesh writers / previews ──────────────────────
def write_gmsh(path: Path) -> None:
    gmsh.write(str(path))
    print(f"✔ saved {path.name}")


def write_medit(path: Path) -> None:
    """Convert current gmsh model to MEDIT .mesh (Dimension 2)."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_msh = Path(tmp.name)
    gmsh.write(str(tmp_msh))
    data = meshio.read(tmp_msh)
    tmp_msh.unlink()
    data.points = data.points[:, :2]          # drop z column
    meshio.write(path, data, file_format="medit")
    print(f"✔ saved {path.name}")


def mesh_png(src: Path, dst: Path) -> None:
    m = meshio.read(src)
    tris = next((c.data for c in m.cells if c.type == "triangle"), None)
    if tris is None:
        return
    tri = mtri.Triangulation(m.points[:, 0], m.points[:, 1], tris)
    plt.triplot(tri, lw=0.4)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig(dst, bbox_inches="tight")
    plt.close()
    print(f"✔ saved {dst.name}")

# ─────────────────────── dense rectangle boundary cloud ────────────────────
def dense_rectangle_cloud(n_pts: int, a: float, b: float, out_ply: Path) -> None:
    n = n_pts // 4
    rem = n_pts - 4 * n
    parts = []

    xs = np.linspace(-a, a, n, endpoint=False)
    parts.append(np.column_stack([xs, -b * np.ones_like(xs), np.zeros_like(xs)]))

    ys = np.linspace(-b, b, n, endpoint=False)
    parts.append(np.column_stack([a * np.ones_like(ys), ys, np.zeros_like(ys)]))

    xs = np.linspace(a, -a, n, endpoint=False)
    parts.append(np.column_stack([xs, b * np.ones_like(xs), np.zeros_like(xs)]))

    ys = np.linspace(b, -b, n + rem, endpoint=False)
    parts.append(np.column_stack([-a * np.ones_like(ys), ys, np.zeros_like(ys)]))

    cloud = np.vstack(parts)
    save_pointcloud_ply(cloud, out_ply)
    print(f"✔ saved {out_ply.name} ({len(cloud)} pts)")

# ───────────────────────────────────── main ────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=("msh", "mesh"), default="msh",
                    help="'msh' = Gmsh 2.2, 'mesh' = MEDIT ASCII")
    ap.add_argument("--resolution", type=int, default=60,
                    help="boundary segments for each mesh")
    ap.add_argument("--pc-resolution", type=int, default=400,
                    help="points in dense PLY cloud (target)")
    ap.add_argument("--a", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.8)
    args = ap.parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    # template --------------------------------------------------------------
    build_template_ellipse(args.resolution, args.a, args.b)
    if args.format == "msh":
        write_gmsh(Path("template.msh"))
        templ_path = Path("template.msh")
    else:
        write_medit(Path("template.mesh"))
        templ_path = Path("template.mesh")
    gmsh.clear()

    # target ----------------------------------------------------------------
    build_target_rectangle(args.resolution, args.a, args.b)
    if args.format == "msh":
        write_gmsh(Path("target.msh"))
        targ_path = Path("target.msh")
    else:
        write_medit(Path("target.mesh"))
        targ_path = Path("target.mesh")
    gmsh.finalize()

    # PNG previews ----------------------------------------------------------
    mesh_png(templ_path, templ_path.with_suffix(".png"))
    mesh_png(targ_path, targ_path.with_suffix(".png"))

    # dense PLY -------------------------------------------------------------
    dense_rectangle_cloud(args.pc_resolution, args.a, args.b, Path("target.ply"))

if __name__ == "__main__":
    main()
