#!/usr/bin/env python3
"""
Generate a *pair* of 3‑D meshes (template + target) for the
elastic‑shape‑matching demo.

* **Template**   – unit sphere centred at the origin.
* **Target**     – smooth but complicated bump function applied to the sphere
                   (looks like a blobby animal silhouette when projected).

Both meshes are saved in **Medit .mesh** format so they can be fed directly
into `shape_matching.py`:

```bash
python generate_3d_example_meshes.py         # produces sphere.mesh  blob.mesh
mpirun -n 4 python shape_matching.py sphere.mesh blob.mesh \
          --nit 120 --tol 1e-6 --save 10
```
"""
from __future__ import annotations
import math, os, sys, gmsh, meshio, numpy as np


def make_sphere(radius: float = 1.0, lc: float = 0.2):
    gmsh.model.occ.addSphere(0, 0, 0, radius, tag=1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], tag=1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)


def deform_nodes(nodes: np.ndarray) -> np.ndarray:
    """Apply a smooth radial bump that roughly resembles ears / legs."""
    x, y, z = nodes.T
    r = np.linalg.norm(nodes, axis=1)
    # spherical coordinates
    theta = np.arccos(np.clip(z / r, -1, 1))        # polar angle
    phi = np.arctan2(y, x)
    # bump amplitude
    bump = 0.3 * np.sin(3 * theta) * np.cos(5 * phi) + 0.15 * np.sin(6*phi)
    new_r = 1.0 + bump
    scale = new_r / r
    return nodes * scale[:, None]


def write_medit(basename: str):
    """Convert Gmsh .msh → Medit .mesh, stripping node data that confuses Medit."""
    gmsh.write(f"{basename}.msh")
    msh = meshio.read(f"{basename}.msh")
    # Medit expects one integer label per vertex; easiest is to drop point_data entirely
    clean_mesh = meshio.Mesh(points=msh.points, cells=msh.cells, cell_data=msh.cell_data)
    meshio.write(f"{basename}.mesh", clean_mesh, file_format="medit")
    os.remove(f"{basename}.msh")


def main():
    gmsh.initialize()
    make_sphere(radius=1.0, lc=0.12)
    gmsh.model.mesh.generate(3)

    # Dump initial mesh to .msh and load with meshio
    gmsh.write("sphere_tmp.msh")
    msh = meshio.read("sphere_tmp.msh")

    # ---------------- template ----------------
    clean = meshio.Mesh(points=msh.points, cells=msh.cells, cell_data=msh.cell_data)
    meshio.write("sphere.mesh", clean, file_format="medit")

    # ---------------- target ------------------
    deformed_pts = deform_nodes(msh.points.copy())
    blob = meshio.Mesh(points=deformed_pts, cells=msh.cells, cell_data=msh.cell_data)
    meshio.write("blob.mesh", blob, file_format="medit")

    print("Wrote sphere.mesh and blob.mesh (Medit format)")
    os.remove("sphere_tmp.msh")
    gmsh.finalize()


if __name__ == "__main__":
    main()
