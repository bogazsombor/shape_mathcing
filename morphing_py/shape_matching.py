"""
shape_matching.py — Elastic shape matching using DolfinX

Usage:
  python shape_matching.py template.msh target.msh [--params params.json] [--nit N] [--tol T] [--save S] [--sol target.sol] [--c_mode]
Options:
  --c_mode      disable mesh scaling, Armijo search, and smoothing (C-style).

Useaga under development: python morphing_py/shape_matching.py template.mesh target.mesh --c_mode

Implements:
  • mesh scaling (normalization)
  • exact signed-distance via AABB tree
  • P1 interpolation of optional phi_target (.sol)
  • Elasticity-based shape matching loop (Armijo or simple backtracking)
  • Region-based Dirichlet BC (internal ω)
  • Mesh smoothing (Laplacian)
  • Visualization/export (PNG, GLB, PLY, Medit .mesh/.sol)
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem import functionspace
from dolfinx.geometry import bb_tree, compute_collisions_points, squared_distance
from dolfinx.fem.petsc import LinearProblem

from constants import DEFAULT_NIT, DEFAULT_TOL, DEFAULT_SAVE, VERBOSE, LS_LAMBDA, LS_MU, LS_RES
from template_io import (
    load_template,
    save_mesh_png,
    save_pointcloud_png,
    save_mesh_glb,
    save_pointcloud_ply,
)
from elastic import make_elastic_forms
from medit_io import write_mesh, write_sol

@dataclass
class ProblemData:
    mesh: mesh.Mesh
    source_mesh: mesh.Mesh
    Vphi: fem.FunctionSpace
    Vu: fem.FunctionSpace
    phi: fem.Function
    u: fem.Function
    bc: fem.dirichletbc
    btree: object
    coord_dim: int
    topo_dim: int
    outdir: Path
    phi_target: fem.Function | None = None
    tol: float = DEFAULT_TOL


def compute_total_functional(prob: ProblemData) -> float:
    """
    Compute the full energy: elastic energy minus data term (∫φ dx).
    """
    # Elastic energy: ½ [λ(div u)^2 + 2μ ε(u):ε(u)]
    u = prob.u
    elastic_form = (LS_LAMBDA * ufl.div(u)**2 + 2 * LS_MU * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(u)))) * ufl.dx
    elastic = 0.5 * fem.assemble_scalar(fem.form(elastic_form))
    # Data term: ∫ φ dx
    data = fem.assemble_scalar(fem.form(prob.phi * ufl.dx))
    return elastic - data


def scale_mesh(msh: mesh.Mesh):
    """Normalize mesh coordinates to [0.05, 0.95] in each dimension."""
    if VERBOSE >= 2:
        print("[DEBUG] scale_mesh: before scaling, bbox:", msh.geometry.x.min(axis=0),
              msh.geometry.x.max(axis=0), flush=True)
    X = msh.geometry.x.copy()
    minc = X.min(axis=0)
    maxc = X.max(axis=0)
    delta = (maxc - minc).max()
    if delta < 1e-12:
        raise RuntimeError("Unable to scale: zero bounding-box size")
    scale = 0.9 / delta
    msh.geometry.x[:] = (X - minc) * scale + 0.05
    if VERBOSE >= 2:
        print("[DEBUG] scale_mesh: after scaling, bbox:", msh.geometry.x.min(axis=0),
              msh.geometry.x.max(axis=0), flush=True)


def create_spaces(msh: mesh.Mesh, topo_dim: int):
    """Create scalar and vector function spaces on the mesh."""
    Vphi = functionspace(msh, ("Lagrange", 1))
    Vu = functionspace(msh, ("Lagrange", 1, (topo_dim,)))
    if VERBOSE >= 2:
        nphi = Vphi.dofmap.index_map.size_local
        nu = Vu.dofmap.index_map.size_local
        print(f'[DEBUG] Created Vphi (ndofs={nphi}) and Vu (ndofs={nu}) spaces', flush=True)
    return Vphi, Vu


def make_dirichlet_bc(Vu, coord_dim: int, radius_factor=0.1):
    """Apply zero Dirichlet BC inside a central spherical region."""
    coords = Vu.tabulate_dof_coordinates().reshape((-1, coord_dim))
    center = coords.mean(axis=0)
    radius = radius_factor * np.linalg.norm(coords - center, axis=1).max()
    omega = np.where(np.linalg.norm(coords - center, axis=1) < radius)[0].astype(np.int32)
    zero_val = np.zeros(Vu.value_size, dtype=PETSc.ScalarType)
    if VERBOSE >= 2:
        print(f"[DEBUG] make_dirichlet_bc: center={center}, radius={radius}, dofs={len(omega)}", flush=True)
    return fem.dirichletbc(zero_val, omega, Vu._cpp_object)


def read_sol_scalar(path: str) -> np.ndarray:
    """Read a .sol file containing scalar values at mesh vertices."""
    if VERBOSE >= 2:
        print(f"[DEBUG] read_sol_scalar: reading {path}", flush=True)
    with open(path) as f:
        for line in f:
            if 'SolAtVertices' in line:
                n = int(next(f))
                next(f)
                data = np.array([float(next(f)) for _ in range(n)], dtype=float)
                if VERBOSE >= 2:
                    print(f"[DEBUG] read_sol_scalar: read {n} values", flush=True)
                return data
    raise RuntimeError(f'Cannot parse .sol file {path}')


def update_distance_field(Vphi, phi, prob: ProblemData):
    """Compute signed-distance field on the template mesh using the source_mesh AABB tree."""
    coords = Vphi.tabulate_dof_coordinates().reshape((-1, prob.coord_dim))
    if VERBOSE >= 2:
        print(f"[DEBUG] update_distance_field: computing distances for {coords.shape[0]} points", flush=True)
    if prob.phi_target:
        phi.x.array[:] = prob.phi_target.eval(coords)
    else:
        coll = compute_collisions_points(prob.btree, coords)
        max_entity = prob.source_mesh.topology.index_map(prob.topo_dim - 1).size_local
        dvals = np.empty(coords.shape[0], dtype=float)
        for i, x in enumerate(coords):
            neigh = coll.links(i)
            if neigh.size > 0:
                min_ds2 = np.inf
                for e in neigh.astype(np.int32):
                    if 0 <= e < max_entity:
                        ds2 = squared_distance(
                            prob.source_mesh,
                            prob.topo_dim - 1,
                            np.array([e], dtype=np.int32),
                            x.reshape(1, -1).astype(prob.source_mesh.geometry.x.dtype)
                        )[0]
                        min_ds2 = min(min_ds2, ds2)
                dvals[i] = float(np.sqrt(min_ds2)) if min_ds2 < np.inf else np.nan
            else:
                dvals[i] = np.nan
        phi.x.array[:] = dvals
    phi.x.scatter_forward()
    if VERBOSE >= 2:
        print(f"[DEBUG] Distance range: {np.nanmin(phi.x.array):.3e} to {np.nanmax(phi.x.array):.3e}", flush=True)


def solve_displacement(prob: ProblemData, a, n, ds):
    """Assemble and solve the linear elasticity problem for the displacement field."""
    if VERBOSE >= 2:
        print("[DEBUG] solve_displacement: assembling and solving elasticity system", flush=True)
    L = fem.form(-prob.phi * ufl.dot(ufl.TestFunction(prob.Vu), n) * ds)
    LinearProblem(a, L, bcs=[prob.bc], u=prob.u,
                  petsc_options={'ksp_type': 'cg', 'pc_type': 'ilu', 'ksp_rtol': LS_RES}).solve()
    if VERBOSE >= 2:
        print('[DEBUG] Elastic solve done', flush=True)


def armijo_search(prob: ProblemData, J0, alpha0=1e8, beta=0.8, c=1e-4, max_iter=40):
    """Perform a backtracking line search (Armijo) on the full functional."""
    comm = MPI.COMM_WORLD; rank = comm.rank
    disp = prob.u.x.array.reshape(-1, prob.topo_dim)
    alpha = alpha0
    for it in range(max_iter):
        prob.mesh.geometry.x[:, :prob.topo_dim] += alpha * disp
        update_distance_field(prob.Vphi, prob.phi, prob)
        J1 = compute_total_functional(prob)
        if J1 < prob.tol:
            if rank == 0:
                print(f"[INFO] Converged (J={J1:.3e} < tol={prob.tol:.3e})", flush=True)
            return 0.0
        if J1 < J0 - c * alpha * np.linalg.norm(disp)**2:
            return alpha
        prob.mesh.geometry.x[:, :prob.topo_dim] -= alpha * disp
        alpha *= beta
    prob.mesh.geometry.x[:, :prob.topo_dim] += alpha0 * disp
    update_distance_field(prob.Vphi, prob.phi, prob)
    return alpha0


def find_simple_step(prob: ProblemData, J0, alpha0=1.0, beta=0.8, min_step=1e-8, max_iter=40):
    """
    Simple backtracking: accept first descent (J < J0).
    """
    disp = prob.u.x.array.reshape(-1, prob.topo_dim)
    alpha = alpha0
    for _ in range(max_iter):
        prob.mesh.geometry.x[:, :prob.topo_dim] += alpha * disp
        update_distance_field(prob.Vphi, prob.phi, prob)
        J1 = fem.assemble_scalar(fem.form(prob.phi * ufl.dx))
        if J1 < J0:
            return alpha
        prob.mesh.geometry.x[:, :prob.topo_dim] -= alpha * disp
        alpha *= beta
        if alpha < min_step:
            break
    return alpha

def smooth_mesh(msh: mesh.Mesh, iterations: int = 1):
    """Laplacian-style smoothing: move each vertex to the average of its neighbors."""
    coords = msh.geometry.x
    v2v = msh.topology.connectivity(0, 0)
    for _ in range(iterations):
        new_coords = coords.copy()
        for v in range(coords.shape[0]):
            nbrs = v2v.links(v)
            if nbrs.size > 0:
                new_coords[v] = coords[nbrs].mean(axis=0)
        coords[:] = new_coords


def run_shape_matching(template, target, nit, tol, save, sol_path=None, c_mode=False):
    """Main driver: load meshes, set up ProblemData, run iterations."""
    comm = MPI.COMM_WORLD; rank = comm.rank
    if rank == 0:
        print(f'[INFO] template={template} target={target} nit={nit} tol={tol} save={save} c_mode={c_mode}', flush=True)

    mesh_t, topo_dim, coord_dim, ftags = load_template(template, comm, rank)
    mesh_s, _, _, _ = load_template(target, comm, rank)
    if not c_mode:
        scale_mesh(mesh_t)
        scale_mesh(mesh_s)
    btree = bb_tree(mesh_s, mesh_s.topology.dim-1)

    phi_target = None
    if sol_path:
        vals = read_sol_scalar(sol_path)
        Vpt = functionspace(mesh_s, ("Lagrange", 1))
        phi_target = fem.Function(Vpt)
        phi_target.x.array[:] = vals; phi_target.x.scatter_forward()

    Vphi, Vu = create_spaces(mesh_t, topo_dim)
    phi = fem.Function(Vphi); u = fem.Function(Vu)
    bc = make_dirichlet_bc(Vu, coord_dim)
    a, n_expr, ds = make_elastic_forms(Vu, topo_dim)

    outdir = Path(f'out_{time.strftime("%Y%m%d_%H%M%S")}')
    if rank == 0:
        outdir.mkdir()

    prob = ProblemData(
        mesh=mesh_t,
        source_mesh=mesh_s,
        Vphi=Vphi, Vu=Vu,
        phi=phi, u=u,
        bc=bc, btree=btree,
        coord_dim=coord_dim, topo_dim=topo_dim,
        outdir=outdir, phi_target=phi_target,
        tol=tol
    )

    update_distance_field(Vphi, phi, prob)

    for k in range(nit):
        if rank == 0:
            print(f'[INFO] Iter {k+1}/{nit}', flush=True)
        update_distance_field(Vphi, phi, prob)
        # use full functional for J0
        J0 = compute_total_functional(prob)
        # print energy
        if VERBOSE >= 1 and rank == 0:
            print(f"[INFO] J={J0:.3e}", flush=True)
        solve_displacement(prob, a, n_expr, ds)
        if c_mode:
            alpha = find_simple_step(prob, J0)
        else:
            alpha = armijo_search(prob, J0)
        if not c_mode:
            smooth_mesh(mesh_t)

        if (k+1) % save == 0 and rank == 0:
            write_mesh(str(outdir/f'template{k+1}.mesh'), mesh_t.geometry.x.copy(), [], [], topo_dim)
            write_sol(str(outdir/f'template{k+1}.sol'), phi.x.array.copy())
            save_mesh_png(mesh_t, outdir/f'template{k+1}.png')
            print(f'[INFO] Dumped iter {k+1}', flush=True)

    if rank == 0:
        print(f'[INFO] Done after {nit} iterations', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('template')
    ap.add_argument('target')
    ap.add_argument('--nit', type=int,   default=DEFAULT_NIT)
    ap.add_argument('--tol', type=float, default=DEFAULT_TOL)
    ap.add_argument('--save',type=int,   default=DEFAULT_SAVE)
    ap.add_argument('--sol', type=str)
    ap.add_argument('--c_mode', action='store_true', help='disable Python-specific steps')
    args = ap.parse_args()
    run_shape_matching(
        args.template, args.target, args.nit,
        args.tol, args.save, args.sol,
        c_mode=args.c_mode
    )

if __name__ == '__main__':
    main()
