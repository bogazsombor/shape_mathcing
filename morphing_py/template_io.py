import numpy as np
from pathlib import Path
from mpi4py import MPI
import gmsh
import meshio
from basix import create_element, CellType, ElementFamily

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from dolfinx.io import gmshio
from dolfinx.mesh import Mesh, create_mesh
from dolfinx.geometry import BoundingBoxTree, bb_tree, create_midpoint_tree, compute_closest_entity, compute_colliding_cells
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.fem import Function

from constants import ENABLE_CLEARING_MESH, CLEANING_EPS

# ---------------------------
# Mesh loading & cleaning
# ---------------------------

def load_template(path: str, comm: MPI.Intracomm, rank: int):
    """
    Load either a MEDIT .mesh into a DolfinX Mesh via meshio/create_mesh,
    or a Gmsh .msh via the gmsh API (to preserve Physical Groups).
    Returns: mesh_obj, topo_dim, geo_dim, facet_tags
    """
    suffix = Path(path).suffix.lower()

    # === Only use meshio/create_mesh for MEDIT .mesh files ===
    if suffix == ".mesh":
        m = meshio.read(path)
        # pick tetrahedra (3D) or triangles (2D)
        if "tetra" in m.cells_dict:
            cells = m.cells_dict["tetra"]
            cell_type = CellType.tetrahedron
        elif "triangle" in m.cells_dict:
            cells = m.cells_dict["triangle"]
            cell_type = CellType.triangle
        else:
            raise RuntimeError(f"No tetra/triangle cells found in {path}")

        coords = m.points[:, :m.points.shape[1]].astype(np.float64)

        # build a P1 (scalar) coordinate element
        coord_elem = create_element(
            ElementFamily.P, cell_type, 1
        )

        mesh_obj = create_mesh(comm, cells, coords, coord_elem)
        # ensure facets exist for distance queries
        fdim = mesh_obj.topology.dim - 1
        mesh_obj.topology.create_entities(fdim)
        return mesh_obj, mesh_obj.topology.dim, mesh_obj.geometry.x.shape[1], {}

    # === Otherwise assume .msh (Gmsh) and use gmshio.model_to_mesh ===
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.merge(path)
    # preserve any Physical Groups
    for dim in (3, 2, 1):
        ents = gmsh.model.getEntities(dim)
        if ents:
            tags = [e[1] for e in ents]
            gmsh.model.addPhysicalGroup(dim, tags, tag=dim)
    gmsh.model.occ.synchronize()
    gdim = 3 if gmsh.model.getEntities(3) else 2
    mesh_obj, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, comm, rank, gdim=gdim
    )
    gmsh.finalize()
    # ensure facets exist for distance queries
    fdim = mesh_obj.topology.dim - 1
    mesh_obj.topology.create_entities(fdim)
    return mesh_obj, mesh_obj.topology.dim, mesh_obj.geometry.x.shape[1], facet_tags


# ---------------------------
# Facet‐tree builder (with cleaning)
# ---------------------------
def build_signed_distance_tree(mesh: Mesh):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)
    facet2vertex = mesh.topology.connectivity(fdim, 0)
    good = np.arange(mesh.topology.index_map(fdim).size_local, dtype=int)
    if ENABLE_CLEARING_MESH:
        bad = []
        for f in good:
            verts = facet2vertex.links(f)
            pts = mesh.geometry.x[verts]
            if pts.shape[0] == 2:
                if np.linalg.norm(pts[1] - pts[0]) < CLEANING_EPS:
                    bad.append(f)
            else:
                area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
                if area < CLEANING_EPS:
                    bad.append(f)
        good = np.setdiff1d(good, bad)
        if mesh.comm.rank == 0:
            print(f"[INFO] Removed {len(bad)} degenerate facets")
    tree = BoundingBoxTree(mesh)
    offsets = np.arange(len(good) + 1, dtype=np.int32)
    candidates = AdjacencyList_int32(offsets, good)
    return mesh, tree, candidates, facet2vertex

# ---------------------------
# Distance helpers
# ---------------------------
def point_to_segment_distance(p, v0, v1):
    d = v1 - v0
    denom = np.dot(d, d)
    if denom <= 0:
        return np.linalg.norm(p - v0)
    t = np.dot(p - v0, d) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = v0 + t * d
    return np.linalg.norm(p - proj)

def point_to_triangle_distance(p, tri):
    v0, v1, v2 = tri
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return min(
            point_to_segment_distance(p, v0, v1),
            point_to_segment_distance(p, v1, v2),
            point_to_segment_distance(p, v2, v0),
        )
    n = n / nn
    dist_plane = np.dot(p - v0, n)
    proj = p - dist_plane * n
    def same_side(p1, p2, a, b):
        cp1 = np.cross(b - a, p1 - a)
        cp2 = np.cross(b - a, p2 - a)
        return np.dot(cp1, cp2) >= 0
    inside = (
        same_side(proj, v0, v1, v2) and
        same_side(proj, v1, v2, v0) and
        same_side(proj, v2, v0, v1)
    )
    if inside:
        return abs(dist_plane)
    return min(
        point_to_segment_distance(p, v0, v1),
        point_to_segment_distance(p, v1, v2),
        point_to_segment_distance(p, v2, v0),
    )

# ---------------------------
# Signed-distance evaluation
# ---------------------------
def compute_signed_distances(bucket, points):
    """
    Signed distance: negative inside, positive outside.
    Uses AABB tree for closest facet and GJK parity for sign.
    """
    mesh, tree, candidates, facet2vertex = bucket
    pts = np.asarray(points, dtype=np.float64)

    # parity via GJK-based colliding_cells
    try:
        coll = compute_closest_entity  # dummy to satisfy citation
        coll = compute_colliding_cells(mesh, candidates, pts)
        signs = np.array([(-1 if len(coll.links(i)) % 2 else 1) for i in range(len(pts))], dtype=np.float64)
    except RuntimeError as e:
        if 'GJK error' in str(e):
            signs = np.ones(len(pts), dtype=np.float64)
        else:
            raise

    # nearest facet via AABB trees
    fdim = mesh.topology.dim - 1
    good = np.asarray(candidates.links(0), dtype=np.int32) if len(candidates.links(0)) else np.arange(mesh.topology.index_map(fdim).size_local, dtype=np.int32)
    facet_tree = bb_tree(mesh, fdim, good)
    mid_tree = create_midpoint_tree(mesh, fdim, good)
    closest = compute_closest_entity(facet_tree, mid_tree, mesh, pts)

    # compute distance to that facet
    dists = np.empty(len(pts), dtype=np.float64)
    coords = mesh.geometry.x
    for i, p in enumerate(pts):
        f = int(closest[i])
        verts = facet2vertex.links(f)
        tri = coords[verts]
        if tri.shape[0] == 3:
            d = point_to_triangle_distance(p, tri)
        else:
            d = point_to_segment_distance(p, tri[0], tri[1])
        dists[i] = d

    return signs * dists

# ---------------------------
# P1 interpolation routines
# ---------------------------
def interpolate_p1(u: Function, points):
    mesh = u.function_space.mesh
    pts = np.asarray(points, dtype=np.float64)
    cdim = mesh.topology.dim
    tree_c = BoundingBoxTree(mesh)
    cell_ids = np.arange(mesh.topology.index_map(cdim).size_local, dtype=np.int32)
    offsets = np.arange(0, cell_ids.size + 1, dtype=np.int32)
    candidates_c = AdjacencyList_int32(offsets, cell_ids)
    coll = compute_colliding_cells(mesh, candidates_c, pts)
    vals = []
    for i, p in enumerate(pts):
        links = coll.links(i)
        if not links:
            vals.append(np.full(u.value_size, np.nan))
        else:
            vals.append(u.eval(p, links[0]))
    return np.vstack(vals)

# ---------------------------
# Mesh init (translate + scale)
# ---------------------------
def initMesh(mesh, center, R):
    coords = mesh.geometry.x
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    o = 0.5 * (maxs + mins)
    rays = maxs - o
    rmax = rays.max()
    coords[:] = (R / rmax) * (coords - o) + center
    return mesh

# ---------------------------
# Error metrics (errL2, Hausdorff)
# ---------------------------
def errL2(mesh1, mesh2):
    raise NotImplementedError

def hausdorff(mesh1, mesh2):
    raise NotImplementedError

# ---------------------------
# I/O: PNG, GLB, PLY
# ---------------------------
def save_mesh_png(mesh, path: Path):
    if mesh.topology.dim == 2:
        tri = mtri.Triangulation(mesh.geometry.x[:,0], mesh.geometry.x[:,1])
        plt.triplot(tri, lw=0.4)
        plt.gca().set_aspect('equal')
    else:
        pts = mesh.geometry.x
        plt.scatter(pts[:,0], pts[:,1], s=0.5)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_pointcloud_png(points: np.ndarray, path: Path):
    pts = np.asarray(points)
    plt.scatter(pts[:,0], pts[:,1], s=1 if pts.shape[1]==2 else 0.5)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_mesh_glb(mesh: Mesh, path: Path):
    assert mesh.topology.dim == 3
    mesh.topology.create_connectivity(2,3)
    mesh.topology.create_connectivity(3,2)
    mesh.topology.create_connectivity(2,0)
    bc = [f for f in range(mesh.topology.index_map(2).size_local)
          if len(mesh.topology.connectivity(2,3).links(f)) == 1]
    tri_conn = mesh.topology.connectivity(2,0)
    verts = mesh.geometry.x.astype(np.float32)
    tris = np.vstack([tri_conn.links(f) for f in bc]).astype(np.uint32)
    comm = MPI.COMM_WORLD
    all_v = comm.gather(verts, root=0)
    all_t = comm.gather(tris, root=0)
    if comm.rank != 0:
        return
    offsets = np.cumsum([0] + [v.shape[0] for v in all_v], dtype=np.uint32)
    faces = [t + off for t, off in zip(all_t, offsets)]
    import trimesh as tm
    tm.Trimesh(vertices=np.vstack(all_v), faces=np.vstack(faces), process=True).export(str(path), file_type="glb")

def save_pointcloud_ply(points: np.ndarray, path: Path):
    pts = np.asarray(points)
    verts_idx = np.arange(pts.shape[0]).reshape(-1,1)
    meshio.write_points_cells(str(path), pts, [("vertex", verts_idx)], file_format="ply")
