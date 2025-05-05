import numpy as np


def write_mesh(path, points, cells, cell_refs, dimension):
    """
    Write a Medit .mesh file including triangles or tetrahedra.

    Parameters
    ----------
    path : str
        Output filename (should end in .mesh).
    points : array_like, shape (npts, dim)
        Vertex coordinates.
    cells : list of (cell_type, ndarray)
        List of tuples: cell_type is 'triangle' or 'tetra', and ndarray is
        an (n_cells, vertices_per_cell) array of zero-based indices.
    cell_refs : list of array_like
        List of reference tags arrays, one per cell block, matching cells.
    dimension : int
        Spatial dimension (2 or 3). Determines whether to write triangles or tetrahedra.
    """
    pts = np.asarray(points)
    dim = int(dimension)
    assert pts.ndim == 2 and pts.shape[1] >= (2 if dim == 2 else 3)

    with open(path, 'w') as f:
        f.write('MeshVersionFormatted 1\n')
        f.write(f'Dimension {dim}\n')

        # Vertices
        f.write(f'Vertices\n{len(pts)}\n')
        for p in pts:
            x, y = p[0], p[1]
            z = p[2] if dim == 3 and p.shape[0] > 2 else 0
            f.write(f'{x} {y} {z} 0\n')

        # Cells
        for (cell_type, arr), refs in zip(cells, cell_refs):
            arr = np.asarray(arr, dtype=int)
            n = len(arr)
            if cell_type.lower().startswith('tri'):
                f.write(f'Triangles\n{n}\n')
                for tri, r in zip(arr, refs):
                    i, j, k = tri + 1  # 1-based indexing
                    f.write(f'{i} {j} {k} {int(r)}\n')
            elif cell_type.lower().startswith('tet'):
                f.write(f'Tetrahedra\n{n}\n')
                for tet, r in zip(arr, refs):
                    a, b, c, d = tet + 1
                    f.write(f'{a} {b} {c} {d} {int(r)}\n')
        f.write('End\n')


def write_sol_scalar(path, values):
    """
    Write a scalar Medit .sol file at vertices.
    """
    vals = np.asarray(values).ravel()
    n = len(vals)
    with open(path, 'w') as f:
        f.write('MeshVersionFormatted 1\n')
        f.write('Dimension 3\n')
        f.write(f'SolAtVertices\n{n}\n1 1\n')
        for v in vals:
            f.write(f'{v}\n')


def write_sol_vector(path, vecs):
    """
    Write a vector Medit .sol file at vertices.
    """
    arr = np.asarray(vecs)
    n = arr.shape[0]
    d = arr.shape[1]
    with open(path, 'w') as f:
        f.write('MeshVersionFormatted 1\n')
        f.write('Dimension 3\n')
        f.write(f'SolAtVertices\n{n}\n{d} 1\n')
        for v in arr:
            f.write(' '.join(str(x) for x in v) + '\n')

# Alias for backward compatibility
write_sol = write_sol_scalar
