"""
Constants for shape matching and SDF deformation.

These parameters control material properties, solver settings,
mesh refinement, numerical tolerances, and initial values.
[C=> morhing.h]
"""

# Material properties
LS_LAMBDA = 10e6      # Lame parameter lambda
LS_MU     = 8.2e6     # Lame parameter mu
LS_E      = 1e14      # Young's modulus
LS_NU     = 0.05      # Poisson ratio

# Solver settings
LS_RES    = 1e-6      # CG residual tolerance
LS_MAXIT  = 5000      # Max CG iterations
LS_TGV    = 1e30      # Divergence threshold
DEFAULT_NIT   = 400   # Default number of iterations
DEFAULT_TOL   = 1e-3  # Default relative tolerance
DEFAULT_SAVE  = 10    # Save every N iterations
VERBOSE       = 2     # 0: none, 1: normal, 2: debug, 3: extreme verbose
LS_ALPHA      = 0.05  # Armijo initial step factor (same as C's LS_ALPHA)
K_NEAREST     = 30     # Number of nearest facets to test in SDF (C uses all facets)

# Mesh refinement
MESH_DREF  = 2        # Global mesh division factor
MESH_ELREF = 2        # Element refinement order
MESH_BREF  = 99       # Boundary marker for refinement

# Numerical epsilons
LONMAX = 8192
PRECI  = 1.0
EPS    = 1e-6
EPS1   = 1e-20
EPS2   = 1e-10
EPSD   = 1e-30
EPSA   = 1e-200
EPST   = -1e-2
EPSR   = 1e2

# Initial SDF values
INIVAL_2D = 1.0
INIVAL_3D = 3.0

# Utility functions
def ls_max(a, b):
    return b if a < b else a

def ls_min(a, b):
    return a if a < b else b

ENABLE_CLEARING_MESH = False
CLEANING_EPS = 1e-10
