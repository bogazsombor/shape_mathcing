"""
Linear elasticity forms for shape matching and deformation.

This module defines the bilinear form for isotropic linear elasticity:

    ε(u) = sym(∇u)
    σ(u) = λ tr(ε(u)) I + 2 μ ε(u)

The form a(u, v) = ∫ σ(u) : ε(v) dx is returned along with
the facet normal and surface measure for applying boundary conditions.
[C=> elas.c]

"""

import ufl
from ufl import nabla_grad
from constants import LS_LAMBDA, LS_MU


def make_elastic_forms(V, topo_dim, lam=LS_LAMBDA, mu=LS_MU):
    """
    Create the elasticity bilinear form and boundary measures.

    Parameters
    ----------
    V : FunctionSpace
        Vector function space for displacement.
    topo_dim : int
        Topological dimension of the mesh (2 or 3).
    lam : float
        Lame parameter λ.
    mu : float
        Lame parameter μ.

    Returns
    -------
    a : ufl.Form
        Bilinear form for stiffness.
    n : ufl.NormalVector
        Facet normal for boundary integrals.
    ds : ufl.Measure
        Surface measure for Neumann conditions.
    """
    # Strain tensor
    def eps(u):
        return ufl.sym(nabla_grad(u))

    # Stress tensor
    def sigma(u):
        return lam * ufl.tr(eps(u)) * ufl.Identity(topo_dim) + 2 * mu * eps(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elastic stiffness form
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx

    # Boundary data
    n = ufl.FacetNormal(V.mesh)
    ds = ufl.Measure("ds", domain=V.mesh)

    return a, n, ds
