# -*- coding: utf-8 -*-
"""
Here I store all assistant functions about polynomial basis functions.

@author: Yi & Lorenzo. Created on Wed Feb 21 21:30:53 2018
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
from functools import partial
from scipy.special import legendre
import matplotlib.pyplot as plt


# %% quad
def lobatto_quad(p):
    """Gauss Lobatto quadrature.
    Args:
        p (int) = order of quadrature

    Returns:
        nodal_pts (np.array) = nodal points of quadrature
        w (np.array) = correspodent weights of the quarature.
    """
    # nodes
    x_0 = np.cos(np.arange(1, p) / p * np.pi)
    nodal_pts = np.zeros((p + 1))
    # final and inital pt
    nodal_pts[0] = 1
    nodal_pts[-1] = -1
    # Newton method for root finding
    for i, ch_pt in enumerate(x_0):
        leg_p = partial(_legendre_prime_lobatto, n=p)
        leg_pp = partial(_legendre_double_prime, n=p)
        nodal_pts[i + 1] = _newton_method(leg_p, leg_pp, ch_pt, 100)

    # weights
    weights = 2 / (p * (p + 1) * (legendre(p)(nodal_pts)) ** 2)
    return nodal_pts[::-1], weights


def gauss_quad(p):
    # Chebychev pts as inital guess
    x_0 = np.cos(np.arange(1, p + 1) / (p + 1) * np.pi)
    nodal_pts = np.empty(p)
    for i, ch_pt in enumerate(x_0):
        leg = legendre(p)
        leg_p = partial(_legendre_prime, n=p)
        nodal_pts[i] = _newton_method(leg, leg_p, ch_pt, 100)

    weights = 2 / (p * legendre(p - 1)(nodal_pts)
                   * _legendre_prime(nodal_pts, p))
    return nodal_pts[::-1], weights


# %% polynomials
def lagrange_basis(nodes, x=None):
    if x is None:
        x = nodes
    if isinstance(nodes, list):
        nodes = np.array(nodes)
    p = np.size(nodes)
    basis = np.ones((p, np.size(x)))
    # lagrange basis functions
    for i in range(p):
        for j in range(p):
            if i != j:
                basis[i, :] *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return basis

# %% polynomials

'''DEPRECATED MODIFICATIONS'''

def lagrange_basis_depr(nodes, idx=None, x=None):
    if x is None:
        x = nodes
    if isinstance(nodes, list):
        nodes = np.array(nodes)

    p = np.size(nodes)

    # IF AN INDEX IS GIVEN, WE WANT ONLY THE ITH POLYNOMIAL AT THE POINTS X
    if idx is not None:
        basis = np.ones(x.shape)
        # lagrange basis functions
        for i in range(p):
            '''test
            for j in range(p):
                basis[np.argwhere(idx!=j)[0],np.argwhere(idx!=j)[1]] *= (x - nodes[i]*np.ones_like(x)) / ((nodes[j] - nodes[i])*np.ones_like(x))
            end test'''
            if i != idx:
                basis *= (x - nodes[i]*np.ones_like(x)) / ((nodes[idx] - nodes[i])*np.ones_like(x))

    # ELSE, EVEN IF A MATRIX OF POINTS IS GIVEN, ALL VALUES SHOULD BE FOUND and stored in a p+1 tensor
    else:
        basis = np.ones((p, x.shape[0], x.shape[1]))

        for i in range(p):
            for j in range(p):
                if i != j:
                    basis[i] *= (x - nodes[j]*np.ones_like(x)) / ((nodes[i] - nodes[j])*np.ones_like(x))
    return basis


def edge_basis_depr(nodes, idx,  x=None):
    """Return the edge polynomials."""
    if x is None:
        x = nodes
    if isinstance(nodes, list):
        nodes = np.array(nodes)
    p = np.size(nodes) - 1
    derivatives_poly = _derivative_poly_depr(p, nodes, x)
    edge_poly = np.zeros(x.shape)
    #for i in range(p):
     #   for j in range(i + 1):
      #      edge_poly[i] -= derivatives_poly[j, :]
    for j in range(idx+1):
        edge_poly -= derivatives_poly[j]

    return edge_poly


def _derivative_poly_depr(p, nodes, x):
    """Return the derivatives of the polynomials in the domain x.
    THE POINT HERE WAS TO CREATE A (P+1)-TENSOR CONTAINING THE NODAL DERIVATIVES FOR EACH VAL OF x, AT EVERY NODE"""
    nodal_derivative = _derivative_poly_nodes(p, nodes)
    polynomials = lagrange_basis_depr(nodes, x=x)
    #print(polynomials)

    #total_nodal_derivative = np.empty((p+1,nodal_derivative.shape[0], nodal_derivative.shape[1]))
    #for i in range(p+1):
        #total_nodal_derivative[i] = np.transpose(nodal_derivative)[i,:]

    #print(total_nodal_derivative)

    # return np.einsum('ijk,ikl->ijl', total_nodal_derivative, polynomials)
    return np.einsum('ij, jkl -> ikl', np.transpose(nodal_derivative), polynomials)

'''END OF MODIFICATIONS'''



def edge_basis(nodes, x=None):
    """Return the edge polynomials."""
    if x is None:
        x = nodes
    if isinstance(nodes, list):
        nodes = np.array(nodes)
    p = np.size(nodes) - 1
    derivatives_poly = _derivative_poly(p, nodes, x)
    edge_poly = np.zeros((p, np.size(x)))
    #print(derivatives_poly)
    #print()
    for i in range(p):
        for j in range(i + 1):
            #print(i, j, derivatives_poly[j, :])
            edge_poly[i] -= derivatives_poly[j, :]

    return edge_poly


# %% plt
def plot_lagrange_basis(nodes, dual=False, plot_density=300, ylim_ratio=0.15,
                        title=True, left=0.15, bottom=0.15,
                        tick_size=15, label_size=15, title_size=15,
                        linewidth=1.2, saveto=None, figsize=(6, 4), usetex=True):
    plt.rc('text', usetex=usetex)
    x = np.linspace(-1, 1, plot_density)
    basis = lagrange_basis(nodes, x=x)
    if dual:
        quad_nodes, quad_weights = gauss_quad(np.size(nodes) + 1)
        quad_basis = lagrange_basis(nodes, x=quad_nodes)
        M = np.einsum('ik,jk,k->ij', quad_basis, quad_basis, quad_weights)
        M = np.linalg.inv(M)
        basis = np.einsum('ik,ij->jk', basis, M)
    bmx = basis.max();
    bmi = basis.min()
    interval = bmx - bmi
    ylim = [bmi - interval * ylim_ratio, bmx + interval * ylim_ratio]
    plt.figure(figsize=figsize)
    for basis_i in basis:
        plt.plot(x, basis_i, linewidth=1 * linewidth)
    for i in nodes:
        plt.plot([i, i], ylim, '--', color=(0.2, 0.2, 0.2, 0.2), linewidth=0.8 * linewidth)
    if not dual:
        plt.plot([-1, 1], [1, 1], ':', color=(0.2, 0.2, 0.2, 0.7), linewidth=0.8 * linewidth)
    plt.plot([-1, 1], [0, 0], '--', color=(0.5, 0.5, 0.5, 1), linewidth=0.8 * linewidth)
    if title is True:
        if dual:
            title = 'dual Lagrange polynomials'
        else:
            title = 'Lagrange polynomials'
    if title is not None:
        plt.title(title, fontsize=title_size)
    plt.gcf().subplots_adjust(left=left)
    plt.gcf().subplots_adjust(bottom=bottom)
    plt.ylim(ylim)
    plt.xlim([-1, 1])
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.xlabel(r"$\xi$", fontsize=label_size)
    if dual:
        plt.ylabel(r"$\tilde{h}_{i}(\xi)$", fontsize=label_size)
    else:
        plt.ylabel(r"$h_{i}(\xi)$", fontsize=label_size)
    plt.show()
    if saveto is not None:
        plt.savefig(saveto, bbox_inches='tight')


def plot_edge_basis(nodes, dual=False, plot_density=300, ylim_ratio=0.15,
                    title=True, left=0.15, bottom=0.15,
                    tick_size=15, label_size=15, title_size=15,
                    linewidth=1.2, saveto=None, figsize=(6, 4), usetex=True):
    plt.rc('text', usetex=usetex)
    x = np.linspace(-1, 1, plot_density)
    basis = edge_basis(nodes, x)
    if dual:
        quad_nodes, quad_weights = gauss_quad(np.size(nodes) + 1)
        quad_basis = edge_basis(nodes, x=quad_nodes)
        M = np.einsum('ik,jk,k->ij', quad_basis, quad_basis, quad_weights)
        M = np.linalg.inv(M)
        basis = np.einsum('ik,ij->jk', basis, M)
    bmx = basis.max();
    bmi = basis.min()
    interval = bmx - bmi
    ylim = [bmi - interval * ylim_ratio, bmx + interval * ylim_ratio]
    plt.figure(figsize=figsize)
    for basis_i in basis:
        plt.plot(x, basis_i, linewidth=1 * linewidth)
    for i in nodes:
        plt.plot([i, i], ylim, '--', color=(0.2, 0.2, 0.2, 0.2), linewidth=0.9 * linewidth)
    plt.plot([-1, 1], [0, 0], '--', color=(0.5, 0.5, 0.5, 1), linewidth=0.9 * linewidth)
    if title is True:
        if dual:
            title = 'dual edge polynomials'
        else:
            title = 'edge polynomials'
    if title is not None:
        plt.title(title, fontsize=title_size)
    plt.gcf().subplots_adjust(left=left)
    plt.gcf().subplots_adjust(bottom=bottom)
    plt.ylim(ylim)
    plt.xlim([-1, 1])
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.xlabel(r"$\xi$", fontsize=label_size)
    if dual:
        plt.ylabel(r"$\tilde{e}_{i}(\xi)$", fontsize=label_size)
    else:
        plt.ylabel(r"$e_{i}(\xi)$", fontsize=label_size)
    plt.show()
    if saveto is not None:
        plt.savefig(saveto, bbox_inches='tight')


# %% functionals
def _derivative_poly_nodes(p, nodes):
    """
    For computation of the derivative at the nodes a more efficient and
    accurate formula can be used, see [1]:

             | \frac{c_{k}}{c_{j}}\frac{1}{x_{k}-x_{j}},          k \neq j
             |
    d_{kj} = <
             | \sum_{l=1,l\neq k}^{p+1}\frac{1}{x_{k}-x_{l}},     k = j
             |

    with
    c_{k} = \prod_{l=1,l\neq k}^{p+1} (x_{k}-x_{l}).

    Parameters
    ----------
    p : int
        degree of polynomial.
    nodes : ndarray
        Lagrange nodes.
        [1] Costa, B., Don, W. S.: On the computation of high order pseudospectral
            derivatives, Applied Numerical Mathematics, vol.33 (1-4), pp. 151-159.

    """
    # compute distances between the nodes
    xi_xj = nodes.reshape(p + 1, 1) - nodes.reshape(1, p + 1)
    # diagonals to one
    xi_xj[np.diag_indices(p + 1)] = 1
    # compute (ci's)
    c_i = np.prod(xi_xj, axis=1)
    # compute ci/cj = ci_cj(i,j)
    c_i_div_cj = np.transpose(c_i.reshape(1, p + 1) / c_i.reshape(p + 1, 1))
    # result formula
    derivative = c_i_div_cj / xi_xj
    # put the diagonals equal to zeros
    derivative[np.diag_indices(p + 1)] = 0
    # compute the diagonal values enforning sum over rows = 0
    derivative[np.diag_indices(p + 1)] = -np.sum(derivative, axis=1)
    return derivative


def _derivative_poly(p, nodes, x):
    """Return the derivatives of the polynomials in the domain x."""
    nodal_derivative = _derivative_poly_nodes(p, nodes)
    polynomials = lagrange_basis(nodes, x)
    #print(np.transpose(nodal_derivative))
    #print(polynomials)
    #print()
    return np.transpose(nodal_derivative) @ polynomials


def _legendre_prime(x, n):
    """Calculate first derivative of the nth Legendre Polynomial recursively.
    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_p (np.array) = value first derivative of L_n.
    """
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x, (int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    legendre_p = (n * legendre(n - 1)(x) - n * x * legendre(n)(x)) / (1 - x ** 2)
    return legendre_p


def _legendre_prime_lobatto(x, n):
    return (1 - x ** 2) ** 2 * _legendre_prime(x, n)


def _legendre_double_prime(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * _legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp * (1 - x ** 2)


def _newton_method(f, dfdx, x_0, n_max, min_error=np.finfo(float).eps * 10):
    """Newton method for rootfinding.

    It garantees quadratic convergence given f'(root) != 0 and abs(f'(Î¾)) < 1
    over the domain considered.

    Args:
        f (obj func) = function
        dfdx (obj func) = derivative of f
        x_0 (float) = starting point
        n_max (int) = max number of iterations
        min_error (float) = min allowed error

    Returns:
        x[-1] (float) = root of f
        x (np.array) = history of convergence
    """
    x = [x_0]
    for i in range(n_max - 1):
        x.append(x[i] - f(x[i]) / dfdx(x[i]))
        if abs(x[i + 1] - x[i]) < min_error: return x[-1]
    print('WARNING : Newton did not converge to machine precision \nRelative error : ',
          x[-1] - x[-2])
    return x[-1]


# %%
if __name__ == "__main__":
    #    def func(x): return np.exp(np.tan(x))
    nodes, weights = lobatto_quad(10)
    h = lagrange_basis(nodes, x=[-0.5, 0.5])
    e = edge_basis(nodes, x=[-0.5, 0.5])
#    integral = np.sum(func(nodes)*weights)
#    plot_lagrange_basis(nodes, dual=False, saveto=None)
#    plot_lagrange_basis(nodes, dual=True, saveto=None)
#    plot_edge_basis(nodes, dual=False, saveto=None)
#    plot_edge_basis(nodes, dual=True, saveto=None)