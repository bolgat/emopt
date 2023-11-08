import numpy as np
import math
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"

NONZERO = 1e-16
SQRT2 = np.sqrt(2.0)
ISQRT2 = np.sqrt(2.0)/2.0

##############################################################
# Nonlinear Function Definitions, sigma_k(x)
##############################################################
def nl_sig(k: float, x: torch.tensor) -> torch.tensor:
    """Sigmoid nonlinear function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
    x : torch.tensor
        Tensor of values to bound.

    Returns
    -------
    torch.tensor
        Bounded values.
    """
    return torch.sigmoid(k*x)

def nl_erf(k: float, x: torch.tensor) -> torch.tensor:
    """Error function nonlinear function (scaled to [0,1] bounds).

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
    x : torch.tensor
        Tensor of values to bound.

    Returns
    -------
    torch.tensor
        Bounded values.
    """
    return 0.5 * (1.0 + torch.erf(k*x))

def nl_lin(k: float, x: torch.tensor) -> torch.tensor:
    """Piecewise linear nonlinear function (implements a [0,1] bounded clamp).

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
    x : torch.tensor
        Tensor of values to bound.

    Returns
    -------
    torch.tensor
        Bounded values.
    """
    return torch.clamp(k*x + 0.5, min=0.0, max=1.0)

def nl_sin(k: float, x: torch.tensor) -> torch.tensor:
    """Piecewise sine nonlinear function (normalized to [0,1] bounds).

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
    x : torch.tensor
        Tensor of values to bound.

    Returns
    -------
    torch.tensor
        Bounded values.
    """
    kx = k*x
    retval = torch.where(kx > np.pi/2.0,
                         1.0,
                         torch.where(kx < -np.pi/2.0,
                                     0.0,
                                     0.5*(torch.sin(kx)+1.0)
                                    )
                        )
    return retval

def nl_quad(k: float, x: torch.tensor) -> torch.tensor:
    """Piecewise quadratic nonlinear function (normalized to [0,1] bounds).

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
    x : torch.tensor
        Tensor of values to bound.

    Returns
    -------
    torch.tensor
        Bounded values.
    """
    kx = k*x
    retval = torch.where(kx < -ISQRT2,
                         0.0,
                         torch.where(kx < 0,
                                     (ISQRT2 + kx)**2,
                                     torch.where(kx > ISQRT2,
                                                 1.0,
                                                 1.0 - (ISQRT2 - kx)**2),
                                    )
                        )
    return retval

##############################################################
# Differentiable logic operations
##############################################################

def union(list_shapes: list) -> torch.tensor:
    """Differentiable union function (main type).

    Parameters
    ----------
    list_shapes : list
        List of torch.tensors. Each must have the same shape.
        Each must have [0,1] bounded values for correct
        functionality.

    Returns
    -------
    torch.tensor
       Tensor of same shape as one element of list_shapes.
    """
    return torch.clamp(sum(list_shapes), min=0.0, max=1.0)

def union_b(list_shapes: list) -> torch.tensor:
    """Differentiable union function (type b).

    Parameters
    ----------
    list_shapes : list
        List of torch.tensors. Each must have the same shape.
        Each must have [0,1] bounded values for correct
        functionality.

    Returns
    -------
    torch.tensor
       Tensor of same shape as one element of list_shapes.
    """
    if len(list_shapes) == 1:
        return list_shapes[0]
    else:
        ub = union_b(list_shapes[:-1])
        return list_shapes[-1] + ub - list_shapes[-1] * ub

def union_c(k: float,
            list_shapes: list,
            nl: callable = nl_sig) -> torch.tensor:
    """Differentiable union function (type c).

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    list_shapes : list
        List of torch.tensors. Each must have the same shape.
        Each must have [0,1] bounded values for correct
        functionality.
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor of same shape as one element of list_shapes.
    """
    return nl(k, sum(list_shapes) - 0.5)

def intersection(list_shapes: list) -> torch.tensor:
    """Differentiable intersection function (main type).

    Parameters
    ----------
    list_shapes : list
        List of torch.tensors. Each must have the same shape.
        Each must have [0,1] bounded values for correct
        functionality.

    Returns
    -------
    torch.tensor
       Tensor of same shape as one element of list_shapes.
    """
    N = len(list_shapes)
    return torch.clamp(sum(list_shapes), min=N-1., max=N) - (N-1.)

def intersection_b(list_shapes: list) -> torch.tensor:
    """Differentiable intersection function (type b).

    Parameters
    ----------
    list_shapes : list
        List of torch.tensors. Each must have the same shape.
        Each must have [0,1] bounded values for correct
        functionality.

    Returns
    -------
    torch.tensor
       Tensor of same shape as one element of list_shapes.
    """
    return math.prod(list_shapes)

def intersection_c(k: float,
                   list_shapes: list,
                   nl: callable = nl_sig) -> torch.tensor:
    """Differentiable intersection function (type c).

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    list_shapes : list
        List of torch.tensors. Each must have the same shape.
        Each must have [0,1] bounded values for correct
        functionality.
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor of same shape as one element of list_shapes.
    """
    N = len(list_shapes)
    return nl(k, sum(list_shapes) - (N - 0.5))

##############################################################
# AutoDiff-Compatible Shape Definitions (AutoDiffGeo)
##############################################################

# Note below that x, y, z are simple 1d tensors,
# Can be generated by torch.linspace (no torch.meshgrid is required)

def step1d(k: float,
        x: torch.tensor,
        v: torch.tensor,
        reverse: bool = False,
        nl: callable = nl_sig) -> torch.tensor:
    """One-dimensional step function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of coordinates. By default we assume that
        len(x.shape) = 1.
    v : torch.tensor
        Step parameter. v is x_0, the coordinate where
        step transition takes place.
    reverse: bool
        Defines the direction of the step transition.
        Defaults to False (output values go from 0 -> 1 as
        x coordinate is increased).
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor of same shape as x.shape.
       Will have [0,1] bounded values with transition about v.
    """
    if reverse:
        return nl(k, v-x)
    else:
        return nl(k, x-v)

def rect1d(k: float,
        x: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """One-dimensional rect (rectangle) function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of coordinates. By default we assume that
        len(x.shape) = 1.
    v : torch.tensor
        v[0] = xmin.
        v[1] = xmax.
        Defines locations of step transitions in 1D
        (rectangle boundaries).
        In particular, x values with v[0] < x < v[1]
        will output value of > 0.5 (bounded by 1).
        Otherwise < 0.5 (bounded by 0).
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor of same shape as x.shape.
       Will have [0,1] bounded values with transitions
       about v[0] and v[1].
    """
    return nl(k, x-v[0]) * nl(k, v[1]-x)

def rect2d(k: float,
        x: torch.tensor,
        y: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """Two-dimensional rect (rectangle) function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of x coordinates. By default we assume that
        len(x.shape) = 1.
    y : torch.tensor
        Tensor of y coordinates. By default we assume that
        len(y.shape) = 1.
    v : torch.tensor
        v[0] = xmin.
        v[1] = xmax.
        v[2] = ymin.
        v[3] = ymax.
        Defines coordinates of step transitions in 2D
        (rectangle boundaries).
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (y.shape[0], x.shape[0]).
       Will have [0,1] bounded values with x transitions about
       v[0], v[1] and y transitions about v[2], v[3].
    """
    return rect1d(k, x, v[:2], nl=nl).view(1,-1) * \
           rect1d(k, y, v[2:], nl=nl).view(-1,1)

def rect3d(k: float,
        x: torch.tensor,
        y: torch.tensor,
        z: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """Three-dimensional rect (cuboid) function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of x coordinates. By default we assume that
        len(x.shape) = 1.
    y : torch.tensor
        Tensor of y coordinates. By default we assume that
        len(y.shape) = 1.
    z : torch.tensor
        Tensor of z coordinates. By default we assume that
        len(y.shape) = 1.
    v : torch.tensor
        v[0] = xmin.
        v[1] = xmax.
        v[2] = ymin.
        v[3] = ymax.
        v[4] = zmin.
        v[5] = zmax.
        Defines coordinates of step transitions in 3D
        (cuboid boundaries).
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (z.shape[0], y.shape[0], x.shape[0]).
       Will have [0,1] bounded values with x transitions about
       v[0], v[1] and y transitions about v[2], v[3] and z
       transitions about v[4], v[5].
    """
    return rect1d(k, x, v[:2], nl=nl).view(1,1,-1) * \
           rect1d(k, y, v[2:4], nl=nl).view(1,-1,1) * \
           rect1d(k, z, v[4:], nl=nl).view(-1,1,1)

def rectnd(k: float,
        list_coords: list,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """n-dimensional rect (cuboid) function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    list_coords : list
        list of torch.tensor coordinates, where each tensor is 1D.
    v : torch.tensor
        v.shape[0] = len(list_coords)
        v.shape[1] = 2
        v[i,0] = min in i'th coordinate
        v[i,1] = max in i'th coordinate
        Defines coordinates of step transitions in n-D
        (cuboid boundaries).
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (..., list_shapes[i].shape[0], ...)
       for i in range(len(list_shapes)).
       Will have [0,1] bounded values with i'th transitions about
       v[i,0] and v[i,1].
       Will have same shape as rect3d when
       list_coords = [z, y, x] for args x, y, z.
    """
    n = len(list_coords)
    view = n*[1]
    shape = torch.as_tensor([1.0]).view(view)
    for count, r in enumerate(list_coords):
        view[count] = -1
        shape = shape * rect1d(k, r, v[count,:], nl=nl).view(view)
        view[count] = 1
    return shape

def depth(shape: torch.tensor,
        k: float,
        r: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """Convenience function to extrude nD shape to (n+1)D shape.

    Parameters
    ----------
    shape: torch.tensor
        Tensor with len(shape.shape) = n.
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    r : torch.tensor
        Tensor of new r coordinates for extrusion.
        By default we assume that len(r.shape) = 1.
    v : torch.tensor
        v[0]=rmin
        v[1]=rmax
        Defines coordinates of step transitions in the new
        dimension (cuboid boundaries).
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
        (n+1)D tensor of extruded shape with depth defined by
        v[0],v[1].
        By default, output.shape = (r.shape[0], *shape.shape)
    """
    n = len(shape.shape)
    view = (n+1)*[1]
    view[0] = -1
    return rect1d(k, r, v, nl=nl).view(view) * shape.unsqueeze(0)

def step2d(k: float,
        x: torch.tensor,
        y: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """Two-dimensional step function.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of x coordinates. By default we assume that
        len(x.shape) = 1.
    y : torch.tensor
        Tensor of y coordinates. By default we assume that
        len(y.shape) = 1.
    v : torch.tensor
        v[0] = nx (normal in x direction)
        v[1] = ny (normal in y direction)
        v[2] = x0 (x-coordinate on transition boundary)
        v[3] = y0 (y-coordinate on transition boundary)
        Note that this function will *NOT* unit-normalize
        nx and ny. This behavior may be desired in situations
        where grid discretization is not uniform. If using
        uniform grid discretization, you should normalize
        v[0],v[1] using:
            v[:2] = v[:2]/v[:2].norm(dim=-1, keepdim=True)
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (y.shape[0], x.shape[0]).
       Will have [0,1] bounded values with transition defined by
       the normal direction and coordinate provided in v.
    """
    #vn = v[:2]/v[:2].norm(dim=-1, keepdim=True)
    vn = v[:2]
    shape = nl(k, (vn[0]*(x-v[2])).view(1,-1) + (vn[1]*(y-v[3])).view(-1,1))
    return shape


def poly2d(k: float,
        x: torch.tensor,
        y: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """Two-dimensional polygon function.
       ***Only supports convex polygons***

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of x coordinates. By default we assume that
        len(x.shape) = 1.
    y : torch.tensor
        Tensor of y coordinates. By default we assume that
        len(y.shape) = 1.
    v : torch.tensor
        v[i,0] = x coordinate of i'th vertex
        v[i,1] = y coordinate of i'th vertex
        v.shape[0] = n, can be arbitrarily large.
        ***Must be oriented clockwise***
        ***Only supports convex polygons***
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (y.shape[0], x.shape[0]).
       Will have [0,1] bounded values with boundaries defined by
       the vertices.
    """
    dxs = torch.diff(v, append=v[0:1,:], dim=0)
    cents = 0.5*dxs + v
    nvec = torch.cat([dxs[:,1:2], -dxs[:,0:1]], dim=1)
    nvec = nvec/nvec.norm(dim=-1, keepdim=True)
    return nl(k, nvec[:,0] * (x.view(1,-1,1) - cents[:,0]) + \
                 nvec[:,1] * (y.view(-1,1,1) - cents[:,1])).prod(-1)

def circ2d(k: float,
        x: torch.tensor,
        y: torch.tensor,
        v: torch.tensor,
        nl: callable = nl_sig) -> torch.tensor:
    """Two-dimensional circle.

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of x coordinates. By default we assume that
        len(x.shape) = 1.
    y : torch.tensor
        Tensor of y coordinates. By default we assume that
        len(y.shape) = 1.
    v : torch.tensor
        v[0] = circle radius
        v[1] = x-coordinate of center
        v[2] = y-coordinate of center
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (y.shape[0], x.shape[0]).
       Will have [0,1] bounded values with circular boundary.
    """
    R = torch.sqrt(NONZERO + ((x-v[1])**2).view(1,-1) + ((y-v[2])**2).view(-1,1))
    return nl(k, v[0] - R)

def polar2d(k: float,
        x: torch.tensor,
        y: torch.tensor,
        v: torch.tensor,
        order: int = 4,
        nl: callable = nl_sig) -> torch.tensor:
    """Two-dimensional star-convex function, with sinusoidal
    boundary. r(theta) = radius * (1.0 + delta * cos(ord*theta))

    Parameters
    ----------
    k : float
        Inverse length scalar to define step transition slope.
        Used in nl function.
    x : torch.tensor
        Tensor of x coordinates. By default we assume that
        len(x.shape) = 1.
    y : torch.tensor
        Tensor of y coordinates. By default we assume that
        len(y.shape) = 1.
    v : torch.tensor
        v[0] = circle radius
        v[1] = cosine perturbation amplitude
        v[2] = rotation (in radians)
        v[3] = x-coordinate of center
        v[4] = y-coordinate of center
    order : int
        order of cosine perturbations
    nl : callable
        A nonlinear function that takes arguments nl(k, x),
        where k is scalar, x is torch.tensor.
        Defaults to emopt.experimental._grid.nl_sig

    Returns
    -------
    torch.tensor
       Tensor with shape = (y.shape[0], x.shape[0]).
       Will have [0,1] bounded values with sinusoidal boundary.
    """
    theta = v[2] + torch.arctan2((x-v[3]).view(1,-1), (y-v[4]).view(-1,1))
    R = torch.sqrt(NONZERO + ((x-v[3])**2).view(1,-1) + ((y-v[4])**2).view(-1,1))
    return nl(k, v[0] * (1.0 + v[1] * torch.cos(ord*theta)) - R)

if __name__=='__main__':
    pass
