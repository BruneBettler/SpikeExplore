import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

def create_flat_geom(contact_site_num, x_flat_contact_dist, y_flat_contact_dist):
    total_flat_width = (contact_site_num-1) * x_flat_contact_dist
    x_positions = np.empty(contact_site_num)
    x_positions[0] = total_flat_width
    x_positions[1::2] = np.arange((contact_site_num) // 2) * x_flat_contact_dist
    x_positions[2::2] = total_flat_width - (np.arange((contact_site_num - 1) // 2) + 1) * x_flat_contact_dist
    
    y_positions = (np.arange(contact_site_num) * y_flat_contact_dist)[::-1]
    return np.column_stack((x_positions, y_positions))

def S(t):
    # Cubic smoothstep function
    return 3.0 * t**2 - 2.0 * t**3


def dS(t):
    # Derivative of the cubic smoothstep:
    return 6.0 * t * (1.0 - t)

# TODO: make sure x0_array is centered around 0 !!! 
def bundled_x(x0_array, final_bundle_width, initial_width=None):
    """
    Linearly compress x-positions toward the center.

    Parameters
    ----------
    x0_array : array-like
        Original x positions of wires.
    final_bundle_width : float
        Desired final total width after bundling.
    initial_width : float or None
        Original total width. If None, infer as 2 * max(abs(x0_array)).

    Returns
    -------
    xf : ndarray
        Final x positions.
    """
    x0_array = np.asarray(x0_array, dtype=float)

    if initial_width is None:
        initial_width = 2.0 * np.max(np.abs(x0_array))

    scale = final_bundle_width / initial_width
    return scale * x0_array

def wire_arc_length(H, dx):
    """
    Arc length of one wire bend for:
        x(t) = x0 + dx * S(t)
        y(t) = y_start + H * t

    Since:
        dx/dt = dx * dS(t)
        dy/dt = H

    the arc length is:
        L(H) = integral_0^1 sqrt((dx * dS(t))^2 + H^2) dt
    """
    integrand = lambda t: np.sqrt((dx * dS(t))**2 + H**2)
    L, _ = quad(integrand, 0.0, 1.0)
    return L

def solve_H(length, dx):
    """
    Solve for H such that the wire keeps its original length.

    Parameters
    ----------
    length : float
        Original wire length.
    dx : float
        Total horizontal displacement.

    Returns
    -------
    H : float
        Final vertical depth of the tip.
    """
    length = float(length)
    dx = float(dx)

    if length < 0:
        raise ValueError("length must be nonnegative.")

    # Minimum possible arc length occurs at H = 0
    min_length = wire_arc_length(0.0, dx)
    if min_length > length:
        raise ValueError(
            f"Wire too short for this displacement: "
            f"minimum possible length is {min_length:.6f}, "
            f"but given length is {length:.6f}."
        )

    # If dx = 0, wire stays vertical
    if np.isclose(dx, 0.0):
        return length

    # Solve L(H) - length = 0 on [0, length]
    f = lambda H: wire_arc_length(H, dx) - length
    return brentq(f, 0.0, length)

def bundle_tip_positions(x0, length, final_bundle_width, initial_width=None):
    """
    Compute final tip positions for a bundle of wires.

    Parameters
    ----------
    x0 : array-like
        Original x positions.
    length : float or array-like
        Original wire lengths.
    final_bundle_width : float
        Desired final total width after bundling.
    initial_width : float or None
        Original total width. If None, infer from x0.

    Returns
    -------
    xf : ndarray
        Final x tip positions.
    yf : ndarray
        Final y tip positions, assuming the top/fixed point is at y = 0.
    """
    x0 = np.asarray(x0, dtype=float)
    length = np.asarray(length, dtype=float)

    if length.ndim == 0:
        length = np.full_like(x0, float(length))
    elif length.shape != x0.shape:
        raise ValueError("length must be a scalar or have the same shape as x0.")

    xf = bundled_x(x0, final_bundle_width, initial_width)
    dx = xf - x0

    yf = np.array([solve_H(L, dxi) for L, dxi in zip(length, dx)])
    return xf, yf


def wire_curve(x0, length, xf, n=300, y_start=0.0):
    """
    Return sampled points along one wire curve.

    Parameters
    ----------
    x0 : float
        Original x position.
    length : float
        Original wire length.
    xf : float
        Final x position.
    n : int
        Number of sample points.
    y_start : float
        Vertical offset of the top/fixed end.

    Returns
    -------
    x, y : ndarray
        Sampled curve coordinates.
    """
    dx = xf - x0
    H = solve_H(length, dx)

    t = np.linspace(0.0, 1.0, n)
    x = x0 + dx * S(t)
    y = y_start + H * t
    return x, y


def plot_bundle(
    x0,
    length,
    final_bundle_width,
    initial_width=None,
    y_start=0.0,
    show_original=True,
    n_curve=300,
    ax=None,
):
    """
    Plot the bundled wire set.

    Parameters
    ----------
    x0 : array-like
        Original x positions.
    length : float or array-like
        Original wire lengths.
    final_bundle_width : float
        Desired final total width after bundling.
    initial_width : float or None
        Original total width.
    y_start : float
        Vertical offset of the fixed ends.
    show_original : bool
        Whether to show the original straight wires.
    n_curve : int
        Number of sample points per wire.
    ax : matplotlib axis or None
        Existing axis to draw on.

    Returns
    -------
    fig, ax, xf, yf
    """
    x0 = np.asarray(x0, dtype=float)
    length = np.asarray(length, dtype=float)

    if length.ndim == 0:
        length = np.full_like(x0, float(length))

    xf, yf = bundle_tip_positions(
        x0=x0,
        length=length,
        final_bundle_width=final_bundle_width,
        initial_width=initial_width,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    if show_original:
        for xi, Li in zip(x0, length):
            ax.plot([xi, xi], [y_start, y_start + Li], "--", alpha=0.35)

    for xi, Li, xfi in zip(x0, length, xf):
        x, y = wire_curve(xi, Li, xfi, n=n_curve, y_start=y_start)
        ax.plot(x, y, linewidth=2)

    ax.scatter(xf, y_start + yf, s=30)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bundled wire geometry")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig, ax, xf, yf