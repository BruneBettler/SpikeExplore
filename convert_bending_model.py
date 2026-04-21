from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


@dataclass
class BendResult:
    x_path: np.ndarray
    y_path: np.ndarray
    pend: np.ndarray
    path_length: float
    theta_used_deg: float
    lmid: float


@dataclass
class BundleRunResult:
    x_paths: List[np.ndarray]
    y_paths: List[np.ndarray]
    pend: np.ndarray
    lout: np.ndarray
    theta_used_deg: np.ndarray
    lmid_used: np.ndarray
    center_start: np.ndarray
    center_stop: np.ndarray
    u_center: np.ndarray
    n_center: np.ndarray


def make_arc(
    tin: np.ndarray,
    tout: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    radius: float,
    n_arc: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful translation of MATLAB makeArc()."""
    if radius == 0:
        return np.array([]), np.array([])

    tin = np.asarray(tin, dtype=float)
    tout = np.asarray(tout, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    crossz = d1[0] * d2[1] - d1[1] * d2[0]

    if crossz > 0:
        n1 = np.array([-d1[1], d1[0]], dtype=float)
        n2 = np.array([-d2[1], d2[0]], dtype=float)
    else:
        n1 = np.array([d1[1], -d1[0]], dtype=float)
        n2 = np.array([d2[1], -d2[0]], dtype=float)

    a = np.column_stack([n1, -n2])
    rhs = tout - tin
    ab = np.linalg.solve(a, rhs)
    center = tin + ab[0] * n1

    a1 = np.arctan2(tin[1] - center[1], tin[0] - center[0])
    a2 = np.arctan2(tout[1] - center[1], tout[0] - center[0])

    if crossz > 0:
        if a2 < a1:
            a2 += 2 * np.pi
    else:
        if a2 > a1:
            a2 -= 2 * np.pi

    ang = np.linspace(a1, a2, n_arc)
    x_arc = center[0] + radius * np.cos(ang)
    y_arc = center[1] + radius * np.sin(ang)
    return x_arc, y_arc


def bend_toward_center_same_length(
    pstart: np.ndarray,
    pstop0: np.ndarray,
    cstart: np.ndarray,
    u_center: np.ndarray,
    n_center: np.ndarray,
    d0: float,
    max_abs_offset: float,
    s1: float,
    max_theta_deg: float,
    bundle_ratio: float,
    r1: float,
    r2: float,
    n_arc: int,
    center_tol: float = 1e-6,
) -> BendResult:
    """
    Translation of bendTowardCenterSameLength.m.

    Parameters match the MATLAB function names closely for readability.
    """
    pstart = np.asarray(pstart, dtype=float)
    pstop0 = np.asarray(pstop0, dtype=float)
    cstart = np.asarray(cstart, dtype=float)
    u_center = np.asarray(u_center, dtype=float)
    n_center = np.asarray(n_center, dtype=float)

    l0 = np.linalg.norm(pstop0 - pstart)
    if l0 == 0:
        raise ValueError("Pstart and Pstop0 must be different.")

    # Original direction of this electrode
    u = (pstop0 - pstart) / l0

    # If very close to center line: no bend
    if abs(d0) < center_tol:
        x_path = np.linspace(pstart[0], pstop0[0], 200)
        y_path = np.linspace(pstart[1], pstop0[1], 200)
        return BendResult(
            x_path=x_path,
            y_path=y_path,
            pend=pstop0.copy(),
            path_length=float(l0),
            theta_used_deg=0.0,
            lmid=0.0,
        )

    # Target final offset
    d_end = bundle_ratio * d0

    # Required lateral shift toward center
    delta_d = d_end - d0

    # Bend angle magnitude grows with distance from center
    theta_mag_deg = max_theta_deg * abs(d0) / max_abs_offset
    theta_mag_deg = max(theta_mag_deg, 1e-3)
    theta_mag = np.deg2rad(theta_mag_deg)

    # Sign of bend: toward center line
    theta = np.sign(delta_d) * theta_mag

    # Middle segment length needed to achieve target offset
    lmid = abs(delta_d) / max(abs(np.sin(theta)), 1e-12)

    # If target shift is tiny, no bend
    if abs(delta_d) < center_tol or lmid < center_tol:
        x_path = np.linspace(pstart[0], pstop0[0], 200)
        y_path = np.linspace(pstart[1], pstop0[1], 200)
        return BendResult(
            x_path=x_path,
            y_path=y_path,
            pend=pstop0.copy(),
            path_length=float(l0),
            theta_used_deg=0.0,
            lmid=0.0,
        )

    # Directions
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    u_mid = rot @ u
    u_mid = u_mid / np.linalg.norm(u_mid)
    u_end = u.copy()

    # Sharp bend locations
    v1 = pstart + s1 * u
    v2 = v1 + lmid * u_mid

    # Bend angles at the two corners
    phi1 = abs(theta)
    phi2 = abs(theta)

    # Fillet trimming distances
    t1 = r1 * np.tan(phi1 / 2)
    t2 = r2 * np.tan(phi2 / 2)

    if t1 >= s1:
        raise ValueError("R1 too large or s1 too small.")
    if (t1 + t2) >= lmid:
        raise ValueError(
            "R1/R2 too large for this bend. Increase Lmid or reduce radii/angle."
        )

    # Tangency points
    t1_in = v1 - t1 * u
    t1_out = v1 + t1 * u_mid

    t2_in = v2 - t2 * u_mid
    t2_out = v2 + t2 * u_end

    # Arc lengths
    larc1 = r1 * phi1
    larc2 = r2 * phi2

    # Length used before final straight
    lused = (
        np.linalg.norm(t1_in - pstart)
        + larc1
        + np.linalg.norm(t2_in - t1_out)
        + larc2
    )

    # Remaining final straight
    llast = l0 - lused
    if llast < 0:
        raise ValueError(
            "Not enough total length for this electrode. "
            "Reduce s1, angle, radii, or bundle strength."
        )

    # Final endpoint
    pend = t2_out + llast * u_end

    # Build arcs
    x_arc1, y_arc1 = make_arc(t1_in, t1_out, u, u_mid, r1, n_arc)
    x_arc2, y_arc2 = make_arc(t2_in, t2_out, u_mid, u_end, r2, n_arc)

    # Straights
    n_line = 50
    x1 = np.linspace(pstart[0], t1_in[0], n_line)
    y1 = np.linspace(pstart[1], t1_in[1], n_line)

    x2 = np.linspace(t1_out[0], t2_in[0], n_line)
    y2 = np.linspace(t1_out[1], t2_in[1], n_line)

    x3 = np.linspace(t2_out[0], pend[0], n_line)
    y3 = np.linspace(t2_out[1], pend[1], n_line)

    x_path = np.concatenate([x1, x_arc1, x2, x_arc2, x3])
    y_path = np.concatenate([y1, y_arc1, y2, y_arc2, y3])

    path_length = float(np.sum(np.sqrt(np.diff(x_path) ** 2 + np.diff(y_path) ** 2)))
    theta_used_deg = float(np.rad2deg(theta))

    return BendResult(
        x_path=x_path,
        y_path=y_path,
        pend=pend,
        path_length=path_length,
        theta_used_deg=theta_used_deg,
        lmid=float(lmid),
    )


def _extract_mat_array(data: dict, name: str) -> np.ndarray:
    if name not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"'{name}' not found in MAT file. Available keys: {available}")
    arr = np.asarray(data[name], dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2); got {arr.shape}")
    return arr


def run_bundle_model(
    electrode_start: np.ndarray,
    electrode_stop: np.ndarray,
    *,
    s1: float = 150.0,
    max_theta_deg: float = 45.0,
    bundle_ratio: float = 0.05,
    r1: float = 300.0,
    r2: float = 300.0,
    n_arc: int = 100,
    center_tol: float = 1e-6,
    make_plot: bool = True,
) -> BundleRunResult:
    """Translation of Final1.m into a reusable Python function."""
    electrode_start = np.asarray(electrode_start, dtype=float)
    electrode_stop = np.asarray(electrode_stop, dtype=float)

    if electrode_start.shape != electrode_stop.shape or electrode_start.ndim != 2 or electrode_start.shape[1] != 2:
        raise ValueError(
            "Electrode_Start and Electrode_Stop must both have shape (N, 2)."
        )

    n_electrodes = electrode_start.shape[0]

    # Center line from mean start to mean stop
    cstart = electrode_start.mean(axis=0)
    cstop = electrode_stop.mean(axis=0)

    u = cstop - cstart
    lu = np.linalg.norm(u)
    if lu == 0:
        raise ValueError("Center line cannot be determined.")
    u = u / lu

    # 2D normal vector
    n = np.array([-u[1], u[0]], dtype=float)

    # Signed offsets of each electrode from center line
    d0 = np.zeros(n_electrodes, dtype=float)
    l0 = np.zeros(n_electrodes, dtype=float)
    for i in range(n_electrodes):
        d0[i] = np.dot(electrode_start[i] - cstart, n)
        l0[i] = np.linalg.norm(electrode_stop[i] - electrode_start[i])

    max_abs_offset = float(np.max(np.abs(d0)))
    if max_abs_offset == 0:
        raise ValueError("All start points lie on the same center line.")

    # Outputs
    x_paths: List[np.ndarray] = [None] * n_electrodes  # type: ignore[list-item]
    y_paths: List[np.ndarray] = [None] * n_electrodes  # type: ignore[list-item]
    pend = np.zeros((n_electrodes, 2), dtype=float)
    lout = np.zeros(n_electrodes, dtype=float)
    theta_used_deg = np.zeros(n_electrodes, dtype=float)
    lmid_used = np.zeros(n_electrodes, dtype=float)

    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(n_electrodes):
        pstart = electrode_start[i]
        pstop0 = electrode_stop[i]

        result = bend_toward_center_same_length(
            pstart,
            pstop0,
            cstart,
            u,
            n,
            d0[i],
            max_abs_offset,
            s1,
            max_theta_deg,
            bundle_ratio,
            r1,
            r2,
            n_arc,
            center_tol,
        )

        x_paths[i] = result.x_path
        y_paths[i] = result.y_path
        pend[i] = result.pend
        lout[i] = result.path_length
        theta_used_deg[i] = result.theta_used_deg
        lmid_used[i] = result.lmid

        if make_plot:
            # original straight line
            ax.plot(
                [pstart[0], pstop0[0]],
                [pstart[1], pstop0[1]],
                "k--",
                linewidth=0.5,
            )
            # bent line
            ax.plot(result.x_path, result.y_path, "b", linewidth=1.5)

    if make_plot:
        center_line_len = float(np.max(l0) * 1.1)
        pcen1 = cstart - 0.1 * center_line_len * u
        pcen2 = cstart + 1.0 * center_line_len * u
        ax.plot([pcen1[0], pcen2[0]], [pcen1[1], pcen2[1]], "r-", linewidth=2)

        ax.plot(
            electrode_start[:, 0],
            electrode_start[:, 1],
            "ko",
            markerfacecolor="k",
            markersize=4,
        )
        ax.plot(
            pend[:, 0],
            pend[:, 1],
            "go",
            markerfacecolor="g",
            markersize=4,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(
            "Bundle formation: outer electrodes bend more, center electrode least"
        )
        ax.legend(
            [
                "Original straight",
                "Bent electrode",
                "Center line",
                "Starts",
                "New ends",
            ]
        )
        plt.show()

    return BundleRunResult(
        x_paths=x_paths,
        y_paths=y_paths,
        pend=pend,
        lout=lout,
        theta_used_deg=theta_used_deg,
        lmid_used=lmid_used,
        center_start=cstart,
        center_stop=cstop,
        u_center=u,
        n_center=n,
    )


def run_from_mat_file(
    mat_path: str | Path,
    *,
    s1: float = 150.0,
    max_theta_deg: float = 45.0,
    bundle_ratio: float = 0.05,
    r1: float = 300.0,
    r2: float = 300.0,
    n_arc: int = 100,
    center_tol: float = 1e-6,
    make_plot: bool = True,
) -> BundleRunResult:
    data = loadmat(mat_path)
    electrode_start = _extract_mat_array(data, "Electrode_Start")
    electrode_stop = _extract_mat_array(data, "Electrode_Stop")

    result = run_bundle_model(
        electrode_start,
        electrode_stop,
        s1=s1,
        max_theta_deg=max_theta_deg,
        bundle_ratio=bundle_ratio,
        r1=r1,
        r2=r2,
        n_arc=n_arc,
        center_tol=center_tol,
        make_plot=make_plot,
    )

    print("New bent endpoints:")
    print(result.pend)
    print("\nUsed bend angles (deg):")
    print(result.theta_used_deg)
    print("\nUsed middle lengths:")
    print(result.lmid_used)
    return result


if __name__ == "__main__":
    default_mat = Path("matlab.mat")
    if not default_mat.exists():
        raise FileNotFoundError(
            "Could not find 'matlab.mat' in the current working directory. "
            "Either place the MAT file there or import run_from_mat_file() manually."
        )

    run_from_mat_file(default_mat)
