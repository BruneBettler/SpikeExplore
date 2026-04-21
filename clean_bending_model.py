from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


ArrayLike = np.ndarray


@dataclass(frozen=True)
class BundleParameters:
    """Parameters controlling the bend geometry.

    Attributes
    ----------
    first_bend_distance:
        Distance from the electrode start point to the first bend corner,
        measured along the original straight electrode direction.
    max_bend_angle_deg:
        Maximum absolute bend angle assigned to the electrode farthest from
        the bundle center line. Electrodes closer to the center bend less.
    bundle_ratio:
        Final signed lateral offset divided by the original signed lateral
        offset. For example, 0.05 compresses every endpoint to 5% of its
        original distance from the center line.
    first_radius:
        Radius of the first circular fillet.
    second_radius:
        Radius of the second circular fillet.
    points_per_arc:
        Number of sample points used to draw each circular arc.
    center_tolerance:
        Electrodes closer than this to the center line are treated as unbent.
    """

    first_bend_distance: float = 150.0
    max_bend_angle_deg: float = 45.0
    bundle_ratio: float = 0.05
    first_radius: float = 300.0
    second_radius: float = 300.0
    points_per_arc: int = 100
    center_tolerance: float = 1e-6


@dataclass(frozen=True)
class ElectrodePath:
    """Full geometry of one bent electrode."""

    path: ArrayLike          # (n_points, 2)
    end: ArrayLike           # (2,)
    path_length: float
    bend_angle_deg: float
    middle_segment_length: float


@dataclass(frozen=True)
class BundleResult:
    """Output for the entire electrode bundle."""

    center_start: ArrayLike
    center_stop: ArrayLike
    center_direction: ArrayLike
    center_normal: ArrayLike
    signed_offsets: ArrayLike
    paths: list[ElectrodePath]

    @property
    def ends(self) -> ArrayLike:
        return np.vstack([p.end for p in self.paths])

    @property
    def bend_angles_deg(self) -> ArrayLike:
        return np.array([p.bend_angle_deg for p in self.paths], dtype=float)

    @property
    def middle_segment_lengths(self) -> ArrayLike:
        return np.array([p.middle_segment_length for p in self.paths], dtype=float)


def _as_point(x: Iterable[float]) -> ArrayLike:
    arr = np.asarray(x, dtype=float).reshape(2)
    return arr


def _unit_vector(v: ArrayLike) -> ArrayLike:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / norm


def _rotate(v: ArrayLike, angle_rad: float) -> ArrayLike:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def _sample_line(p0: ArrayLike, p1: ArrayLike, n: int = 50) -> ArrayLike:
    t = np.linspace(0.0, 1.0, n)
    return p0[None, :] + t[:, None] * (p1 - p0)[None, :]


def _make_fillet_arc(
    tangent_in: ArrayLike,
    tangent_out: ArrayLike,
    direction_in: ArrayLike,
    direction_out: ArrayLike,
    radius: float,
    n_points: int,
) -> ArrayLike:
    """Construct the circular arc joining two tangent points.

    The arc is tangent to `direction_in` at `tangent_in` and tangent to
    `direction_out` at `tangent_out`.
    """
    if radius == 0:
        return np.empty((0, 2), dtype=float)

    cross_z = direction_in[0] * direction_out[1] - direction_in[1] * direction_out[0]
    if cross_z > 0:
        normal_in = np.array([-direction_in[1], direction_in[0]], dtype=float)
        normal_out = np.array([-direction_out[1], direction_out[0]], dtype=float)
    else:
        normal_in = np.array([direction_in[1], -direction_in[0]], dtype=float)
        normal_out = np.array([direction_out[1], -direction_out[0]], dtype=float)

    # Solve tangent_in + a * normal_in = tangent_out + b * normal_out
    system = np.column_stack([normal_in, -normal_out])
    rhs = tangent_out - tangent_in
    a, _b = np.linalg.solve(system, rhs)
    center = tangent_in + a * normal_in

    angle_in = np.arctan2(*(tangent_in - center)[::-1])
    angle_out = np.arctan2(*(tangent_out - center)[::-1])

    if cross_z > 0 and angle_out < angle_in:
        angle_out += 2 * np.pi
    elif cross_z <= 0 and angle_out > angle_in:
        angle_out -= 2 * np.pi

    angles = np.linspace(angle_in, angle_out, n_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def bend_electrode_toward_center(
    start: Iterable[float],
    stop: Iterable[float],
    signed_offset: float,
    max_abs_offset: float,
    params: BundleParameters,
) -> ElectrodePath:
    """Bend one electrode toward the bundle center while preserving length.

    This is the clean Python equivalent of the MATLAB function
    `bendTowardCenterSameLength.m`.
    """
    start = _as_point(start)
    stop = _as_point(stop)

    original_length = np.linalg.norm(stop - start)
    if original_length == 0:
        raise ValueError("Electrode start and stop must be different.")

    original_direction = _unit_vector(stop - start)

    if abs(signed_offset) < params.center_tolerance:
        straight = _sample_line(start, stop, n=200)
        return ElectrodePath(
            path=straight,
            end=stop,
            path_length=original_length,
            bend_angle_deg=0.0,
            middle_segment_length=0.0,
        )

    target_offset = params.bundle_ratio * signed_offset
    lateral_shift = target_offset - signed_offset

    bend_angle_mag_deg = params.max_bend_angle_deg * abs(signed_offset) / max_abs_offset
    bend_angle_mag_deg = max(bend_angle_mag_deg, 1e-3)
    bend_angle_rad = np.deg2rad(bend_angle_mag_deg) * np.sign(lateral_shift)

    middle_length = abs(lateral_shift) / max(abs(np.sin(bend_angle_rad)), 1e-12)

    if abs(lateral_shift) < params.center_tolerance or middle_length < params.center_tolerance:
        straight = _sample_line(start, stop, n=200)
        return ElectrodePath(
            path=straight,
            end=stop,
            path_length=original_length,
            bend_angle_deg=0.0,
            middle_segment_length=0.0,
        )

    middle_direction = _unit_vector(_rotate(original_direction, bend_angle_rad))

    corner_1 = start + params.first_bend_distance * original_direction
    corner_2 = corner_1 + middle_length * middle_direction

    phi = abs(bend_angle_rad)
    trim_1 = params.first_radius * np.tan(phi / 2.0)
    trim_2 = params.second_radius * np.tan(phi / 2.0)

    if trim_1 >= params.first_bend_distance:
        raise ValueError("first_radius is too large, or first_bend_distance is too small.")
    if trim_1 + trim_2 >= middle_length:
        raise ValueError(
            "Bend radii are too large for the requested bend. "
            "Increase middle length or reduce radius/angle."
        )

    tangent_1_in = corner_1 - trim_1 * original_direction
    tangent_1_out = corner_1 + trim_1 * middle_direction
    tangent_2_in = corner_2 - trim_2 * middle_direction
    tangent_2_out = corner_2 + trim_2 * original_direction

    arc_length_1 = params.first_radius * phi
    arc_length_2 = params.second_radius * phi
    used_length = (
        np.linalg.norm(tangent_1_in - start)
        + arc_length_1
        + np.linalg.norm(tangent_2_in - tangent_1_out)
        + arc_length_2
    )

    final_straight_length = original_length - used_length
    if final_straight_length < 0:
        raise ValueError(
            "Not enough electrode length for this geometry. "
            "Reduce bend distance, radii, angle, or bundling strength."
        )

    end = tangent_2_out + final_straight_length * original_direction

    line_1 = _sample_line(start, tangent_1_in)
    arc_1 = _make_fillet_arc(
        tangent_1_in,
        tangent_1_out,
        original_direction,
        middle_direction,
        params.first_radius,
        params.points_per_arc,
    )
    line_2 = _sample_line(tangent_1_out, tangent_2_in)
    arc_2 = _make_fillet_arc(
        tangent_2_in,
        tangent_2_out,
        middle_direction,
        original_direction,
        params.second_radius,
        params.points_per_arc,
    )
    line_3 = _sample_line(tangent_2_out, end)

    path = np.vstack([line_1, arc_1, line_2, arc_2, line_3])
    sampled_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    return ElectrodePath(
        path=path,
        end=end,
        path_length=sampled_length,
        bend_angle_deg=float(np.rad2deg(bend_angle_rad)),
        middle_segment_length=float(middle_length),
    )


def build_bundle(
    electrode_starts: ArrayLike,
    electrode_stops: ArrayLike,
    params: BundleParameters = BundleParameters(),
) -> BundleResult:
    """Build the full bent bundle for all electrodes.

    The center line is defined by the mean start point and mean stop point.
    Each electrode is assigned a signed lateral offset relative to that line,
    and outer electrodes are bent more strongly than inner electrodes.
    """
    starts = np.asarray(electrode_starts, dtype=float)
    stops = np.asarray(electrode_stops, dtype=float)

    if starts.shape != stops.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("electrode_starts and electrode_stops must both have shape (N, 2).")

    center_start = starts.mean(axis=0)
    center_stop = stops.mean(axis=0)
    center_direction = _unit_vector(center_stop - center_start)
    center_normal = np.array([-center_direction[1], center_direction[0]], dtype=float)

    signed_offsets = (starts - center_start[None, :]) @ center_normal
    max_abs_offset = float(np.max(np.abs(signed_offsets)))
    if max_abs_offset == 0:
        raise ValueError("All electrode starts lie on the center line; bundling is undefined.")

    paths = [
        bend_electrode_toward_center(
            start=starts[i],
            stop=stops[i],
            signed_offset=float(signed_offsets[i]),
            max_abs_offset=max_abs_offset,
            params=params,
        )
        for i in range(starts.shape[0])
    ]

    return BundleResult(
        center_start=center_start,
        center_stop=center_stop,
        center_direction=center_direction,
        center_normal=center_normal,
        signed_offsets=signed_offsets,
        paths=paths,
    )


def load_electrode_geometry(mat_file: str | Path) -> tuple[ArrayLike, ArrayLike]:
    """Load `Electrode_Start` and `Electrode_Stop` from a MATLAB .mat file."""
    data = loadmat(mat_file)
    try:
        starts = np.asarray(data["Electrode_Start"], dtype=float)
        stops = np.asarray(data["Electrode_Stop"], dtype=float)
    except KeyError as exc:
        raise KeyError(
            "MAT file must contain variables 'Electrode_Start' and 'Electrode_Stop'."
        ) from exc
    return starts, stops


def plot_bundle(
    electrode_starts: ArrayLike,
    electrode_stops: ArrayLike,
    result: BundleResult,
    *,
    ax=None,
):
    """Plot original straight electrodes, bent electrodes, and the center line."""
    import matplotlib.pyplot as plt

    starts = np.asarray(electrode_starts, dtype=float)
    stops = np.asarray(electrode_stops, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    for start, stop, path in zip(starts, stops, result.paths):
        ax.plot([start[0], stop[0]], [start[1], stop[1]], "k--", linewidth=0.6)
        ax.plot(path.path[:, 0], path.path[:, 1], color="tab:blue", linewidth=1.5)

    lengths = np.linalg.norm(stops - starts, axis=1)
    center_line_length = 1.1 * float(np.max(lengths))
    p0 = result.center_start - 0.1 * center_line_length * result.center_direction
    p1 = result.center_start + 1.0 * center_line_length * result.center_direction
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="tab:red", linewidth=2.0)

    ax.plot(starts[:, 0], starts[:, 1], "ko", markersize=4)
    ax.plot(result.ends[:, 0], result.ends[:, 1], "go", markersize=4)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Bundle formation with finite-radius bends")
    return ax


def run_from_mat_file(
    mat_file: str | Path,
    params: BundleParameters = BundleParameters(),
    *,
    make_plot: bool = True,
) -> BundleResult:
    """Convenience wrapper matching the MATLAB workflow."""
    starts, stops = load_electrode_geometry(mat_file)
    result = build_bundle(starts, stops, params)

    if make_plot:
        plot_bundle(starts, stops, result)

    return result

def run_from_array(
    electrode_starts: np.ndarray,
    electrode_stops: np.ndarray,
    params: BundleParameters = BundleParameters(),
    *,
    make_plot: bool = True,
) -> BundleResult:
    
    result = build_bundle(electrode_starts, electrode_stops, params)

    if make_plot:
        plot_bundle(electrode_starts, electrode_stops, result)

    return result


if __name__ == "__main__":
    mat_path = Path("matlab.mat")
    result = run_from_mat_file(mat_path, make_plot=True)

    np.set_printoptions(precision=4, suppress=True)
    print("New bent endpoints:")
    print(result.ends)
    print("\nUsed bend angles (deg):")
    print(result.bend_angles_deg)
    print("\nUsed middle lengths:")
    print(result.middle_segment_lengths)
