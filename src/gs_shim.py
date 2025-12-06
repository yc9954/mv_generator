"""
Lightweight Gaussian Splatting shim generator using Open3D.

Creates a simple background plane pointcloud for DEBUG_SHIM mode,
avoiding the need for a full 4DGS build during development/testing.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def generate_bg_plane(
    root: str,
    grid_size: Tuple[int, int] = (600, 300),
    z_depth: float = 4.0,
    output_name: str = "bg_plane.ply"
) -> str:
    """
    Generate a simple background plane pointcloud as PLY file.

    Creates a grid of 3D points representing a background plane,
    suitable for lightweight demo/testing without full 4DGS.

    Args:
        root: Root directory path
        grid_size: (width, height) number of points in grid
        z_depth: Z-coordinate depth for the plane
        output_name: Output PLY filename

    Returns:
        Full path to generated PLY file
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for gs_shim. Install with: pip install open3d"
        )

    output_path = Path(root) / "gs_shim" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate grid of points
    width, height = grid_size
    x = np.linspace(-2.0, 2.0, width)
    y = np.linspace(-1.0, 1.0, height)
    xv, yv = np.meshgrid(x, y)

    # Flatten to get point positions
    points = np.stack([
        xv.flatten(),
        yv.flatten(),
        np.full(xv.size, z_depth)
    ], axis=-1)

    # Generate colors (simple gradient for visual interest)
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = (xv.flatten() + 2.0) / 4.0  # Red gradient
    colors[:, 1] = (yv.flatten() + 1.0) / 2.0  # Green gradient
    colors[:, 2] = 0.5  # Constant blue

    # Clip colors to valid range
    colors = np.clip(colors, 0.0, 1.0)

    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save as PLY
    o3d.io.write_point_cloud(str(output_path), pcd)

    return str(output_path)


def generate_bg_plane_with_depth_variation(
    root: str,
    grid_size: Tuple[int, int] = (600, 300),
    z_base: float = 4.0,
    z_variation: float = 0.5,
    output_name: str = "bg_plane_varied.ply"
) -> str:
    """
    Generate background plane with depth variation for more realistic appearance.

    Args:
        root: Root directory path
        grid_size: (width, height) number of points in grid
        z_base: Base Z-coordinate depth
        z_variation: Random variation in Z depth
        output_name: Output PLY filename

    Returns:
        Full path to generated PLY file
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for gs_shim. Install with: pip install open3d"
        )

    output_path = Path(root) / "gs_shim" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate grid of points
    width, height = grid_size
    x = np.linspace(-2.0, 2.0, width)
    y = np.linspace(-1.0, 1.0, height)
    xv, yv = np.meshgrid(x, y)

    # Add random depth variation
    np.random.seed(42)  # Deterministic for testing
    z_values = z_base + np.random.uniform(-z_variation, z_variation, xv.size)

    # Flatten to get point positions
    points = np.stack([
        xv.flatten(),
        yv.flatten(),
        z_values
    ], axis=-1)

    # Generate colors with slight noise
    base_colors = np.zeros((points.shape[0], 3))
    base_colors[:, 0] = (xv.flatten() + 2.0) / 4.0
    base_colors[:, 1] = (yv.flatten() + 1.0) / 2.0
    base_colors[:, 2] = 0.5

    # Add color noise
    noise = np.random.uniform(-0.05, 0.05, base_colors.shape)
    colors = np.clip(base_colors + noise, 0.0, 1.0)

    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save as PLY
    o3d.io.write_point_cloud(str(output_path), pcd)

    return str(output_path)


def create_simple_scene(
    root: str,
    include_floor: bool = True,
    include_walls: bool = True
) -> str:
    """
    Create a simple 3D scene with floor and walls as pointcloud.

    Args:
        root: Root directory path
        include_floor: Whether to include floor plane
        include_walls: Whether to include wall planes

    Returns:
        Full path to generated PLY file
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for gs_shim. Install with: pip install open3d"
        )

    output_path = Path(root) / "gs_shim" / "simple_scene.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_points = []
    all_colors = []

    # Floor plane (y = -1)
    if include_floor:
        floor_grid = 200
        x = np.linspace(-3.0, 3.0, floor_grid)
        z = np.linspace(0.0, 6.0, floor_grid)
        xv, zv = np.meshgrid(x, z)

        floor_points = np.stack([
            xv.flatten(),
            np.full(xv.size, -1.0),
            zv.flatten()
        ], axis=-1)

        floor_colors = np.tile([0.7, 0.7, 0.7], (floor_points.shape[0], 1))

        all_points.append(floor_points)
        all_colors.append(floor_colors)

    # Back wall (z = 5)
    if include_walls:
        wall_grid = 150
        x = np.linspace(-3.0, 3.0, wall_grid)
        y = np.linspace(-1.0, 2.0, wall_grid)
        xv, yv = np.meshgrid(x, y)

        wall_points = np.stack([
            xv.flatten(),
            yv.flatten(),
            np.full(xv.size, 5.0)
        ], axis=-1)

        wall_colors = np.tile([0.9, 0.9, 0.85], (wall_points.shape[0], 1))

        all_points.append(wall_points)
        all_colors.append(wall_colors)

    # Combine all points
    if all_points:
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)

        # Create Open3D pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save as PLY
        o3d.io.write_point_cloud(str(output_path), pcd)

        return str(output_path)
    else:
        raise ValueError("No geometry to generate")


def load_and_preview_ply(ply_path: str) -> dict:
    """
    Load PLY file and return basic statistics.

    Args:
        ply_path: Path to PLY file

    Returns:
        Dictionary with point count, bounds, etc.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for gs_shim. Install with: pip install open3d"
        )

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    stats = {
        "point_count": len(points),
        "has_colors": len(colors) > 0,
        "bounds_min": points.min(axis=0).tolist() if len(points) > 0 else None,
        "bounds_max": points.max(axis=0).tolist() if len(points) > 0 else None,
    }

    return stats
