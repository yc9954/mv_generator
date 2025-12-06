"""Unit tests for gs_shim module."""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if open3d is available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from gs_shim import (
    generate_bg_plane,
    generate_bg_plane_with_depth_variation,
    create_simple_scene,
    load_and_preview_ply
)


@pytest.fixture
def temp_root():
    """Create a temporary root directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_creates_file(temp_root):
    """Test that generate_bg_plane creates a PLY file."""
    ply_path = generate_bg_plane(temp_root)

    assert os.path.exists(ply_path)
    assert ply_path.endswith(".ply")
    assert "gs_shim" in ply_path


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_has_points(temp_root):
    """Test that generated PLY file contains points."""
    ply_path = generate_bg_plane(temp_root, grid_size=(100, 50))

    # Load and check point count
    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    # Should have 100 * 50 = 5000 points
    assert len(points) == 5000


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_more_than_100_points(temp_root):
    """Test that generated PLY has more than 100 points (requirement)."""
    ply_path = generate_bg_plane(temp_root, grid_size=(20, 10))

    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    # Should have 20 * 10 = 200 points > 100
    assert len(points) > 100


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_has_colors(temp_root):
    """Test that generated PLY file has color information."""
    ply_path = generate_bg_plane(temp_root)

    pcd = o3d.io.read_point_cloud(ply_path)
    colors = o3d.utility.Vector3dVector.numpy(pcd.colors)

    assert len(colors) > 0
    # Colors should be in range [0, 1]
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_z_depth(temp_root):
    """Test that points are at specified Z depth."""
    z_depth = 5.0
    ply_path = generate_bg_plane(temp_root, z_depth=z_depth)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    # All points should have Z coordinate close to z_depth
    z_coords = points[:, 2]
    assert all(abs(z - z_depth) < 0.01 for z in z_coords)


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_custom_output_name(temp_root):
    """Test generate_bg_plane with custom output name."""
    custom_name = "custom_plane.ply"
    ply_path = generate_bg_plane(temp_root, output_name=custom_name)

    assert os.path.exists(ply_path)
    assert ply_path.endswith(custom_name)


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_generate_bg_plane_with_depth_variation(temp_root):
    """Test generate_bg_plane_with_depth_variation creates PLY."""
    ply_path = generate_bg_plane_with_depth_variation(temp_root)

    assert os.path.exists(ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    assert len(points) > 100

    # Z coordinates should vary around base depth
    z_coords = points[:, 2]
    assert z_coords.std() > 0  # Should have variation


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_create_simple_scene_with_floor(temp_root):
    """Test create_simple_scene with floor enabled."""
    ply_path = create_simple_scene(temp_root, include_floor=True, include_walls=False)

    assert os.path.exists(ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    assert len(points) > 100


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_create_simple_scene_with_walls(temp_root):
    """Test create_simple_scene with walls enabled."""
    ply_path = create_simple_scene(temp_root, include_floor=False, include_walls=True)

    assert os.path.exists(ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    assert len(points) > 100


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_create_simple_scene_full(temp_root):
    """Test create_simple_scene with both floor and walls."""
    ply_path = create_simple_scene(temp_root, include_floor=True, include_walls=True)

    assert os.path.exists(ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = o3d.utility.Vector3dVector.numpy(pcd.points)

    # Should have more points with both floor and walls
    assert len(points) > 200


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_create_simple_scene_no_geometry(temp_root):
    """Test create_simple_scene raises error with no geometry."""
    with pytest.raises(ValueError, match="No geometry to generate"):
        create_simple_scene(temp_root, include_floor=False, include_walls=False)


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_load_and_preview_ply(temp_root):
    """Test load_and_preview_ply returns correct statistics."""
    # Generate a PLY first
    ply_path = generate_bg_plane(temp_root, grid_size=(50, 30))

    stats = load_and_preview_ply(ply_path)

    assert stats["point_count"] == 50 * 30
    assert stats["has_colors"] is True
    assert stats["bounds_min"] is not None
    assert stats["bounds_max"] is not None
    assert len(stats["bounds_min"]) == 3
    assert len(stats["bounds_max"]) == 3


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")
def test_load_and_preview_ply_bounds(temp_root):
    """Test load_and_preview_ply computes correct bounds."""
    ply_path = generate_bg_plane(temp_root, z_depth=4.0)

    stats = load_and_preview_ply(ply_path)

    # Check bounds make sense
    bounds_min = stats["bounds_min"]
    bounds_max = stats["bounds_max"]

    # X should be in [-2, 2]
    assert bounds_min[0] >= -2.1
    assert bounds_max[0] <= 2.1

    # Y should be in [-1, 1]
    assert bounds_min[1] >= -1.1
    assert bounds_max[1] <= 1.1

    # Z should be around 4.0
    assert abs(bounds_min[2] - 4.0) < 0.1
    assert abs(bounds_max[2] - 4.0) < 0.1


def test_import_without_open3d():
    """Test that importing gs_shim works even without open3d."""
    # This test verifies the module can be imported
    # Functions will raise ImportError when called
    import gs_shim
    assert gs_shim is not None
