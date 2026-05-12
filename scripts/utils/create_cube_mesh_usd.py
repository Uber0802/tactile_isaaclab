#!/usr/bin/env python3
"""
Create a box USD asset with real UsdGeom.Mesh collision geometry.

Unlike UsdGeom.Cube (a shape primitive), UsdGeom.Mesh allows PhysX to
build a proper SDF from the triangle mesh, which is required for the
TacSL visuotactile force-field sensor to work correctly.

Usage (run inside Isaac Sim Python environment):
    ./isaaclab.sh -p scripts/utils/create_cube_mesh_usd.py \
        --size 0.04 0.04 0.04 \
        --output assets/Props/cube_mesh_4cm.usd

Or generate all block variants used in the stacking task:
    ./isaaclab.sh -p scripts/utils/create_cube_mesh_usd.py --all
"""

import argparse
import os

# ── Must be imported BEFORE pxr ──────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate a cube USD with UsdGeom.Mesh collision.")
parser.add_argument("--size", nargs=3, type=float, default=[0.04, 0.04, 0.04],
                    metavar=("W", "D", "H"), help="Box half-extents in metres (default: 4 cm cube).")
parser.add_argument("--output", type=str, default="./cube_mesh.usd",
                    help="Output USD file path.")
parser.add_argument("--sdf_resolution", type=int, default=256,
                    help="SDF resolution for PhysX SDF mesh collision (default: 256).")
parser.add_argument("--color", nargs=3, type=float, default=[0.2, 0.4, 0.8],
                    metavar=("R", "G", "B"), help="Diffuse colour (default: blue).")
parser.add_argument("--all", action="store_true",
                    help="Generate blue/red/green 4 cm cubes matching the stacking task.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── pxr imports (post-app-launch) ────────────────────────────────────────────
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def _unit_cube_mesh():
    """Return (points, normals, face_vertex_counts, face_vertex_indices) for a unit cube [-0.5, 0.5]^3."""
    # 8 corners
    pts = [
        (-0.5, -0.5, -0.5),  # 0
        ( 0.5, -0.5, -0.5),  # 1
        ( 0.5,  0.5, -0.5),  # 2
        (-0.5,  0.5, -0.5),  # 3
        (-0.5, -0.5,  0.5),  # 4
        ( 0.5, -0.5,  0.5),  # 5
        ( 0.5,  0.5,  0.5),  # 6
        (-0.5,  0.5,  0.5),  # 7
    ]
    # 6 faces × 2 triangles = 12 triangles (winding: CCW viewed from outside)
    faces = [
        # -Z
        (0, 3, 2), (0, 2, 1),
        # +Z
        (4, 5, 6), (4, 6, 7),
        # -X
        (0, 4, 7), (0, 7, 3),
        # +X
        (1, 2, 6), (1, 6, 5),
        # -Y
        (0, 1, 5), (0, 5, 4),
        # +Y
        (2, 3, 7), (2, 7, 6),
    ]
    points = [Gf.Vec3f(*p) for p in pts]
    indices = [idx for tri in faces for idx in tri]
    counts = [3] * len(faces)
    return points, counts, indices


def _find_mesh_prim(stage: Usd.Stage) -> Usd.Prim | None:
    """
    Return the first UsdGeom.Mesh prim in the stage.
    Walks the tree so it is robust to different hierarchy layouts.
    """
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            return prim
    return None
 
def _apply_sdf_collision(prim: Usd.Prim, sdf_resolution: int) -> None:
    """
    Apply PhysX SDF collision APIs to an existing UsdGeom.Mesh prim.
    Safe to call even if CollisionAPI is already present (Apply is idempotent).
    """
    UsdPhysics.CollisionAPI.Apply(prim)
 
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col.CreateApproximationAttr("sdf")
 
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    sdf_api.CreateSdfResolutionAttr(sdf_resolution)
    sdf_api.CreateSdfMarginAttr(0.01)              # 1 % bounding-box margin
    sdf_api.CreateSdfNarrowBandThicknessAttr(0.01)
    sdf_api.CreateSdfSubgridResolutionAttr(6)      # sparse sub-grid
 
    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_col.CreateContactOffsetAttr(0.001)
    physx_col.CreateRestOffsetAttr(0.0)

def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://") or path.startswith("omniverse://")

def upgrade_block_usd(
    reference_usd_path: str,
    output_path: str,
    sdf_resolution: int = 256,
) -> str:
    """
    Create a USD that references an existing block USD and overrides its Mesh
    prim to add PhysX SDF collision.  All other properties are inherited.
 
    Args:
        reference_usd_path: Absolute path to the source Nucleus USD.
        output_path:        Destination .usd file path.
        sdf_resolution:     PhysX SDF voxel resolution.
 
    Returns:
        The resolved output path.
    """
    # Don't abspath URLs — only resolve local paths
    if not _is_url(reference_usd_path):
        reference_usd_path = os.path.abspath(reference_usd_path)
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
 
    # ── Inspect the reference to locate the Mesh prim ────────────────────────
    ref_stage = Usd.Stage.Open(reference_usd_path)
    ref_mesh_prim = _find_mesh_prim(ref_stage)
    if ref_mesh_prim is None:
        raise RuntimeError(
            f"No UsdGeom.Mesh found in reference USD: {reference_usd_path}\n"
            "The block must already use a triangle mesh for SDF to work."
        )
 
    # Prim path within the reference (e.g. "/red_block/Cube")
    mesh_prim_path = str(ref_mesh_prim.GetPath())
 
    # Default prim of the reference (e.g. "red_block")
    ref_default_prim = ref_stage.GetDefaultPrim()
    default_prim_name = ref_default_prim.GetName() if ref_default_prim else "block"
 
    print(f"[upgrade_block_usd] Reference : {reference_usd_path}")
    print(f"  Default prim  : /{default_prim_name}")
    print(f"  Mesh prim     : {mesh_prim_path}")
 
    # ── Build the output stage ────────────────────────────────────────────────
    stage = Usd.Stage.CreateNew(output_path)
 
    # Inherit stage metadata from the reference
    stage.SetMetadata("metersPerUnit", ref_stage.GetMetadata("metersPerUnit") or 1.0)
    up_axis = UsdGeom.GetStageUpAxis(ref_stage)
    UsdGeom.SetStageUpAxis(stage, up_axis)
 
    # ── Reference the original USD ────────────────────────────────────────────
    root_prim = stage.DefinePrim(f"/{default_prim_name}")
    root_prim.GetReferences().AddReference(reference_usd_path)
    stage.SetDefaultPrim(root_prim)
 
    # ── Override only the Mesh prim — add SDF collision APIs ──────────────────
    override_prim = stage.OverridePrim(mesh_prim_path)
    _apply_sdf_collision(override_prim, sdf_resolution)
 
    stage.Save()
 
    print(f"  Output        : {output_path}")
    print(f"  SDF resolution: {sdf_resolution}")
    print(f"  Override prim : {mesh_prim_path}  (SDF APIs added)")
    print()
    return output_path

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_VARIANTS = [
    ("red_block_sdf.usd",   "red_block.usd"),
    ("blue_block_sdf.usd",  "blue_block.usd"),
    ("green_block_sdf.usd", "green_block.usd"),
]

def _nucleus_dir() -> str:
    nucleus = os.environ.get("ISAAC_NUCLEUS_DIR")
    if nucleus:
        return nucleus
    # 2. Isaac Lab's built-in asset utility (preferred)
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR as _NUCLEUS
        return _NUCLEUS
    except ImportError:
        raise EnvironmentError(
            "ISAAC_NUCLEUS_DIR is not set. "
            "Source the Isaac Lab setup script before running this tool."
        )

if __name__ == "__main__":
    nucleus = _nucleus_dir()
    out_dir = os.path.join(os.path.dirname(__file__), "../../assets/Props")
    for out_filename, src_filename in BLOCK_VARIANTS:
        upgrade_block_usd(
            reference_usd_path=os.path.join(nucleus, "Props/Blocks", src_filename),
            output_path=os.path.join(out_dir, out_filename),
            sdf_resolution=args.sdf_resolution,
        )
        
    simulation_app.close()
