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


def create_cube_mesh_usd(
    output_path: str,
    size: tuple[float, float, float] = (0.04, 0.04, 0.04),
    color: tuple[float, float, float] = (0.2, 0.4, 0.8),
    sdf_resolution: int = 256,
) -> str:
    """
    Create a USD file containing a rigid box whose collision shape is a
    UsdGeom.Mesh (triangle mesh), enabling PhysX SDF collision to work.

    Args:
        output_path: Destination .usd file path.
        size: (width, depth, height) of the box in metres.
        color: (R, G, B) diffuse colour in [0, 1].
        sdf_resolution: PhysX SDF resolution (higher = more accurate but slower).

    Returns:
        The resolved output path.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # ── Root Xform (rigid body) ───────────────────────────────────────────────
    root_path = "/Cube"
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root_xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)
    UsdPhysics.MassAPI(root_prim).CreateMassAttr(0.1)

    # Set default prim
    stage.SetDefaultPrim(root_prim)

    # ── Visual mesh (separate, no collision) ──────────────────────────────────
    vis_path = root_path + "/VisualMesh"
    vis_mesh = UsdGeom.Mesh.Define(stage, vis_path)
    points, counts, indices = _unit_cube_mesh()
    # Scale points to desired box size
    sx, sy, sz = size
    scaled_points = [Gf.Vec3f(p[0] * sx, p[1] * sy, p[2] * sz) for p in points]
    vis_mesh.GetPointsAttr().Set(scaled_points)
    vis_mesh.GetFaceVertexCountsAttr().Set(counts)
    vis_mesh.GetFaceVertexIndicesAttr().Set(indices)
    vis_mesh.GetSubdivisionSchemeAttr().Set("none")

    # Material
    mat_path = root_path + "/Looks/CubeMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = mat_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(vis_mesh.GetPrim()).Bind(mat)

    # ── Collision mesh (UsdGeom.Mesh with SDF approximation) ──────────────────
    col_path = root_path + "/CollisionMesh"
    col_mesh = UsdGeom.Mesh.Define(stage, col_path)
    col_mesh.GetPointsAttr().Set(scaled_points)
    col_mesh.GetFaceVertexCountsAttr().Set(counts)
    col_mesh.GetFaceVertexIndicesAttr().Set(indices)
    col_mesh.GetSubdivisionSchemeAttr().Set("none")

    # Make it invisible (collision only)
    UsdGeom.Imageable(col_mesh.GetPrim()).MakeInvisible()

    # Apply collision APIs
    col_prim = col_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(col_prim)

    # Apply MeshCollisionAPI with SDF approximation
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(col_prim)
    mesh_collision_api.CreateApproximationAttr("sdf")

    # Apply PhysX SDF mesh API
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(col_prim)
    sdf_api.CreateSdfResolutionAttr(sdf_resolution)
    sdf_api.CreateSdfMarginAttr(0.01)         # 1% bounding-box margin
    sdf_api.CreateSdfNarrowBandThicknessAttr(0.01)
    sdf_api.CreateSdfSubgridResolutionAttr(6)  # sparse SDF

    # ── PhysX collision properties ────────────────────────────────────────────
    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(col_prim)
    physx_col.CreateContactOffsetAttr(0.001)
    physx_col.CreateRestOffsetAttr(0.0)

    stage.Save()
    print(f"[create_cube_mesh_usd] Saved: {output_path}")
    print(f"  Size:           {sx:.4f} x {sy:.4f} x {sz:.4f} m")
    print(f"  SDF resolution: {sdf_resolution}")
    print(f"  Collision prim: {col_path}  (UsdGeom.Mesh + SDF approximation)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_VARIANTS = [
    ("blue_block_mesh.usd",  (0.04, 0.04, 0.04), (0.0, 0.1, 0.8)),
    ("red_block_mesh.usd",   (0.04, 0.04, 0.04), (0.8, 0.1, 0.0)),
    ("green_block_mesh.usd", (0.04, 0.04, 0.04), (0.1, 0.7, 0.1)),
]

if __name__ == "__main__":
    if args.all:
        out_dir = os.path.join(os.path.dirname(__file__), "../../assets/Props")
        for filename, size, color in BLOCK_VARIANTS:
            create_cube_mesh_usd(
                output_path=os.path.join(out_dir, filename),
                size=size,
                color=color,
                sdf_resolution=args.sdf_resolution,
            )
    else:
        create_cube_mesh_usd(
            output_path=args.output,
            size=tuple(args.size),
            color=tuple(args.color),
            sdf_resolution=args.sdf_resolution,
        )

    simulation_app.close()
