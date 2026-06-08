#!/usr/bin/env python3
"""
Inspect mesh properties of the cube (red_block) USD asset.

Usage:
    ./isaaclab.sh -p scripts/utils/test_mesh.py
    ./isaaclab.sh -p scripts/utils/test_mesh.py --usd /path/to/asset.usd
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect USD mesh properties.")
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Path to USD asset. Defaults to nucleus red_block.usd.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- USD imports (after AppLauncher) ---
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402
from isaaclab_tasks.direct.factory.factory_tasks_cfg import NutM16, Peg8mm  # noqa: E402


def _find_mesh_prim(stage: Usd.Stage):
    """Return first CollisionAPI mesh, or first mesh found."""
    first_mesh = None
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        if first_mesh is None:
            first_mesh = prim
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return prim
    return first_mesh



def show_mesh_properties(mesh_prim: Usd.Prim) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)

    print("=" * 60)
    print(f"Prim path  : {mesh_prim.GetPath()}")
    print(f"Prim type  : {mesh_prim.GetTypeName()}")

    # --- Xform ops (scale / translate / rotate) ---
    xformable = UsdGeom.Xformable(mesh_prim)
    ops = xformable.GetOrderedXformOps()
    print(f"\nXform ops  : {len(ops)}")
    for op in ops:
        print(f"  {op.GetOpName():30s}  value={op.Get()}")

    # --- Points (vertices) ---
    points = mesh.GetPointsAttr().Get()
    if points is not None:
        print(f"\nVertex count : {len(points)}")
        for i, p in enumerate(points[:8]):
            print(f"  [{i:3d}]  ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})")
        if len(points) > 8:
            print(f"  ... ({len(points) - 8} more)")
    else:
        print("\nVertices  : None")

    # --- Normals ---
    normals = mesh.GetNormalsAttr().Get()
    if normals is not None:
        print(f"\nNormal count : {len(normals)}")
        interp = mesh.GetNormalsInterpolation()
        print(f"Normal interp: {interp}")
        for i, n in enumerate(normals[:6]):
            print(f"  [{i:3d}]  ({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})")
        if len(normals) > 6:
            print(f"  ... ({len(normals) - 6} more)")
    else:
        print("\nNormals   : None")

    # --- Face vertex counts ---
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    if fvc is not None:
        unique_counts = sorted(set(fvc))
        print(f"\nFace count   : {len(fvc)}")
        print(f"Verts/face   : {unique_counts}")
    else:
        print("\nFace vertex counts: None")

    # --- Face vertex indices ---
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    if fvi is not None:
        print(f"\nFace vertex indices count: {len(fvi)}")
        print(f"  First 24: {list(fvi[:24])}")

    # --- Extent ---
    extent = mesh.GetExtentAttr().Get()
    if extent is not None:
        lo, hi = extent[0], extent[1]
        size = Gf.Vec3f(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        print(f"\nExtent min : ({lo[0]:+.6f}, {lo[1]:+.6f}, {lo[2]:+.6f})")
        print(f"Extent max : ({hi[0]:+.6f}, {hi[1]:+.6f}, {hi[2]:+.6f})")
        print(f"Size (m)   : ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})")
    else:
        print("\nExtent    : None")

    # --- Subdivision scheme ---
    subdiv = mesh.GetSubdivisionSchemeAttr().Get()
    print(f"\nSubdivision: {subdiv}")

    # --- Applied APIs ---
    apis = mesh_prim.GetAppliedSchemas()
    print(f"\nApplied APIs ({len(apis)}):")
    for api in apis:
        print(f"  {api}")

    # --- Physics: collision approximation ---
    if mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mc = UsdPhysics.MeshCollisionAPI(mesh_prim)
        approx = mc.GetApproximationAttr().Get()
        print(f"\nCollision approx: {approx}")

    # --- PhysX SDF ---
    if mesh_prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI(mesh_prim)
        print(f"\nSDF resolution       : {sdf.GetSdfResolutionAttr().Get()}")
        print(f"SDF margin           : {sdf.GetSdfMarginAttr().Get()}")
        print(f"SDF narrow band      : {sdf.GetSdfNarrowBandThicknessAttr().Get()}")
        print(f"SDF subgrid res      : {sdf.GetSdfSubgridResolutionAttr().Get()}")

    # --- PhysX Collision tuning ---
    if mesh_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        pc = PhysxSchema.PhysxCollisionAPI(mesh_prim)
        print(f"\nContact offset       : {pc.GetContactOffsetAttr().Get()}")
        print(f"Rest offset          : {pc.GetRestOffsetAttr().Get()}")
        print(f"Min torsional patch r: {pc.GetMinTorsionalPatchRadiusAttr().Get()}")

    print("=" * 60)


def main():
    import omni.usd

    usd_path = args.usd
    if usd_path is None:
        usd_path = Peg8mm().usd_path

    print(f"Opening: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Failed to open: {usd_path}")

    mesh_prim = _find_mesh_prim(stage)
    if mesh_prim is None:
        raise RuntimeError("No UsdGeom.Mesh found in stage.")

    show_mesh_properties(mesh_prim)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
