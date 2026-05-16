#!/usr/bin/env python3
"""
Upgrade existing USD mesh assets with robust PhysX SDF collision.

This tool:
- Opens an existing USD asset
- Finds the collision mesh
- Bakes local mesh scaling into vertices
- Recomputes extents
- Applies PhysX SDF collision APIs
- Flattens the final result into a standalone USD

Designed for:
- Isaac Sim
- Isaac Lab
- TacSL / tactile sensing
- Small-object stable contact simulation

Examples:

Generate a single upgraded block:
    ./isaaclab.sh -p scripts/utils/upgrade_sdf_collision.py \
        --output ./red_block_sdf.usd

Generate all standard block variants:
    ./isaaclab.sh -p scripts/utils/upgrade_sdf_collision.py \
        --all
"""

import argparse
import os
from typing import Optional

# Must be imported before pxr
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Upgrade USD mesh assets with PhysX SDF collision."
)

parser.add_argument(
    "--output",
    type=str,
    default="./cube_mesh.usd",
    help="Output USD path.",
)

parser.add_argument(
    "--sdf-resolution",
    type=int,
    default=256,
    help="SDF voxel resolution. Recommended: 256 for ~4cm objects.",
)

parser.add_argument(
    "--contact-offset",
    type=float,
    default=0.001,
    help="PhysX contact offset.",
)

parser.add_argument(
    "--sdf-margin",
    type=float,
    default=0.01,
    help="SDF margin in meters.",
)

parser.add_argument(
    "--narrow-band",
    type=float,
    default=0.01,
    help="SDF narrow band thickness.",
)

parser.add_argument(
    "--all",
    action="store_true",
    help="Generate all standard block variants.",
)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# USD imports
# -----------------------------------------------------------------------------

from pxr import (  # noqa: E402
    Gf,
    PhysxSchema,
    Usd,
    UsdGeom,
    UsdPhysics,
    Vt,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _is_url(path: str) -> bool:
    return (
        path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("omniverse://")
    )


def _find_mesh_prim(stage: Usd.Stage) -> Optional[Usd.Prim]:
    """
    Find the best candidate mesh prim.

    Preference order:
    1. Mesh already marked with CollisionAPI
    2. First mesh encountered
    """

    first_mesh = None

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        if first_mesh is None:
            first_mesh = prim

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return prim

    return first_mesh


def _compute_extent(points):
    """
    Compute axis-aligned extent manually.
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    min_pt = Gf.Vec3f(min(xs), min(ys), min(zs))
    max_pt = Gf.Vec3f(max(xs), max(ys), max(zs))

    return Vt.Vec3fArray([min_pt, max_pt])


def _bake_local_scale(mesh_prim: Usd.Prim) -> None:
    """
    Bake local scale ops directly authored on the mesh prim.

    This avoids PhysX SDF issues caused by scaled mesh geometry
    while preserving parent hierarchy transforms.
    """

    mesh = UsdGeom.Mesh(mesh_prim)

    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    if not points:
        return

    xformable = UsdGeom.Xformable(mesh_prim)

    scale_op = None

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    if scale_op is None:
        return

    scale = scale_op.Get()

    if scale is None:
        return

    sx, sy, sz = scale[0], scale[1], scale[2]
    print(sx, sy, sz)

    transformed_points = Vt.Vec3fArray(
        [
            Gf.Vec3f(
                float(p[0] * sx),
                float(p[1] * sy),
                float(p[2] * sz),
            )
            for p in points
        ]
    )

    points_attr.Set(transformed_points)

    mesh.GetExtentAttr().Set(
        _compute_extent(transformed_points)
    )

    # Reset scale op
    scale_op.Set(type(scale)(1.0, 1.0, 1.0))


def _apply_sdf_collision(
    mesh_prim: Usd.Prim,
    sdf_resolution: int,
    contact_offset: float,
    sdf_margin: float,
    narrow_band: float,
) -> None:
    """
    Apply PhysX SDF collision APIs.
    """

    mesh = UsdGeom.Mesh(mesh_prim)

    # Double-sided meshes can confuse SDF generation
    double_sided_attr = mesh.GetDoubleSidedAttr()

    if double_sided_attr and double_sided_attr.Get():
        double_sided_attr.Set(False)

    # Collision API
    UsdPhysics.CollisionAPI.Apply(mesh_prim)

    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr("sdf")

    # SDF collision
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)

    sdf_api.CreateSdfResolutionAttr(sdf_resolution)
    sdf_api.CreateSdfMarginAttr(sdf_margin)
    sdf_api.CreateSdfNarrowBandThicknessAttr(narrow_band)
    sdf_api.CreateSdfSubgridResolutionAttr(6)

    # PhysX collision tuning
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)

    physx_collision.CreateContactOffsetAttr(contact_offset)
    physx_collision.CreateRestOffsetAttr(0.0)

    # Helps reduce rotational jitter for small objects
    physx_collision.CreateMinTorsionalPatchRadiusAttr(0.005)


def _upgrade_asset(input_path: str, output_path: str) -> None:
    """
    Upgrade a USD asset with SDF collision.
    """

    if not _is_url(input_path):
        input_path = os.path.abspath(input_path)

    output_path = os.path.abspath(output_path)

    out_dir = os.path.dirname(output_path)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Opening USD: {input_path}")

    stage = Usd.Stage.Open(input_path)

    if stage is None:
        raise RuntimeError(f"Failed to open USD: {input_path}")

    mesh_prim = _find_mesh_prim(stage)

    if mesh_prim is None:
        raise RuntimeError("No UsdGeom.Mesh found in stage.")

    print(f"Using mesh prim: {mesh_prim.GetPath()}")

    # Bake local mesh scaling into vertices
    _bake_local_scale(mesh_prim)

    # Apply SDF collision APIs
    _apply_sdf_collision(
        mesh_prim=mesh_prim,
        sdf_resolution=args.sdf_resolution,
        contact_offset=args.contact_offset,
        sdf_margin=args.sdf_margin,
        narrow_band=args.narrow_band,
    )

    print("Exporting flattened USD...")

    stage.Export(output_path)

    print("Done.")
    print(f"Output: {output_path}")
    print()


# -----------------------------------------------------------------------------
# Standard block generation
# -----------------------------------------------------------------------------

BLOCK_VARIANTS = [
    ("red_block_sdf.usd", "red_block.usd"),
    ("blue_block_sdf.usd", "blue_block.usd"),
    ("green_block_sdf.usd", "green_block.usd"),
]


def _nucleus_dir() -> str:
    nucleus = os.environ.get("ISAAC_NUCLEUS_DIR")

    if nucleus:
        return nucleus

    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR as _NUCLEUS
        return _NUCLEUS
    except ImportError:
        raise EnvironmentError(
            "ISAAC_NUCLEUS_DIR is not set."
        )


def main():
    nucleus = _nucleus_dir()

    if args.all:
        out_dir = os.path.join(
            os.path.dirname(__file__),
            "../../assets/Props",
        )

        for out_filename, src_filename in BLOCK_VARIANTS:
            input_path = os.path.join(
                nucleus,
                "Props/Blocks",
                src_filename,
            )

            output_path = os.path.join(
                out_dir,
                out_filename,
            )

            _upgrade_asset(input_path, output_path)

    else:
        input_path = os.path.join(
            nucleus,
            "Props/Blocks",
            "red_block.usd",
        )

        _upgrade_asset(
            input_path=input_path,
            output_path=args.output,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()