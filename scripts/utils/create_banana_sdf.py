#!/usr/bin/env python3
import argparse
import os
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate SDF collision USD for the remote banana asset."
)

parser.add_argument(
    "--output",
    type=str,
    default="assets/Props/banana_sdf.usd",
    help="Output USD path.",
)

parser.add_argument(
    "--sdf-resolution",
    type=int,
    default=1024,
    help="SDF voxel resolution.",
)

parser.add_argument(
    "--target-scale",
    type=float,
    default=1.0,
    help="Uniform scale to bake into the mesh.",
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="Mass to set on the rigid body in kg. If not specified, defaults to 0.05 scaled by target_scale^3 (physically-accurate volume scaling).",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# USD imports
# -----------------------------------------------------------------------------

from pxr import (
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

def _find_mesh_prim(stage: Usd.Stage):
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
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    min_pt = Gf.Vec3f(min(xs), min(ys), min(zs))
    max_pt = Gf.Vec3f(max(xs), max(ys), max(zs))
    return Vt.Vec3fArray([min_pt, max_pt])

def _bake_world_transform(mesh_prim: Usd.Prim, target_scale: float) -> None:
    """Bake the full parent-chain world transform + target_scale into mesh points."""
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    if not points:
        return

    xform_cache = UsdGeom.XformCache()
    local_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)

    transformed_points = Vt.Vec3fArray([
        Gf.Vec3f(*(float(v * target_scale) for v in local_to_world.Transform(Gf.Vec3d(*p))))
        for p in points
    ])
    points_attr.Set(transformed_points)
    mesh.GetExtentAttr().Set(_compute_extent(transformed_points))

    # Clear transforms on the mesh prim and all ancestor prims
    prim = mesh_prim
    while prim and not prim.IsPseudoRoot():
        if prim.IsA(UsdGeom.Xformable):
            UsdGeom.Xformable(prim).ClearXformOpOrder()
        prim = prim.GetParent()

def _apply_sdf_collision(mesh_prim: Usd.Prim, sdf_resolution: int):
    mesh = UsdGeom.Mesh(mesh_prim)
    double_sided_attr = mesh.GetDoubleSidedAttr()
    if double_sided_attr and double_sided_attr.Get():
        double_sided_attr.Set(False)

    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr("sdf")

    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
    sdf_api.CreateSdfResolutionAttr(sdf_resolution)
    sdf_api.CreateSdfMarginAttr(0.002)
    sdf_api.CreateSdfNarrowBandThicknessAttr(0.005)
    sdf_api.CreateSdfSubgridResolutionAttr(6)

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision.CreateContactOffsetAttr(0.005)
    physx_collision.CreateRestOffsetAttr(0.0)
    physx_collision.CreateMinTorsionalPatchRadiusAttr(0.005)

def main():
    # 011 Banana
    input_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/011_banana.usd"
    output_path = os.path.abspath(args.output)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Opening remote Banana USD: {input_path}")
    stage = Usd.Stage.Open(input_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {input_path}")

    mesh_prim = _find_mesh_prim(stage)
    if mesh_prim is None:
        raise RuntimeError("No UsdGeom.Mesh found in stage.")

    print(f"Upgrading mesh prim: {mesh_prim.GetPath()} with target_scale={args.target_scale}")
    _bake_world_transform(mesh_prim, args.target_scale)
    _apply_sdf_collision(mesh_prim, args.sdf_resolution)

    # Rigid Body and Mass
    UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    
    if args.mass is not None:
        mass_val = args.mass
    else:
        mass_val = 0.05 * (args.target_scale ** 3)
        
    mass_api.CreateMassAttr(mass_val)
    print(f"Applied mass: {mass_val} kg")

    print(f"Exporting to: {output_path}")
    stage.Export(output_path)
    print("Success.")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
