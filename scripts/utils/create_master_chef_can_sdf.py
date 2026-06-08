#!/usr/bin/env python3
import argparse
import os
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate High-Fidelity SDF collision USD for the YCB master chef can."
)

parser.add_argument(
    "--output",
    type=str,
    default="assets/Props/master_chef_can_sdf.usd",
    help="Output USD path.",
)

parser.add_argument(
    "--sdf-resolution",
    type=int,
    default=256,
    help="SDF voxel resolution.",
)
parser.add_argument(
    "--target-scale",
    type=float,
    default=0.5,
    help="Uniform scale to bake into the mesh.",
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="Mass to set on the rigid body in kg. If not specified, defaults to 0.02 scaled by target_scale^3.",
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
    UsdShade,
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

def _apply_robust_sdf_collision(mesh_prim: Usd.Prim, sdf_resolution: int):
    mesh = UsdGeom.Mesh(mesh_prim)
    double_sided_attr = mesh.GetDoubleSidedAttr()
    if double_sided_attr and double_sided_attr.Get():
        double_sided_attr.Set(False)

    scale = args.target_scale

    # Collision API
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr("sdf")

    # SDF collision parameters (matched with high-fidelity factory task peg settings)
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
    sdf_api.CreateSdfResolutionAttr(sdf_resolution)
    sdf_api.CreateSdfMarginAttr(0.01)
    sdf_api.CreateSdfNarrowBandThicknessAttr(0.01)
    sdf_api.CreateSdfSubgridResolutionAttr(6)

    # PhysX tuning
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision.CreateContactOffsetAttr(0.005)
    physx_collision.CreateRestOffsetAttr(0.0)
    physx_collision.CreateMinTorsionalPatchRadiusAttr(0.005)

def main():
    # YCB Master Chef Can 002
    input_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd"
    output_path = os.path.abspath(args.output)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Opening remote Master Chef Can USD: {input_path}")
    stage = Usd.Stage.Open(input_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {input_path}")

    mesh_prim = _find_mesh_prim(stage)
    if mesh_prim is None:
        raise RuntimeError("No UsdGeom.Mesh found in stage.")

    print(f"Upgrading mesh prim: {mesh_prim.GetPath()} with target_scale={args.target_scale}")
    _bake_world_transform(mesh_prim, args.target_scale)
    _apply_robust_sdf_collision(mesh_prim, args.sdf_resolution)

    # Apply RigidBody and Mass APIs
    UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    
    if args.mass is not None:
        mass_val = args.mass
    else:
        mass_val = 0.02 * (args.target_scale ** 3)
    mass_api.CreateMassAttr(mass_val)
    print(f"Applied mass: {mass_val} kg")

    # Compute and apply physically-correct Center of Mass & Diagonal Inertia
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    if points and len(points) > 0:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        
        mean_x = sum(xs) / len(points)
        mean_y = sum(ys) / len(points)
        mean_z = sum(zs) / len(points)
        center = Gf.Vec3f(mean_x, mean_y, mean_z)
        
        # Bounding box dimensions for inertia tensor
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        d = max(zs) - min(zs)
        
        # Solid cuboid inertia formulas
        ixx = (1.0 / 12.0) * mass_val * (h**2 + d**2)
        iyy = (1.0 / 12.0) * mass_val * (w**2 + d**2)
        izz = (1.0 / 12.0) * mass_val * (w**2 + h**2)
        
        # Ensure non-zero inertia
        ixx = max(ixx, 1e-6)
        iyy = max(iyy, 1e-6)
        izz = max(izz, 1e-6)
        
        diagonal_inertia = Gf.Vec3f(ixx, iyy, izz)
        
        mass_api.CreateCenterOfMassAttr().Set(center)
        mass_api.CreateDiagonalInertiaAttr().Set(diagonal_inertia)
        print(f"Set center of mass: {center}")
        print(f"Set diagonal inertia: {diagonal_inertia}")

    # Apply Physics Material with static/dynamic friction=2.0 and combine_mode=max
    physics_material_path = f"{mesh_prim.GetPath()}/physicsMaterial"
    material_prim = stage.DefinePrim(physics_material_path, "Material")
    UsdPhysics.MaterialAPI.Apply(material_prim)
    
    mat_api = UsdPhysics.MaterialAPI(material_prim)
    mat_api.CreateStaticFrictionAttr().Set(2.0)
    mat_api.CreateDynamicFrictionAttr().Set(2.0)
    mat_api.CreateRestitutionAttr().Set(0.0)
    
    px_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    px_mat_api.CreateFrictionCombineModeAttr().Set("max")
    px_mat_api.CreateRestitutionCombineModeAttr().Set("average")
    
    # Bind material to the mesh
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
        UsdShade.Material(material_prim), UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    print(f"Applied high friction (2.0) physics material: {physics_material_path}")

    print(f"Exporting to: {output_path}")
    stage.Export(output_path)
    print("Success.")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
