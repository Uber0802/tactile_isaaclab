#!/usr/bin/env python3
import argparse
import os
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate SDF collision USD for the local lamp bulb asset."
)

parser.add_argument(
    "--input",
    type=str,
    default="assets/Props/lamp_bulb/configuration/lamp_bulb_physics.usd",
    help="Input USD path.",
)

parser.add_argument(
    "--output",
    type=str,
    default="assets/Props/lamp_bulb_sdf.usd",
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
    default=0.03,
    help="Mass to set on the rigid body in kg. Default is 0.03 kg.",
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
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    Vt,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _find_mesh_prim(stage: Usd.Stage):
    # Try to find mesh with "collision" or "collider" in its path
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        path_str = str(prim.GetPath()).lower()
        if "collision" in path_str or "collider" in path_str:
            return prim
            
    # Try to find mesh with CollisionAPI
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return prim
            
    # Fallback to first mesh
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            return prim
            
    return None

def _compute_extent(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    min_pt = Gf.Vec3f(min(xs), min(ys), min(zs))
    max_pt = Gf.Vec3f(max(xs), max(ys), max(zs))
    return Vt.Vec3fArray([min_pt, max_pt])

def _bake_world_transform_relative_to_rb(mesh_prim: Usd.Prim, rb_prim: Usd.Prim, target_scale: float) -> None:
    """Bake the transform of mesh_prim relative to rb_prim into the mesh points."""
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    if not points:
        return

    xform_cache = UsdGeom.XformCache()
    mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
    rb_to_world = xform_cache.GetLocalToWorldTransform(rb_prim)

    # Compute local-to-rigid-body transformation matrix
    local_to_rb = mesh_to_world * rb_to_world.GetInverse()

    transformed_points = Vt.Vec3fArray([
        Gf.Vec3f(*(float(v * target_scale) for v in local_to_rb.Transform(Gf.Vec3d(*p))))
        for p in points
    ])
    points_attr.Set(transformed_points)
    mesh.GetExtentAttr().Set(_compute_extent(transformed_points))

    # Clear transforms on intermediate prims up to (but not including) the rigid body
    prim = mesh_prim
    while prim and prim != rb_prim:
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
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Opening local lamp bulb USD: {input_path}")
    stage = Usd.Stage.Open(input_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {input_path}")

    # 1. Resolve all instances first
    changed = True
    while changed:
        changed = False
        for prim in stage.TraverseAll():
            if prim.IsInstance():
                print(f"Resolving instance prim: {prim.GetPath()}")
                prim.SetInstanceable(False)
                changed = True
                break

    # 2. Flatten the stage to merge all references and instances into a single layer
    print("Flattening stage to resolve all references...")
    flat_layer = stage.Flatten()
    
    # Create a new in-memory stage from the flattened layer
    flat_stage = Usd.Stage.Open(flat_layer)
    if flat_stage is None:
        raise RuntimeError("Failed to open flattened stage.")

    # 3. Replace the convex hull collision mesh with the high-fidelity visual mesh on the flattened stage
    visual_mesh_path = Sdf.Path("/lamp_bulb/lamp_bulb/visuals/lamp_bulb/mesh")
    collision_parent_path = Sdf.Path("/lamp_bulb/lamp_bulb/collisions")
    collision_mesh_path = Sdf.Path("/lamp_bulb/lamp_bulb/collisions/lamp_bulb/mesh")

    if flat_stage.GetPrimAtPath(visual_mesh_path).IsValid():
        print(f"Copying visual mesh {visual_mesh_path} to collision mesh {collision_mesh_path} to use high-fidelity geometry.")
        # Remove old collisions prim (including the convex hull mesh and any merge/collision APIs on it)
        if flat_stage.GetPrimAtPath(collision_parent_path).IsValid():
            flat_stage.RemovePrim(collision_parent_path)
        
        # Define new collisions parent Xform
        UsdGeom.Xform.Define(flat_stage, collision_parent_path)
        
        # Define intermediate parent Xform to prevent Sdf.CopySpec verification errors
        intermediate_parent_path = Sdf.Path("/lamp_bulb/lamp_bulb/collisions/lamp_bulb")
        UsdGeom.Xform.Define(flat_stage, intermediate_parent_path)
        
        # Copy the visual mesh to the collision mesh path
        Sdf.CopySpec(flat_stage.GetRootLayer(), visual_mesh_path, flat_stage.GetRootLayer(), collision_mesh_path)
    else:
        print("Warning: Visual mesh /lamp_bulb/lamp_bulb/visuals/lamp_bulb/mesh not found. Falling back to existing collision mesh.")

    # 4. Remove all root prims except the main one (/lamp_bulb)
    for prim in flat_stage.GetPseudoRoot().GetChildren():
        if prim.GetName() != "lamp_bulb":
            print(f"Removing unused root prim: {prim.GetPath()}")
            flat_stage.RemovePrim(prim.GetPath())

    # 5. Find the collision mesh prim under the resolved /lamp_bulb hierarchy
    mesh_prim = flat_stage.GetPrimAtPath(collision_mesh_path)
    if not mesh_prim.IsValid():
        mesh_prim = _find_mesh_prim(flat_stage)
        
    if mesh_prim is None or not mesh_prim.IsValid():
        raise RuntimeError("No UsdGeom.Mesh found in stage.")
    print(f"Using collision mesh prim: {mesh_prim.GetPath()}")

    # 6. Find the parent rigid body of the mesh prim to apply mass to
    rb_prim = None
    curr = mesh_prim
    while curr and not curr.IsPseudoRoot():
        if curr.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_prim = curr
            break
        curr = curr.GetParent()

    if rb_prim is None:
        rb_prim = flat_stage.GetPrimAtPath("/lamp_bulb/lamp_bulb")
        if rb_prim.IsValid():
            UsdPhysics.RigidBodyAPI.Apply(rb_prim)
            print(f"Applied RigidBodyAPI to: {rb_prim.GetPath()}")
        else:
            rb_prim = None

    # 7. Bake world transform relative to the rigid body
    if rb_prim:
        print(f"Baking transform of {mesh_prim.GetPath()} relative to rigid body: {rb_prim.GetPath()}")
        _bake_world_transform_relative_to_rb(mesh_prim, rb_prim, args.target_scale)
    else:
        print("No rigid body found. Baking absolute world transform.")
        # Fallback to absolute bake if no rigid body
        xform_cache = UsdGeom.XformCache()
        local_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get()
        if points:
            transformed_points = Vt.Vec3fArray([
                Gf.Vec3f(*(float(v * args.target_scale) for v in local_to_world.Transform(Gf.Vec3d(*p))))
                for p in points
            ])
            points_attr.Set(transformed_points)
            mesh.GetExtentAttr().Set(_compute_extent(transformed_points))
        prim = mesh_prim
        while prim and not prim.IsPseudoRoot():
            if prim.IsA(UsdGeom.Xformable):
                UsdGeom.Xformable(prim).ClearXformOpOrder()
            prim = prim.GetParent()

    # 8. Apply SDF collision parameters to the mesh
    _apply_sdf_collision(mesh_prim, args.sdf_resolution)

    # 8b. Add a solid analytical Sphere collider to protect the glass dome from penetration
    sphere_path = Sdf.Path("/lamp_bulb/lamp_bulb/collisions/sphere")
    sphere = UsdGeom.Sphere.Define(flat_stage, sphere_path)
    sphere.CreateRadiusAttr(0.030)
    sphere.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.038, 0.0))
    
    UsdPhysics.CollisionAPI.Apply(flat_stage.GetPrimAtPath(sphere_path))
    physx_sph_collision = PhysxSchema.PhysxCollisionAPI.Apply(flat_stage.GetPrimAtPath(sphere_path))
    physx_sph_collision.CreateContactOffsetAttr(0.005)
    physx_sph_collision.CreateRestOffsetAttr(0.0)
    print(f"Added analytical Sphere collider at: {sphere_path} to protect the glass dome.")

    # 9. Apply/update mass on the rigid body
    mass_val = args.mass
    target_rb = rb_prim if rb_prim else mesh_prim
    if not target_rb.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(target_rb)
    mass_api = UsdPhysics.MassAPI.Apply(target_rb)
    mass_api.CreateMassAttr(mass_val)
    print(f"Applied mass: {mass_val} kg to rigid body: {target_rb.GetPath()}")

    print(f"Exporting flattened, modified stage to: {output_path}")
    flat_stage.GetRootLayer().Export(output_path)
    print("Success.")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
