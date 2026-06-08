#!/usr/bin/env python3
import argparse
import os
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate SDF collision USD for the local lighter asset."
)

parser.add_argument(
    "--input",
    type=str,
    default="assets/Props/lighter/lighter.usd",
    help="Input USD path.",
)

parser.add_argument(
    "--output",
    type=str,
    default="assets/Props/lighter_sdf.usd",
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
    sdf_api.CreateSdfMarginAttr(0.001)
    sdf_api.CreateSdfNarrowBandThicknessAttr(0.002)
    sdf_api.CreateSdfSubgridResolutionAttr(6)

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision.CreateContactOffsetAttr(0.002)
    physx_collision.CreateRestOffsetAttr(0.0)
    physx_collision.CreateMinTorsionalPatchRadiusAttr(0.005)

def main():
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Opening local lighter USD: {input_path}")
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

    # 3. Find the main root prim
    root_prim = None
    for prim in flat_stage.GetPseudoRoot().GetChildren():
        if prim.GetName() != "Render":
            root_prim = prim
            break
    if not root_prim:
        raise RuntimeError("No root prim found on the stage.")
    root_path = root_prim.GetPath()
    print(f"Using root prim: {root_path}")

    # 4. Remove all root prims except the main one and Render
    for prim in flat_stage.GetPseudoRoot().GetChildren():
        if prim.GetName() not in [root_prim.GetName(), "Render"]:
            print(f"Removing unused root prim: {prim.GetPath()}")
            flat_stage.RemovePrim(prim.GetPath())

    # 5. Remove joints to prevent articulation warnings
    # 5. Find all visual mesh prims using a safe recursive traversal
    print("Finding visual mesh prims...")
    visual_mesh_prims = []
    
    def collect_visual_meshes(prim):
        if prim.GetTypeName() == "Mesh":
            path_str = str(prim.GetPath())
            if "/visuals/" in path_str:
                visual_mesh_prims.append(prim)
        for child in prim.GetChildren():
            collect_visual_meshes(child)
            
    collect_visual_meshes(root_prim)

    # 6. Remove joints to convert articulation to single rigid body
    joints_path = root_path.AppendChild("joints")
    if flat_stage.GetPrimAtPath(joints_path).IsValid():
        print(f"Removing joints prim: {joints_path} to convert to single rigid body")
        flat_stage.RemovePrim(joints_path)

    # 7. Merge all visual meshes to single collision mesh arrays
    print(f"Merging {len(visual_mesh_prims)} visual meshes...")
    merged_points = []
    merged_indices = []
    merged_counts = []
    
    xform_cache = UsdGeom.XformCache()
    root_to_world = xform_cache.GetLocalToWorldTransform(root_prim)
    root_to_world_inv = root_to_world.GetInverse()
    
    for mesh_prim in visual_mesh_prims:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        
        if not points or not indices or not counts:
            continue
            
        mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
        local_to_root = mesh_to_world * root_to_world_inv
        
        vertex_offset = len(merged_points)
        
        for p in points:
            p_root = local_to_root.Transform(Gf.Vec3d(*p))
            merged_points.append(Gf.Vec3f(*(float(v * args.target_scale) for v in p_root)))
            
        for idx in indices:
            merged_indices.append(idx + vertex_offset)
            
        for count in counts:
            merged_counts.append(count)

    # 8. Clean up old collision prims and RigidBody/Mass APIs from sub-links
    for child in root_prim.GetChildren():
        # Remove collisions prim if it exists
        collisions_path = child.GetPath().AppendChild("collisions")
        if flat_stage.GetPrimAtPath(collisions_path).IsValid():
            print(f"Removing old collisions: {collisions_path}")
            flat_stage.RemovePrim(collisions_path)
            
        # Remove physics APIs
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if child.HasAPI(UsdPhysics.MassAPI):
            child.RemoveAPI(UsdPhysics.MassAPI)

    # 9. Create single merged collision mesh under root
    collisions_parent_path = root_path.AppendChild("collisions")
    if flat_stage.GetPrimAtPath(collisions_parent_path).IsValid():
        flat_stage.RemovePrim(collisions_parent_path)
        
    UsdGeom.Xform.Define(flat_stage, collisions_parent_path)
    
    collision_mesh_path = collisions_parent_path.AppendChild("mesh")
    collision_mesh = UsdGeom.Mesh.Define(flat_stage, collision_mesh_path)
    
    collision_mesh.GetPointsAttr().Set(Vt.Vec3fArray(merged_points))
    collision_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(merged_indices))
    collision_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(merged_counts))
    collision_mesh.GetExtentAttr().Set(_compute_extent(merged_points))
    
    print(f"Applying SDF collision to the single merged collision mesh...")
    _apply_sdf_collision(collision_mesh.GetPrim(), args.sdf_resolution)

    # 10. Compute center of mass and diagonal inertia from bounding box of merged points
    xs = [p[0] for p in merged_points]
    ys = [p[1] for p in merged_points]
    zs = [p[2] for p in merged_points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    w = max_x - min_x
    h = max_y - min_y
    d = max_z - min_z
    
    center = Gf.Vec3f((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0)
    
    m = args.mass
    ixx = (1.0 / 12.0) * m * (h**2 + d**2)
    iyy = (1.0 / 12.0) * m * (w**2 + d**2)
    izz = (1.0 / 12.0) * m * (w**2 + h**2)
    diagonal_inertia = Gf.Vec3f(ixx, iyy, izz)
    
    print(f"BBox dimensions: width={w:.4f}, height={h:.4f}, depth={d:.4f}")
    print(f"Computed Center of Mass: {center}")
    print(f"Computed Diagonal Inertia: {diagonal_inertia}")

    # 11. Apply RigidBodyAPI and MassAPI to the root prim to make it the single rigid body
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        print(f"Applied RigidBodyAPI to root prim: {root_path}")
    
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(m)
    mass_api.CreateCenterOfMassAttr(center)
    mass_api.CreateDiagonalInertiaAttr(diagonal_inertia)
    print(f"Applied mass properties to root prim: {root_path}")

    # Export flattened, upgraded stage
    print(f"Exporting flattened, modified stage to: {output_path}")
    flat_stage.GetRootLayer().Export(output_path)
    print("Success.")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
