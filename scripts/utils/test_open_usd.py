#!/usr/bin/env python3
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test opening lamp bulb USDs.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom

def inspect_file(name, path):
    print(f"\n==========================================")
    print(f"Testing {name}: {path}")
    print(f"File exists: {os.path.exists(path)}")
    try:
        stage = Usd.Stage.Open(path)
        if stage is None:
            print("Failed to open stage: returned None")
            return
        print("Successfully opened stage.")
        print(f"Root prims: {[p.GetName() for p in stage.GetPseudoRoot().GetChildren()]}")
        
        mesh_prims = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)
        
        print(f"Found {len(mesh_prims)} Mesh prim(s):")
        for prim in mesh_prims:
            print(f"  Path: {prim.GetPath()}")
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            print(f"    Points: {len(pts) if pts else 0}")
            
        # Inspect composition/references on root prims
        for prim in stage.GetPseudoRoot().GetChildren():
            print(f"Root Prim '{prim.GetName()}' info:")
            query = prim.GetPrimStack()
            for spec in query:
                print(f"  Spec path: {spec.path}")
                if spec.hasReferences:
                    for ref in spec.referenceList.prependedItems:
                        print(f"    Prepend Reference: assetPath='{ref.assetPath}', primPath='{ref.primPath}'")
                if spec.hasPayloads:
                    for pay in spec.payloadList.prependedItems:
                        print(f"    Prepend Payload: assetPath='{pay.assetPath}', primPath='{pay.primPath}'")
    except Exception as e:
        print(f"Error opening or traversing file: {e}")

if __name__ == "__main__":
    try:
        inspect_file("lamp_bulb.usd", "assets/Props/lamp_bulb/lamp_bulb.usd")
        inspect_file("lamp_bulb_physics.usd", "assets/Props/lamp_bulb/configuration/lamp_bulb_physics.usd")
    finally:
        simulation_app.close()
