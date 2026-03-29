import argparse
from pathlib import Path
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meshes_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    meshes_dir = Path(args.meshes_dir)
    out_dir = Path(args.out_dir)
    out_mesh_dir = out_dir / 'meshes'
    out_mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(meshes_dir.glob('group_*.ply'))
    if not mesh_files:
        raise SystemExit('No group_*.ply meshes found')

    # Convert to OBJ
    mesh_assets = []
    for mf in mesh_files:
        mesh = trimesh.load(mf, force='mesh')
        if mesh.is_empty:
            continue
        obj_name = mf.stem + '.obj'
        obj_path = out_mesh_dir / obj_name
        mesh.export(obj_path)
        mesh_assets.append(obj_name)

    # Build MuJoCo XML
    xml = []
    xml.append('<mujoco model="scene">')
    xml.append('  <compiler angle="degree" inertiafromgeom="true"/>')
    xml.append('  <option timestep="0.01" gravity="0 0 -9.81"/>')
    xml.append('  <asset>')
    for obj in mesh_assets:
        xml.append(f'    <mesh name="{obj[:-4]}" file="meshes/{obj}"/>')
    xml.append('  </asset>')
    xml.append('  <worldbody>')
    xml.append('    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>')
    for obj in mesh_assets:
        name = obj[:-4]
        xml.append(f'    <body name="{name}" pos="0 0 0">')
        xml.append(f'      <geom type="mesh" mesh="{name}" mass="1" friction="0.6 0.1 0.1" rgba="0.7 0.7 0.9 1"/>')
        xml.append('    </body>')
    xml.append('  </worldbody>')
    xml.append('</mujoco>')

    out_xml = out_dir / 'scene.xml'
    out_xml.write_text('\n'.join(xml))
    print('Wrote', out_xml)


if __name__ == '__main__':
    main()
