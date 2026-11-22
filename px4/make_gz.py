"""Gazebo assets for PX4 SITL: the 2D forest layout as a world (striped posts + walls, same textures as Webots) and an
x500 with a 'fly eye' made of 3 cameras (110 deg each, 160x8 px, yaw -110/0/+110 -> 330 deg), which the controller
resamples to the 120 evenly spaced ommatidia of the 2D sim.
Run from the project root (host): python -m px4.make_gz [seed]"""
import sys
from pathlib import Path
import numpy as np
from flydrone.world import World
from webots.make_world import textures

ROOT = Path(__file__).resolve().parents[1]; PX = ROOT / 'px4/PX4-Autopilot/Tools/simulation/gz'
TEX = '/work/webots/worlds/textures'          # path inside the container
CAM_W, CAM_H, CAM_FOV, CAM_YAWS = 160, 8, np.radians(110), (-110, 0, 110)


def mat(png, scale=None):
    return (f'<material><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular><pbr><metal><albedo_map>{TEX}/{png}</albedo_map>'
            f'<roughness>1</roughness><metalness>0</metalness></metal></pbr></material>')


def model():
    cams = ''.join(f'''
      <sensor name="eye_{i}" type="camera">
        <pose>0 0 0 0 0 {np.radians(y):.5f}</pose>
        <camera><horizontal_fov>{CAM_FOV:.5f}</horizontal_fov><image><width>{CAM_W}</width><height>{CAM_H}</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>80</far></clip></camera>
        <always_on>1</always_on><update_rate>30</update_rate><topic>flyeye/cam{i}</topic>
      </sensor>''' for i, y in enumerate(CAM_YAWS))
    d = PX / 'models/x500_flyeye'; d.mkdir(parents=True, exist_ok=True)
    (d / 'model.config').write_text('<?xml version="1.0"?><model><name>x500_flyeye</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>'
                                    '<description>x500 with a 330 deg fly eye (3 cameras)</description></model>\n')
    (d / 'model.sdf').write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<sdf version='1.9'>
  <model name='x500_flyeye'>
    <include merge='true'><uri>x500</uri></include>
    <link name="flyeye_link">
      <pose relative_to="base_link">0 0 0.18 0 0 0</pose>
      <inertial><mass>0.02</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz></inertia></inertial>{cams}
    </link>
    <joint name="flyeye_joint" type="fixed"><parent>base_link</parent><child>flyeye_link</child></joint>
  </model>
</sdf>
''')


def world(seed=0):
    textures(); w = World(seed=seed); L = w.size; o = w.start
    base = (PX / 'worlds/default.sdf').read_text(encoding='utf-8-sig').replace('<world name="default">', f'<world name="forest_{seed}">')
    parts = []
    for i, (c, r, k, ph) in enumerate(zip(w.c, w.r, w.k, w.ph)):
        x, y = c - o
        parts.append(f'''<model name="post{i}"><static>true</static><pose>{x:.3f} {y:.3f} 2.5 0 0 {ph:.3f}</pose><link name="l">
  <visual name="v"><geometry><cylinder><radius>{r:.3f}</radius><length>5</length></cylinder></geometry>{mat(f"post_k{k}.png")}</visual></link></model>''')
    for i, (cx, cy, sx, sy) in enumerate([(L / 2 - o[0], -o[1], L, .1), (L / 2 - o[0], L - o[1], L, .1), (-o[0], L / 2 - o[1], .1, L), (L - o[0], L / 2 - o[1], .1, L)]):
        parts.append(f'''<model name="wall{i}"><static>true</static><pose>{cx:.2f} {cy:.2f} 2.5 0 0 0</pose><link name="l">
  <visual name="v"><geometry><box><size>{sx:.2f} {sy:.2f} 5</size></box></geometry>{mat("wall.png")}</visual></link></model>''')
    base = base.replace('<shadows>true</shadows>', '<shadows>false</shadows>').replace('<cast_shadows>true</cast_shadows>', '<cast_shadows>false</cast_shadows>')
    s = base.replace('</world>', '\n'.join(parts) + '\n  </world>')
    out = PX / f'worlds/forest_{seed}.sdf'; out.write_text(s); print('saved', out); return out


if __name__ == '__main__':
    model(); world(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
