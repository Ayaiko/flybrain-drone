"""Generate a Webots forest world with the same layout as the 2D sim (World(seed)): 40x40 m walled arena, striped posts.
Run from the project root: python -m webots.make_world [seed] [mode] [duration_s]   -> webots/worlds/forest_<seed>.wbt"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from flydrone.world import World

ROOT = Path(__file__).resolve().parents[1]; W = ROOT / 'webots/worlds'; TEX = W / 'textures'


def textures():
    TEX.mkdir(parents=True, exist_ok=True)
    x = np.arange(512) / 512
    for k in range(3, 7):    # same brightness profile as the 2D posts: 0.5 + 0.4 sin(k * angle)
        row = 0.5 + 0.4 * np.sin(2 * np.pi * k * x)
        Image.fromarray(np.tile((row * 255).astype(np.uint8), (32, 1))).convert('RGB').save(TEX / f'post_k{k}.png')
    row = 0.5 + 0.35 * np.sin(2 * np.pi * x)   # one wall period (1.2 m), repeated by TextureTransform
    Image.fromarray(np.tile((row * 255).astype(np.uint8), (32, 1))).convert('RGB').save(TEX / 'wall.png')


def make(seed=0, mode='lplc2', duration=120):
    textures(); w = World(seed=seed); L = w.size; o = w.start      # world origin = drone start
    posts = [dict(x=float(c[0] - o[0]), y=float(c[1] - o[1]), r=float(r), k=int(k), ph=float(ph)) for c, r, k, ph in zip(w.c, w.r, w.k, w.ph)]
    walls = dict(xmin=-o[0], xmax=L - o[0], ymin=-o[1], ymax=L - o[1])
    meta = dict(seed=seed, size=L, origin=[float(o[0]), float(o[1])], posts=posts, walls=walls)
    (W / f'forest_{seed}.json').write_text(json.dumps(meta))
    s = ['#VRML_SIM R2025a utf8',
         'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/dji/mavic/protos/Mavic2Pro.proto"',
         'WorldInfo { title "flybrain forest" basicTimeStep 8 defaultDamping Damping { linear 0.5 angular 0.5 } }',
         'Viewpoint { orientation -0.25 0.25 0.93 1.7 position 0 -14 9 follow "Mavic 2 PRO" followSmoothness 0.3 }',
         'Background { skyColor [ 0.62 0.66 0.72 ] }',
         'DirectionalLight { direction -0.4 0.3 -1 intensity 2.2 castShadows FALSE }',
         'Solid { translation %.2f %.2f -0.01 children [ Shape { appearance PBRAppearance { baseColor 0.42 0.42 0.4 roughness 1 metalness 0 } geometry DEF G Plane { size %.1f %.1f } } ] name "ground" boundingObject USE G }'
         % ((walls['xmin'] + walls['xmax']) / 2, (walls['ymin'] + walls['ymax']) / 2, L + 20, L + 20)]
    for i, p in enumerate(posts):
        s.append('Solid { translation %.3f %.3f 2.5 rotation 0 0 1 %.3f children [ Shape { appearance PBRAppearance { baseColorMap ImageTexture { url [ "textures/post_k%d.png" ] } roughness 1 metalness 0 } '
                 'geometry Cylinder { height 5 radius %.3f subdivision 32 } } ] name "post%d" }' % (p['x'], p['y'], p['ph'], p['k'], p['r'], i))
    for i, (cx, cy, sx, sy, rep) in enumerate([((walls['xmin'] + walls['xmax']) / 2, walls['ymin'], L, .1, L / 1.2), ((walls['xmin'] + walls['xmax']) / 2, walls['ymax'], L, .1, L / 1.2),
                                                (walls['xmin'], (walls['ymin'] + walls['ymax']) / 2, .1, L, L / 1.2), (walls['xmax'], (walls['ymin'] + walls['ymax']) / 2, .1, L, L / 1.2)]):
        s.append('Solid { translation %.2f %.2f 2.5 children [ Shape { appearance PBRAppearance { baseColorMap ImageTexture { url [ "textures/wall.png" ] } '
                 'textureTransform TextureTransform { scale %.2f 1 } roughness 1 metalness 0 } geometry Box { size %.2f %.2f 5 } } ] name "wall%d" }' % (cx, cy, rep, sx, sy, i))
    # fly eye: 480 x 8 px spherical camera covering 330 deg, looking forward (+x), 4 columns per ommatidium
    s.append('Mavic2Pro { translation 0 0 0.1 rotation 0 0 1 %.4f controller "flyvision_mavic" controllerArgs [ "%s" "%d" "%d" ] supervisor TRUE '
             'bodySlot [ Camera { translation 0.1 0 0.05 name "flyeye" width 480 height 8 projection "spherical" fieldOfView 5.7596 near 0.05 } ] }'
             % (np.pi / 2, mode, seed, duration))
    out = W / f'forest_{seed}_{mode}.wbt'; out.write_text('\n'.join(s) + '\n'); print('saved', out, len(posts), 'posts'); return out


if __name__ == '__main__':
    a = sys.argv[1:]; make(int(a[0]) if a else 0, a[1] if len(a) > 1 else 'lplc2', int(a[2]) if len(a) > 2 else 120)
