"""Inside the container, with a <name>_preview world running: save the preview cameras and the 3 fly-eye cameras as
PNGs in px4/forest/preview/<name>_*.png.   python3 px4/grab_preview.py <name> [n_cams]"""
import sys, time
from pathlib import Path
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
name, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
out = Path('/work/px4/forest/preview'); out.mkdir(parents=True, exist_ok=True); got = {}
def cb(k):
    def f(msg): got[k] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3].copy()
    return f
node = Node()
for i in range(n): node.subscribe(Image, f'/preview/cam{i}', cb(f'view{i}'))
for i in range(3): node.subscribe(Image, f'/flyeye/cam{i}', cb(f'eye{i}'))
t = time.time()
while len(got) < n + 3 and time.time() - t < 60: time.sleep(.2)
from PIL import Image as PI
for k, a in got.items(): PI.fromarray(a).save(out / f'{name}_{k}.png')
print('saved', sorted(got))
