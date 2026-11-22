"""Grab one frame from each fly-eye camera (inside the container) and print brightness stats."""
import time, numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
got = {}
def cb(i):
    def f(msg): got[i] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    return f
n = Node()
for i in range(3): n.subscribe(Image, f'/flyeye/cam{i}', cb(i))
t = time.time()
while len(got) < 3 and time.time() - t < 10: time.sleep(.1)
for i in sorted(got):
    g = got[i].mean(2).mean(0); print(i, got[i].shape, 'mean', round(g.mean(), 1), 'std', round(g.std(), 1), 'profile', np.round(g[::16]).astype(int).tolist())
np.save('/work/px4/grab.npy', np.stack([got[i] for i in range(3)]) if len(got) == 3 else np.zeros(1))
