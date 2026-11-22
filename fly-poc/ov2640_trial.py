"""Compare the native fly-poc camera with a hypothetical ESP32-CAM JPEG stream."""
import io
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from PIL import Image
import run
from courses import courses

# Keep the original horizontal FOV. Capture 4:3, then crop to the model aspect.
WIDTH, HEIGHT = 160, 120
RAYS = np.meshgrid((np.arange(WIDTH) + .5 - WIDTH / 2) / (WIDTH / 2),
                   (HEIGHT / 2 - np.arange(HEIGHT) - .5) / (WIDTH / 2))
native_frame = run.scene_frame


def jpeg_frame(obstacles, x, y, rng):
    with patch.multiple(run, dx=RAYS[0], dz=RAYS[1]):
        raw = native_frame(obstacles, x, y, rng)
    buffer = io.BytesIO()
    Image.fromarray(raw).save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    with Image.open(buffer) as image:
        cropped = image.crop((0, 12, WIDTH, 108))
        return np.asarray(cropped.resize((run.W, run.H), Image.Resampling.BOX))


def main():
    results, examples = [], []
    with np.load(run.ROOT / 'readout.npz') as weights:
        model = tuple(weights[key] for key in ('mean', 'sd', 'coef'))
    for course in ('กล่องสามใบ', 'ประตูคู่เยื้องแนว'):
        for profile in ('กล้องเดิม', 'OV2640 จำลอง'):
            with ExitStack() as stack:
                if profile == 'OV2640 จำลอง':
                    stack.enter_context(patch.object(run, 'scene_frame', jpeg_frame))
                for seed in range(12000, 12004):
                    trial = run.evaluate(model, None, seed, seed == 12000, course=courses[course])
                    results.append(dict(course=course, profile=profile, seed=seed,
                                        result=trial['result'], end_y=trial['end_y'],
                                        clearance=trial['min_clearance']))
                    if trial['frames']:
                        trial['name'] = f'{course} · {profile}'
                        examples.append(trial)
            print(course, profile, [(r['result'], r['end_y']) for r in results[-4:]], flush=True)
    (run.ROOT / 'ov2640-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    template = (run.ROOT / 'viewer.template').read_text()
    a, b = template.index('<option>'), template.index('</select>')
    options = ''.join(f"<option>{trial['name']}</option>" for trial in examples)
    template = (template[:a] + options + template[b:]).replace('กล้องหน้าของโดรนจำลอง', 'เปรียบเทียบกล้องเดิมกับ OV2640 จำลอง')
    (run.ROOT / 'ov2640-replay.html').write_text(template.replace('REPLAY_DATA', json.dumps(examples)))


if __name__ == '__main__':
    main()
