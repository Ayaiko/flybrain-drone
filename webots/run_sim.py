"""Run one headless Webots flight and return its summary. Kills leftover Webots processes before and after
(they sometimes linger after simulationQuit and block the next launch).
python -m webots.run_sim <mode old|lplc2|none> <seed> <duration_s> [ENV=val ...]"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; WB = '/Applications/Webots.app/Contents/MacOS/webots'


def kill(): subprocess.run(['pkill', '-9', '-f', 'MacOS/webots'], capture_output=True)


def run(mode, seed, duration, env=None, timeout=None):
    from webots.make_world import make
    kill(); world = make(seed, mode, duration); out = ROOT / f'webots/runs/{mode}_{seed}.json'
    t_start = time.time(); old = out.stat().st_mtime if out.exists() else 0
    e = dict(os.environ, **(env or {}))
    # launch through LaunchServices hidden (-j) and in the background (-g) so no window pops up or steals focus;
    # env vars go through open --env; the controller writes its result file, which is what we wait for
    cmd = ['open', '-n', '-g', '-j', '-a', '/Applications/Webots.app']
    for k, v in (env or {}).items(): cmd += ['--env', f'{k}={v}']
    cmd += ['--stdout', str(ROOT / 'webots/runs/wb.log'), '--stderr', str(ROOT / 'webots/runs/wb.log'),
            '--args', '--mode=fast', '--batch', '--minimize', '--no-rendering', '--stdout', '--stderr', str(world)]
    subprocess.run(cmd, check=True)
    timeout = timeout or 60 + duration * 2
    while time.time() - t_start < timeout:
        time.sleep(1)
        if out.exists() and out.stat().st_mtime > old and out.stat().st_mtime > t_start: break
    else:
        kill(); raise TimeoutError(f'webots run {mode} {seed} timed out; see webots/runs/wb.log')
    time.sleep(1); kill()
    return json.loads(out.read_text())['summary']


if __name__ == '__main__':
    a = sys.argv[1:]; env = dict(x.split('=', 1) for x in a[3:])
    print(json.dumps(run(a[0], int(a[1]), float(a[2]), env)))


def run_batch(jobs, env=None, per_job=90):
    """jobs: list of (mode, seed, duration). One hidden Webots launch runs them all, loading each world in turn."""
    from webots.make_world import make
    kill(); worlds = [str(make(seed, mode, dur)) for mode, seed, dur in jobs]
    outs = [ROOT / f'webots/runs/{mode}_{seed}.json' for mode, seed, _ in jobs]
    t_start = time.time()
    for o in outs:
        if o.exists(): o.unlink()
    (ROOT / 'webots/runs/queue.json').write_text(json.dumps(worlds[1:]))
    cmd = ['open', '-n', '-g', '-j', '-a', '/Applications/Webots.app']
    for k, v in (env or {}).items(): cmd += ['--env', f'{k}={v}']
    cmd += ['--stdout', str(ROOT / 'webots/runs/wb.log'), '--stderr', str(ROOT / 'webots/runs/wb.log'),
            '--args', '--mode=fast', '--batch', '--minimize', '--no-rendering', '--stdout', '--stderr', worlds[0]]
    subprocess.run(cmd, check=True)
    done_n, last_progress = 0, time.time()
    while True:
        time.sleep(2); n = sum(o.exists() for o in outs)
        if n > done_n: done_n, last_progress = n, time.time(); print(f'  {n}/{len(outs)} done', flush=True)
        if n == len(outs) or time.time() - last_progress > per_job: break
    time.sleep(1); kill()
    return [json.loads(o.read_text())['summary'] if o.exists() else None for o in outs]
