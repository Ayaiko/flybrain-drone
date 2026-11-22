"""Replay TUM fr1/desk at measured recording speed. Requires ffmpeg, numpy and run.py dependencies."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
from run import ROOT, bias, names, png, read_eye
from gate_trial import VisionGate

def select_window(poses, image_times):
    # Compare 0.25 s chord speeds to the target; reject tracking gaps.
    best = None
    for start in np.arange(max(poses[0,0], image_times[0])+.6,
                           min(poses[-1,0], image_times[-1])-4.05,.05):
        times = start+np.arange(17)*.25
        segment = poses[(poses[:,0]>=start-.1)&(poses[:,0]<=times[-1]+.1)]
        if len(segment)<2 or np.diff(segment[:,0]).max()>.1:
            continue
        coords = np.array([np.interp(times,poses[:,0],axis) for axis in poses[:,1:4].T]).T
        speeds = np.linalg.norm(np.diff(coords,axis=0),axis=1)/.25
        loss = np.mean((speeds-.52)**2)
        if best is None or loss<best[0]:
            best = loss,float(start),speeds
    if best is None:
        raise ValueError('No continuous four-second motion-capture window')
    return best[1:]

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('archive', help='TUM archive for timestamped RGB anchors (first RGB entries suffice)')
    cli.add_argument('video')
    cli.add_argument('trajectory')
    cli.add_argument('--ffmpeg', default='ffmpeg')
    args = cli.parse_args()
    out = ROOT/'real-desk'
    out.mkdir(exist_ok=True)
    poses = np.loadtxt(args.trajectory)
    def ffmpeg(arguments, data=None):
        return subprocess.run([args.ffmpeg,'-v','error',*arguments],input=data,
                              stdout=subprocess.PIPE,check=True).stdout
    images = {}
    with tarfile.open(args.archive,'r|gz') as archive:
        for member in archive:
            path = Path(member.name)
            if path.parent.name=='rgb' and path.suffix=='.png':
                images[float(path.stem)] = archive.extractfile(member).read()
                if len(images)==48:
                    break
    def thumbnails(source,data=None):
        raw = ffmpeg([*source,'-vf','scale=40:30:flags=area','-pix_fmt','gray','-f','rawvideo','pipe:1'],data)
        return np.frombuffer(raw,'u1').reshape(-1,1200).astype('f4')
    anchors = sorted(images)
    original = thumbnails(['-f','image2pipe','-vcodec','png','-i','pipe:0'],b''.join(images[t] for t in anchors))
    movie = thumbnails(['-i',args.video])
    matched = []
    for stamp,thumb in zip(anchors,original):
        distances = np.mean(abs(movie-thumb),axis=1)
        index = int(np.argmin(distances))
        if distances[index]<5:
            matched.append((index,stamp))
    indices,stamps = np.array(matched).T
    if len(indices)<20 or min(np.diff(indices))<=0:
        raise ValueError('Insufficient consistent frame timestamp matches')
    available = np.interp(np.arange(len(movie)),indices,stamps)
    start,speeds = select_window(poses,stamps)
    targets = start+np.arange(-6,40)*.1
    ids = [int(np.abs(available-t).argmin()) for t in targets]
    selected = available[ids]
    error = np.max(np.abs(selected-targets))
    if error>.055 or len(set(ids))!=len(ids):
        raise ValueError(f'RGB timestamps inadequate: {error}')
    def select_frames(indices):
        return 'select='+ '+'.join(f'eq(n\\,{i})' for i in indices)
    ffmpeg(['-i',args.video,'-vf',select_frames(ids[6:])+',setpts=N/(10*TB)','-r','10',
            '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(out/'clip.mp4')])
    pixels = ffmpeg(['-i',args.video,'-vf',select_frames(ids)+',scale=80:48:flags=area',
                     '-fps_mode','passthrough','-pix_fmt','gray','-f','rawvideo','pipe:1'])
    frames = np.frombuffer(pixels,'u1').reshape(len(selected),48,80)
    gate,state = VisionGate(),bias.copy()
    rows,preview = [],[]
    with np.load(ROOT/'readout.npz') as model:
        for i,(img,stamp) in enumerate(zip(frames,selected)):
            feat = read_eye(img,state)
            action = names[int(np.argmax(np.r_[(feat-model['mean'])/model['sd'],1.]@model['coef']))]
            if i<6:
                continue
            cmd = gate.command(img)
            rows.append(dict(t=round((i-6)*.1,2),timestamp=stamp,flyvis=action,gate=cmd.reason,
                             lateral=cmd.lateral,forward=cmd.forward))
            preview.append(png(img))
    stats = dict(source='TUM RGB-D freiburg1_desk (Sturm et al., 2012)',
                 source_url='https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',
                 video_url="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk-rgb.avi",
                 matched_anchors=matched,avi_indices=ids[6:],start=start,segment_s=4,target_mps=.52,mean_mps=float(speeds.mean()),quarter_second_mps=speeds.tolist(),
                 max_rgb_time_error_s=float(error),flyvis=dict(Counter(r['flyvis'] for r in rows)),
                 gate=dict(Counter(r['gate'] for r in rows)),frames=rows)
    (out/'results.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    html = """<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ทดสอบภาพจริง · TUM Desk</title><style>body{font:17px system-ui;max-width:980px;margin:3em auto;padding:1em;background:#edf0eb;color:#182720}main{display:flex;gap:12px;align-items:center}video{width:65%}img{width:32%;image-rendering:pixelated}output{display:block;margin-top:1em}</style><h1>ภาพจริงจากกล้องในสำนักงาน</h1><p>TUM RGB-D · Freiburg1 Desk · ภาพต้นทางและภาพขาวดำที่โมเดลได้รับ</p><main><video controls playsinline src="clip.mp4"></video><img alt="ภาพป้อนเข้าโมเดล"></main><output></output><script>
    const frames=FRAMES,imgs=IMAGES,v=document.querySelector('video');
    function draw(){const i=Math.min(39,Math.floor(v.currentTime*10)),r=frames[i];document.querySelector('img').src='data:image/png;base64,'+imgs[i];document.querySelector('output').textContent=r.t.toFixed(1)+' s · FlyVis '+r.flyvis+' · ตัวตรวจช่อง '+r.gate+' · ความเร็วที่เสนอ (ข้าง, หน้า) '+r.lateral.toFixed(2)+', '+r.forward.toFixed(2)+' m/s';}
    v.addEventListener('timeupdate',draw);v.addEventListener('seeked',draw);function update(){draw();if(!v.paused)requestAnimationFrame(update)}v.addEventListener('play',update);draw();</script></html>"""
    (out/'replay.html').write_text(html.replace('FRAMES',json.dumps(rows)).replace('IMAGES',json.dumps(preview)))
    print({k:stats[k] for k in ['start','mean_mps','max_rgb_time_error_s','flyvis','gate']},speeds.min(),speeds.max())

if __name__=='__main__':
    main()
