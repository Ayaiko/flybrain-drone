"""Box trial adapted from flybrain-drone/indoor_poc/716_voltage/voltage_box.py. Effective loaded thrust is an input."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent
DT=.05
R=.091
G=9.81
# Coordinates x (right), y (forward), z (height). One central floor-mounted box.
BOX=np.array([-.19,.19,1.18,1.60,1.1])
def cast(pos,heading):
    direction=np.array([np.sin(heading),np.cos(heading)])
    lo=np.array([BOX[0],BOX[2]])
    hi=np.array([BOX[1],BOX[3]])
    t0=(lo-pos[:2])/np.where(abs(direction)<1e-10,1e-10,direction)
    t1=(hi-pos[:2])/np.where(abs(direction)<1e-10,1e-10,direction)
    near=max(0.,max(np.minimum(t0,t1)))
    return near if near<=min(np.maximum(t0,t1)) and pos[2]<BOX[4] else 2.

def clearance(p):
    x,y,z=p
    return np.linalg.norm([max(BOX[0]-x,x-BOX[1],0),max(BOX[2]-y,y-BOX[3],0),max(z-BOX[4],0)])-R

def trial(mass, motor_gf, seed, interface=None):
    rng=np.random.default_rng(seed)
    p=np.array([rng.uniform(-.055,.055),0.,.65])
    vel=np.zeros(3)
    # Pair hardware spread across masses and voltages; split sensor noise from vehicle state.
    gain=rng.uniform(.97,1.03)
    yaw=0.
    mode='approach'
    goal_yaw=0.
    queue=[]
    readings=[]
    speed=np.zeros(2)
    offset=0.
    min_gap=5.
    stopped=None
    brake_pos=None
    stop_dist=None
    hist=[]
    result='timeout'
    choices={}
    is_still=0
    for tick in range(1200):
        t=tick*DT
        thrust = 4 * motor_gf * gain / mass * G
        # IMU-derived short-term yaw/velocity estimates; bounded estimator errors assumed.
        sensed_yaw=yaw+rng.normal(0,.008)
        sensed_vel=vel[:2]+rng.normal(0,.008,2)
        baro=p[2]+rng.normal(0,.015)
        # SHT31 and BH1750 are payload/telemetry; several samples per second suffice here.
        if tick%10==0:
            temperature,humidity,lux=25+rng.normal(0,.2),55+rng.normal(0,.8),400+rng.normal(0,8)
        if tick%2==0:
            ray=min(cast(p,yaw+angle) for angle in (-.07,0,.07))
            value=np.clip(ray+rng.normal(0,.012),.035,2)
            delay=int(rng.integers(2,5))
            if rng.random()>.06:
                queue.append((tick+delay,tick,sensed_yaw,value))
        arrivals=[s for s in queue if s[0]<=tick]
        queue=[s for s in queue if s[0]>tick]
        if arrivals:
            readings.extend(arrivals)
        readings=[s for s in readings if tick-s[1]<=6]
        recent=[s for s in readings if abs(s[2]-sensed_yaw)<.035]
        distance=max(recent,key=lambda s:s[1])[3] if recent else None
        if interface is not None:
            distance = interface.range_sample(tick, distance)
        clear_view = True
        speed[:]=0
        if mode=='approach':
            if distance is not None and distance<.47:
                mode='brake'
                brake_pos=p.copy()
            elif distance is not None:
                speed[1]=.35
        elif mode=='brake':
            is_still=is_still+1 if np.linalg.norm(sensed_vel)<.035 else 0
            if is_still>=3:
                stopped=clearance(p)
                stop_dist=p[1]-brake_pos[1]
                mode='scan left'
                goal_yaw=-np.deg2rad(55)
        elif mode in ('scan left','scan right'):
            samples=[s for s in readings if abs(s[2]-goal_yaw)<.035]
            if abs(sensed_yaw-goal_yaw)<.035 and len(samples)>=2:
                if clear_view is not None:
                    choices[mode]=min(s[3] for s in samples) if clear_view else 0.
                if clear_view is None:
                    pass
                elif mode=='scan left':
                    mode='scan right'
                    goal_yaw=np.deg2rad(55)
                else:
                    side=max(choices,key=choices.get)
                    if choices[side]<.85:
                        result='held - no free side'
                        break
                    goal_yaw=np.deg2rad(-55 if side=='scan left' else 55)
                    mode='detour'
        elif mode=='detour':
            if abs(sensed_yaw-goal_yaw)<.055 and distance is not None and distance>.27 and clear_view:
                speed[:]=.25*np.array([np.sin(goal_yaw),np.cos(goal_yaw)])
            offset+=sensed_vel[0]*DT
            if abs(offset)>=.42:
                goal_yaw=0.
                mode='pass'
        else:
            if abs(sensed_yaw)<.055 and distance is not None and distance>.28 and clear_view:
                speed[1]=.35
        if interface is not None:
            speed, goal_yaw = interface.deliver(tick, speed, goal_yaw, yaw)
        yaw+=np.clip(goal_yaw-yaw,-1.5*DT,1.5*DT)
        # Vertical priority, up to 12 degrees tilt. Horizontal thrust remaining after lift.
        vertical=np.clip(G+5*(.65-baro)-3*vel[2],0,thrust)
        horiz_limit=min(vertical*np.tan(np.deg2rad(12)),np.sqrt(max(thrust**2-vertical**2,0)))
        desired=(speed-vel[:2])/.18
        acc_xy=desired*min(1.,horiz_limit/max(np.linalg.norm(desired),1e-8))
        acc=np.r_[acc_xy,vertical-G]
        p+=vel*DT+.5*acc*DT**2
        vel+=acc*DT
        gap=clearance(p)
        min_gap=min(min_gap,gap)
        hist.append((p.copy(),yaw,mode,distance,motor_gf))
        if gap<=0: result='box contact';break
        if p[2]<.5: result='altitude lost';break
        if abs(p[0])+R>1: result='wall contact';break
        if p[1]>=2.15: result='finish' if stopped is not None else 'missed stop';break
    return dict(mass_g=mass,motor_gf=motor_gf,seed=seed,result=result,
                stop_gap_m=stopped,braking_travel_m=stop_dist,min_box_gap_m=min_gap,
                seconds=len(hist)*DT),hist

def point(p):
    x,y,z=p
    return int(95+x*100+y*66),int(195+x*29-y*20-z*100)

def panel(hist,tick,r):
    idx=min(tick,len(hist)-1)
    p,yaw,mode,d,motor_gf=hist[idx]
    im=Image.new('RGB',(370,255),'white')
    g=ImageDraw.Draw(im)
    for x in (-1,-.5,0,.5,1):g.line([point((x,0,0)),point((x,2.4,0))],fill='silver')
    for y in np.arange(0,2.5,.4):g.line([point((-1,y,0)),point((1,y,0))],fill='silver')
    l,h,a,b,z=BOX
    for face,c in [([(h,a,0),(h,b,0),(h,b,z),(h,a,z)],'#999999'),
                   ([(l,a,0),(h,a,0),(h,a,z),(l,a,z)],'#bbbbbb'),
                   ([(l,a,z),(h,a,z),(h,b,z),(l,b,z)],'#dddddd')]:
        g.polygon([point(v) for v in face],fill=c,outline='black')
    line=[point(v[0]) for v in hist[:idx+1:2]]
    if len(line)>1:g.line(line,fill='#13577c',width=2)
    x,y=point(p)
    g.ellipse((x-4,y-4,x+4,y+4),fill='black')
    if d is not None:g.line([(x,y),point(p+min(d,.5)*np.array([np.sin(yaw),np.cos(yaw),0]))],fill='#00917a',width=2)
    g.text((8,5),f"{r['mass_g']} g | {motor_gf} gf/motor | seed {r['seed']}",fill='black')
    g.text((8,220),f"{idx*DT:.1f} s   {mode if idx<len(hist)-1 else r['result']}",fill='black')
    return im

if __name__ == '__main__':
    import csv
    from collections import Counter

    rows = []
    for mass in (56, 59, 62):
        for motor_gf in (13, 14, 15, 16, 18, 22):
            batch = [trial(mass, motor_gf, seed)[0] for seed in range(12)]
            rows.extend(batch)
            print(mass, motor_gf, Counter(r['result'] for r in batch))
    with (OUT / 'trials.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    examples = [trial(62, thrust, 0) for thrust in (15, 18)]
    frames = []
    for tick in range(0, max(len(h) for _, h in examples) + 4, 4):
        frame = Image.new('RGB', (740, 255), 'white')
        for column, (result, history) in enumerate(examples):
            frame.paste(panel(history, tick, result), (370 * column, 0))
        frames.append(frame)
    frames[0].save(OUT / 'comparison.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
