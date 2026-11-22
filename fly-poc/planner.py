"""Local velocity planner for the fixed-heading, forward-goal indoor simulator."""
from dataclasses import dataclass
from math import hypot, isfinite

@dataclass(frozen=True)
class Airframe:
    radius: float = .24
    cruise: float = .8
    acceleration: float = 1.6
    brake: float = 1.6
    margin: float = .15
    frame_timeout: float = .25

@dataclass(frozen=True)
class Command:
    lateral: float
    forward: float
    reason: str

def plan(scores, captured_at, now, velocity, airframe, front_range=None):
    age = now - captured_at
    if not 0 <= age <= airframe.frame_timeout:
        return Command(0, 0, 'ภาพเก่า')
    if len(scores) != 4 or not all(map(isfinite, scores)):
        return Command(0, 0, 'คะแนนผิดปกติ')
    if not all(map(isfinite, velocity)):
        return Command(0, 0, 'ข้อมูลความเร็วผิดปกติ')
    action = max(range(4), key=scores.__getitem__)
    if action == 3:
        return Command(0, 0, 'โมเดลเสนอให้หยุด')
    forward = airframe.cruise * (1 if action == 0 else .59)
    lateral = (0, -.95, .95)[action] * airframe.cruise
    if front_range is not None:
        if not isfinite(front_range) or front_range < 0:
            return Command(0, 0, 'ข้อมูลระยะผิดปกติ')
        speed = max(hypot(*velocity), hypot(lateral, forward))
        reserve = airframe.radius + airframe.margin + speed * (age + .1)
        if front_range <= reserve + speed**2 / (2 * airframe.brake):
            return Command(0, 0, 'ระยะด้านหน้าไม่พอเบรก')
    return Command(lateral, forward, 'ตามทางที่โมเดลเสนอ')
