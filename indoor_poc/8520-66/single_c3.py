"""Single-controller level recovery, simulated IMU + four independent PWM channels.
Local small-angle roll/pitch dynamics; no yaw or horizontal navigation claim.
"""
import csv
import math
import random
from pathlib import Path

DT = .002
# X geometry: 57 mm radial distance from centre to each shaft.
ARM = .057 / math.sqrt(2)
SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))


def simulate(mass_g, thrust_gf, seed, bus_block):
    rng = random.Random(seed)
    mass = mass_g / 1000
    g = 9.81
    gains = [rng.uniform(.97, 1.03) for _ in SIGNS]
    hover = mass_g / (4 * thrust_gf)
    duty = [min(hover, 1.)] * 4
    actual = duty.copy()
    angle = [math.radians(10), math.radians(-5)]
    rate = [0., 0.]
    integ = [0., 0.]
    altitude, climb = .65, 0.
    floor = altitude
    inertia = mass * .038 ** 2
    peak_after = 0.
    missed = 0
    for tick in range(2500):
        if tick == 500:
            rate[0] += math.radians(90)
        if tick % 2 == 0:
            if bus_block and tick % 50 < 20:
                missed += 1
            else:
                # Near-level attitude estimate and gyro with bounded noise.
                correction = []
                for axis in (0, 1):
                    estimate = angle[axis] + math.radians(rng.uniform(-.4, .4))
                    gyro = rate[axis] + math.radians(rng.uniform(-1., 1.))
                    integ[axis] = max(-.25, min(.25, integ[axis] + estimate * .004))
                    correction.append(inertia * (-55 * estimate - 12 * gyro - 42 * integ[axis])
                                      / (4 * ARM * thrust_gf * g / 1000))
                base = hover * (1 + (5 * (.65 - altitude) - 3 * climb) / g)
                duty = [max(0., min(1., base + a * correction[0] + b * correction[1]))
                        for a, b in SIGNS]
        force = []
        for i in range(4):
            actual[i] += (duty[i] - actual[i]) * (1 - math.exp(-DT / .035))
            force.append(actual[i] * gains[i] * thrust_gf * g / 1000)
        for axis in (0, 1):
            torque = ARM * sum(sign[axis] * f for sign, f in zip(SIGNS, force))
            rate[axis] += torque / inertia * DT
            angle[axis] += rate[axis] * DT
        climb += (sum(force) * math.cos(angle[0]) * math.cos(angle[1]) / mass - g) * DT
        altitude += climb * DT
        floor = min(altitude, floor)
        if tick >= 1000:
            peak_after = max(peak_after, *(abs(math.degrees(x)) for x in angle))
        if altitude < .5:
            break
    return dict(mass_g=mass_g, thrust_gf=thrust_gf, seed=seed, blocked=bus_block,
                pass_level=floor >= .5 and peak_after < 3, min_height_m=floor,
                max_tilt_after_2s_deg=peak_after if tick >= 1000 else None,
                missed_updates=missed)


if __name__ == '__main__':
    rows = []
    for mass in (53, 56, 59, 62):
        for thrust in (14, 16, 18):
            for blocked in (False, True):
                batch = [simulate(mass, thrust, seed, blocked) for seed in range(12)]
                rows.extend(batch)
                print(mass, thrust, blocked, sum(r['pass_level'] for r in batch))
    with Path(__file__).with_name('single-c3-results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
