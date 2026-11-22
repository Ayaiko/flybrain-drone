"""Software-in-loop command boundary. Commands are velocity/yaw targets, not RC pulses."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from box_trial import trial


class Interface:
    def __init__(self, fault):
        self.fault = fault
        self.last_tick = -100
        self.speed = np.zeros(2)
        self.heading = 0.
        self.hold_ticks = 0

    def range_sample(self, tick, distance):
        if self.fault == 'range outage' and 30 <= tick < 65:
            return None
        return distance

    def deliver(self, tick, speed, heading, yaw):
        lost = (self.fault == 'link outage' and 30 <= tick < 65
                or self.fault == 'link disconnected' and tick >= 30)
        if not lost:
            self.speed = speed.copy()
            self.heading = heading
            self.last_tick = tick
        if tick - self.last_tick >= 3:
            self.hold_ticks += 1
            return np.zeros(2), yaw
        return self.speed.copy(), self.heading


def main():
    rows = []
    for fault in ('normal', 'range outage', 'link outage', 'link disconnected'):
        for seed in range(24):
            interface = Interface(fault)
            result, _ = trial(62, 18, seed, interface)
            rows.append(dict(fault=fault, hold_ticks=interface.hold_ticks, **result))
        batch = [r for r in rows if r['fault'] == fault]
        print(fault, Counter(r['result'] for r in batch),
              round(min(r['min_box_gap_m'] for r in batch), 3),
              min(r['hold_ticks'] for r in batch))
    with Path(__file__).with_name('interface-results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
