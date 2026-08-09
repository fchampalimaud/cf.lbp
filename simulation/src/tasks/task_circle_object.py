"""
task_circle_object.py — a small object (robot-sized) orbits the arena centre.

The object moves in a circle at a configurable radius and period. It acts as
a physical obstacle: the robot's CollisionSensor and DistanceSensor detect it,
and the physics engine pushes the robot away on contact.
"""

import numpy as np
from tasks.task_base import BaseTask


class CircleObject(BaseTask):
    name        = 'circle_object'
    description = 'A robot-sized object orbits the arena centre at a fixed radius and period.'

    ORBIT_FRACTION = 0.6   # orbit radius as fraction of arena_scale
    PERIOD         = 30   # seconds per full revolution

    def setup(self, world, sim_cfg):
        self._t   = 0.0
        r_obj     = sim_cfg.body_radius
        orbit_r   = sim_cfg.arena_scale * self.ORBIT_FRACTION
        self._obj = {
            'x':     orbit_r,
            'y':     0.0,
            'r':     r_obj,
            'color': [0.2, 0.7, 1.0],   # light blue
        }
        world.objects.append(self._obj)

    def tick(self, world, bot_pos, sim_cfg, dt):
        self._t  += dt
        orbit_r   = sim_cfg.arena_scale * self.ORBIT_FRACTION
        angle     = 2.0 * np.pi * self._t / self.PERIOD
        self._obj['x'] = orbit_r * np.cos(angle)
        self._obj['y'] = orbit_r * np.sin(angle)

    def reset(self, world, sim_cfg):
        # Remove any stale entry before re-adding
        if self._obj in world.objects:
            world.objects.remove(self._obj)
        self.setup(world, sim_cfg)
