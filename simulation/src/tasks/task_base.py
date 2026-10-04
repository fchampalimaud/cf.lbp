"""
task_base.py — contract for pluggable simulation tasks.

A Task runs alongside the simulation each tick. Tasks can move objects,
shift gradients, emit stimuli, or do anything else that modifies world state.

Minimal implementation:

    from tasks.task_base import BaseTask

    class SweepGradient(BaseTask):
        name = 'sweep_gradient'
        description = 'Moves gradient A back and forth across the arena.'

        def setup(self, world, sim_cfg):
            self._t = 0.0

        def tick(self, world, bot_positions, sim_cfg, dt):
            self._t += dt
            for patch in world.patches:
                if patch.get('label') == 'A':
                    patch['x'] = sim_cfg.arena_scale * 0.8 * np.sin(self._t)
"""


class BaseTask:
    """
    Base class for all simulation tasks.

    Subclass and override setup() and tick(). Optionally override reset()
    if you need to re-initialise state without creating a new instance.
    """
    name        = 'task'
    description = ''

    def setup(self, world, sim_cfg):
        """Called once when the task is activated (or on simulation reset)."""
        pass

    def tick(self, world, bot_positions, sim_cfg, dt):
        """
        Called once per simulation step, after every agent has moved.

        Parameters
        ----------
        world         : World — grants access to world.patches and world.objects
        bot_positions : list of [x, y, theta], one per agent (in agent order),
                        read-only (do not mutate)
        sim_cfg       : SimConfig
        dt            : float — time step in seconds
        """
        pass

    def reset(self, world, sim_cfg):
        """Called on simulation reset. Defaults to re-running setup()."""
        self.setup(world, sim_cfg)
