"""
network_viz_context.py — the dependency-injection seam between the network
visualizer and SimulatorApp.

NetworkVizContext doesn't fit any of the visualizer's four responsibility
categories (data/layout, rendering, interaction/editing, serialization) — it's
composition-root infrastructure, so it gets its own small file rather than
being force-fit into one of them.
"""


class NetworkVizContext:
    """Narrow interface exposed to NetworkVisualizerWindow instead of a full
    SimulatorApp back-reference. Read-only passthroughs for live circuit/brain
    data (the visualizer mutates the returned objects directly — that's its
    job), plus a handful of delegated operations."""

    def __init__(self, app):
        self._app = app

    @property
    def circuit(self):     return self._app.circuit
    @property
    def brain(self):       return self._app.brain
    @property
    def brain_mgr(self):   return self._app.brain_mgr
    @property
    def sim_cfg(self):     return self._app.sim_cfg
    @property
    def bot_pos(self):     return self._app.bot_pos

    @property
    def connection_params(self):
        return self._app._connection_params

    @connection_params.setter
    def connection_params(self, value):
        self._app._connection_params = value

    def load_brain(self):
        self._app.load_brain()

    def add_joint(self):
        self._app._add_joint()

    def rebuild_channels(self):
        self._app._rebuild_channels()

    def rebuild_brain_params(self):
        self._app._rebuild_brain_params()

    def toggle_osc_layer(self, name):
        self._app._toggle_osc_layer(name)

    def tracked_osc_items(self):
        return self._app._osc_ctrl._osc_items

    def sync_body_after_removal(self, poses, bodies):
        self._app._arena.update_child_bodies(poses, bodies, self._app.sim_cfg)

    def sync_bodies(self):
        """Redraw the robot's child bodies in the arena from the current circuit."""
        from rigid_body import world_poses
        c = self._app.circuit
        poses = world_poses(self._app.bot_pos, c.bodies, c.joints)
        self._app._arena.update_child_bodies(poses, c.bodies, self._app.sim_cfg)

    def notify_closed(self):
        self._app._net_viz = None
