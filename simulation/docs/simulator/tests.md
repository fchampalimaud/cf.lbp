# Unit Tests

A file-by-file, test-by-test listing of the simulator's automated test suite under `simulation/2d/tests/`. Pair this with [Class Reference](reference.md) when you need to know exactly what behaviour a class is expected to guarantee.

```bash
cd simulation/2d
pip install pytest pytest-qt
pytest tests/ -v
```

Tests that need a Qt event loop (visualiser build, drag/drop, freeze regressions) require `pytest-qt` and are skipped automatically if it isn't installed. `test_mujoco_build_xml_textures` is skipped automatically if `mujoco` isn't installed.

---

## `tests/test_smoke.py`

Core layer math, serialisation, and visualiser smoke tests.

| Test | Verifies |
|---|---|
| `test_leaky_decays_to_zero` | `LeakyLayer` output decays toward zero with no input |
| `test_leaky_converges_to_input` | `LeakyLayer` output converges to a constant target under linear activation |
| `test_leaky_relu_clamps_negative` | Default relu activation clamps negative output to zero |
| `test_hard_sigmoid_matches_torch_and_numpy_paths` | `hard_sigmoid` matches PyTorch's own `F.hardsigmoid` on the torch path and agrees exactly with the numpy path used by sensors — `_activate`'s two branches never silently diverge |
| `test_leaky_tau_decay_none_holds_value` | `tau_decay=None` disables decay entirely — value holds when input drops |
| `test_layer_output_mode_derivative_is_rate_of_change` | `output_mode='derivative'` (the generic `DynamicsBase` mechanism shared by every layer/sensor) replaces the raw summed input u with the finite-difference rate of change between consecutive ticks' raw input, before the leaky filter/activation run; layers don't special-case the first tick (`_prev_out` starts at a zero buffer), unlike sensors |
| `test_layer_output_mode_integral_accumulates` | `output_mode='integral'` replaces the raw summed input u with the running ∫u dt (forward-Euler), before the leaky filter/activation run |
| `test_output_mode_transforms_raw_input_not_leaky_filtered_output` | Regression: `output_mode` must transform the raw input u before the leaky filter, not the layer's final output — with a real leaky lag (`tau_rise=tau_decay=0.1`), the internal `_integral` buffer must stop growing the instant the raw input hits zero, even though the leaky filter's own state is still decaying |
| `test_output_mode_invalid_raises` | `DynamicsBase._init_dynamics` raises `ValueError` for an `output_mode` outside `{none, derivative, integral}`, mirroring `_activate`'s unknown-activation error |
| `test_sensor_output_mode_derivative_and_integral` | `BaseSensor._apply_output_mode` (via `_process()`) supports the same none/derivative/integral contract as layers, applied to the raw reading before the leaky filter/activation, but with sensor-specific zero-on-first-tick semantics for `derivative` |
| `test_modulator_row_transform_derivative` | `LayerBase._transform_modulator_value`'s `'derivative'` mode computes the rate of change of a raw modulator-bus reading, keyed by `(name, mode)` in `self._mod_row_state` — independent of the layer's own `output_mode` state, zero on the first call for a given key |
| `test_modulator_row_transform_integral` | `'integral'` mode accumulates a raw modulator reading over time (forward-Euler), same math as `_apply_output_mode`'s integral branch but tracked per `(name, mode)` key instead of a single per-layer buffer |
| `test_modulator_row_state_independent_per_key` | Two different `(name, mode)` modulator subscriptions on the same layer don't share state, and modulator-row state doesn't collide with the layer's own `output_mode` state, even when both happen to be `'derivative'` |
| `test_learning_layer_drives_plasticity_row_gates_on_threshold` | A `modulators` row flagged `drives_plasticity=True` only contributes to reward — and therefore only triggers a weight update — on ticks where its mode-transformed value crosses `threshold`, the generalized replacement for the old always-on `reward_modulator` field |
| `test_learning_layer_reward_modulator_backcompat_default_is_none` | `reward_modulator` now defaults to `None` (not `'dopamine'`) so a freshly constructed learning layer never silently picks up a stray reward channel via the legacy field |
| `test_learning_layer_reward_modulator_explicit_value_still_works` | Back-compat: a layer constructed with an explicit `reward_modulator` (as happens loading old saved JSON, which always serialized this field before it left `param_defs()`) still drives plasticity via that legacy channel, additively with any `drives_plasticity` rows |
| `test_snapshot_layer_teach_connection_excluded_from_output` | `SnapshotLayer`'s teach connection (identified by `conn.src == teach_source`) never contributes to its own output — only gate connections do, regardless of the teach source's value |
| `test_snapshot_layer_write_overwrites_outgoing_weights_exactly` | On a `drives_plasticity` trigger, every outgoing connection's weights are hard-set to `-teach_val` (Le Moël Eq. 14) — not nudged by a Hebbian rule |
| `test_snapshot_layer_outgoing_weights_frozen_between_triggers` | Outgoing weights stay exactly as last written on ticks where the reward doesn't cross threshold, regardless of how much the teach source's value drifts in the meantime |
| `test_connection_kind_classifies_teach_before_td` | `_connection_kind` returns `TEACH` for the `src==teach_source` connection into a `SnapshotLayer`, and `TD` for that same layer's other (gate) incoming connection — `TEACH` is checked with higher priority since a `SnapshotLayer` is itself a `LearningLayerBase` |
| `test_reichardt_integer_shift_matches_bilinear` | `Reichardt2dLayer`'s fast integer-pixel shift path matches the general bilinear `grid_sample` path bit-for-bit, for both axis-aligned and fractional direction sets |
| `test_reset_zeroes_state` | `reset()` zeroes layer output regardless of prior activity |
| `test_ensure_n_raises_on_mismatch` | `_ensure_n` raises `ValueError` when called with a different `n` than already declared |
| `test_ensure_n_initialises_when_none` | `_ensure_n` initialises buffers on first call when `n` was unset at construction |
| `test_x0_seeds_initial_state` | `x0` seeds `_x` and the `output_mode='integral'` accumulator at construction, both when `n` is given immediately and via the deferred `_ensure_n` path |
| `test_x0_per_neuron_list` | `x0` accepts a per-neuron list, not just a scalar broadcast |
| `test_x0_wrong_length_raises` | A per-neuron `x0` list whose length doesn't match `n` raises `ValueError` |
| `test_x0_restored_on_reset_not_zeroed` | `reset()` returns `_x` to `x0`, not to a hardcoded zero — otherwise a bounded integrator (e.g. `x0=0.5`) would come back from reset at the wrong end of its range |
| `test_x0_seeds_integral_accumulator_not_bias` | `x0` sets the one-time starting point of an `output_mode='integral'` accumulator; a zero raw input leaves it exactly at `x0` rather than drifting the way a nonzero `bias` would |
| `test_collision_sensor_silent_when_never_hit` | `CollisionSensor` produces zero output while never registering a hit, even with `noise_std > 0` (regression: a second, ungated noise pass in `_process()` used to leak through) |
| `test_step_network_size_reconciliation_is_gated` | `step_network`'s size-reconciliation pass only re-runs when the connections list identity changes, not every tick (regression for the O(connections) caching fix) |
| `test_product_layer_multiplies_connections` | `ProductLayer` combines incoming connections by elementwise product, not sum (with `tau_rise=0.0` to isolate the fan-in contract from leaky filtering); a `ProductLayer` with zero connections outputs all-ones (not all-zeros) before bias/dynamics/activation/scale |
| `test_product_layer_has_leaky_dynamics` | `ProductLayer` is a `DynamicsBase`/`LeakyLayer` subclass with real leaky dynamics (tau_rise/tau_decay/noise/bias/activation/scale), not an instantaneous SumLayer-style combinator — a tiny-dt step lands strictly between the starting output and the fully-converged product |
| `test_bonsai_export_rejects_product_layer` | `generate_bonsai_xml` raises `ValueError` naming the offending layer(s) when the circuit contains a `ProductLayer`, since LBP.Torch's `JoinAdditive` has no multiplicative equivalent |
| `test_layer_from_dict_backward_compat_derivative` | `_layer_from_dict` maps an old JSON boolean `derivative` field (the retired `LeakyLayer`/`ProductLayer`/`Leaky2dLayer`-only x-vs-u mode) onto the new `output_mode` |
| `test_sensor_from_dict_backward_compat_differential` | `_sensor_from_dict` maps an old JSON boolean `differential` field (any sensor type) onto the new `output_mode` |
| `test_layer_to_dict_persists_auto_injected_output_mode` | `_layer_to_dict` persists `output_mode` even for layers (e.g. `AdaptiveLayer`) that don't list it in their own `param_defs()` — it's only reachable via the dialog's `DynamicsBase` auto-injection, so serialization must merge the same fallback or silently lose the value on reload |
| `test_serialisation_round_trip` | Save → load preserves layer params and connection weight shape |
| `test_build_no_crash` | `NetworkVisualizerWindow.build()` doesn't raise on a minimal circuit |
| `test_weight_matrix_cosine_pattern_is_circulant` | `WeightMatrixDialog`'s Cosine pattern produces a circulant (rotation-invariant) matrix, constant along i-j diagonals like its Gaussian/Mexican-hat siblings — regression for a sign bug that banded along i+j anti-diagonals instead (visually "rotated perpendicular") |
| `test_activation_panel_pin_and_update` | Pinning a layer via `_toggle_activation_entry` shows the Activations panel and adds an `ActivationEntryWidget`; `_update_activation_panel` populates its bar chart with the layer's current per-neuron `output`, indexed 0..n-1; unpinning removes it |
| `test_side_view_drag_preserves_sensor_z` | Dragging a sensor container in the side view updates its `z`, not just its column (regression: cross-column and same-column drag handlers used to leave `sensor.z` frozen) |
| `test_paste_selection_bumps_connections_identity` | `_paste_selection` gives `circuit.connections` a fresh list identity after appending, so the `conn_id`-gated caches in `step_network` don't silently miss pasted connections (regression: `.append()` alone doesn't change `id()`) |
| `test_world_serializer_round_trip` | `save_world_file` → `load_world_file` preserves world state, including per-object/per-wall/floor texture assignments |
| `test_mujoco_build_xml_textures` | MuJoCo XML builder dedupes repeated texture files into one `<texture>`/`<material>` pair, uses `material=` for textured geometry and `rgba=` for untextured, and produces byte-identical output when no textures are used |
| `test_collision_sensor_vectorized_matches_reference_loop` | `CollisionSensor._detect_hits`'s numpy-vectorized geometry matches the original nested-loop implementation exactly, across round/square arenas and 2-point/triangle/square-polygon walls |
| `test_mujoco_collision_eligible_gating` | `_mujoco_collision_eligible` accepts any root-mounted `CollisionSensor` regardless of radius (dedicated geometry is built at exactly that sensor's own probe radius) — only a sensor mounted off the root body correctly falls back to the analytic path |
| `test_log_collision_sensor_routing_reports_correct_path` | `log_collision_sensor_routing` prints, per `CollisionSensor`, which path it actually uses and why (MuJoCo contact path, or analytic — not root-mounted) — the only way to confirm the routing decision from the running app's console instead of reading the code |
| `test_mujoco_collision_sensor_fires_on_contact_not_before` | A literal-touch `CollisionSensor` (`radius=1.0`) and a lookahead one (`radius=1.2`, the default), sampled side by side via `MuJoCoEngine.tick_physics` against their own dedicated per-sector geometry, each fire exactly at their own correct threshold — lookahead fires strictly earlier than touch, never later, and neither fires from open space |
| `test_mujoco_collision_matches_analytic_randomized` | The MuJoCo per-sector-geometry collision path disagrees with the analytic reference (`CollisionSensor._detect_hits`) on fewer than 10% of randomized scenes (objects, walls, round/square arenas, varying `n`/`angle_spread`/`arc_angle`/`radius`) — the two approximate the same geometry differently (real MuJoCo collision shapes vs. discrete point-probes), so this is a tolerance check, not bit-for-bit equality |

## `tests/test_viz_freeze.py`

Regression tests guarding against a real UI freeze bug: the network visualiser's scatter-plot rendering silently stalling the Qt event loop. See the module docstring for what these tests can and cannot verify — a real OS-level focus switch cannot be reproduced in headless Qt, so `freeze_repro.py` must be run manually to check that case.

| Test | Verifies |
|---|---|
| `test_autorange_off_after_disable` | `disableAutoRange()` disables both axes on the `ViewBox` |
| `test_setdata_does_not_reenable_autorange` | `ScatterPlotItem.setData()` doesn't silently re-enable auto-range |
| `test_setdata_speed` | `setData()` with 30 nodes completes in under 20 ms (p95) |
| `test_no_stall_10hz_updates` | 10 Hz scatter updates don't stall the Qt event loop |
| `test_reentrancy_guard` | Reentrant `setData()` calls (as triggered from `_redraw_nodes()`) don't deadlock or crash |
| `test_no_stall_simulated_focus_switches` | Qt-side window activate/raise handling during scatter updates doesn't stall the event loop |
