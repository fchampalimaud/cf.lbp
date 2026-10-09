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
| `test_leaky_tau_decay_none_is_instant` | `tau_decay=None` snaps the value straight to the target the instant input drops (no hold, no lag) |
| `test_leaky_tau_rise_none_is_instant` | `tau_rise=None` snaps straight to the target the instant input rises, then decays normally with `tau_decay` — a fast-attack/slow-decay envelope |
| `test_leaky_both_taus_none_is_full_passthrough` | Both taus unset: no state tracking at all, output equals input exactly |
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
| `test_learning_rate_scales_with_dt` | Learning rates are per tick at the default dt 0.01 and scale with dt: one tick at dt 0.02 learns twice what one tick at 0.01 does, and the GUI's dt (0.010000000000000002) gives exactly the default result |
| `test_td_credits_previous_input_delta_current` | `TDLayer` credits the previous tick's input (nothing learned on the first tick); `DeltaLayer` credits the current input |
| `test_fast_taus_reports_only_taus_shorter_than_dt` | `fast_taus` reports taus shorter than dt (and leaves them untouched), ignores 0 / blank taus; the warning text says "blow up" when dt/τ > 2 |
| `test_mute_keeps_layer_state_and_constant_comes_back` | Muting zeroes a layer's output without touching its state: an `AccumulatorLayer` resumes from what it had accumulated, and a `ConstantLayer`'s value is back after unmute |
| `test_learning_layer_bias_integral_and_late_tau_rise` | A learning layer adds `bias` with `tau_rise = 0`, runs with `output_mode='integral'`, and keeps working when `tau_rise` is raised after construction |
| `test_modulator_rows_sharing_mode_advance_once_per_tick` | A `pre` and a `post` row on the same modulator and mode advance their shared derivative state once per tick, so both see the same non-zero derivative |
| `test_codegen_keeps_every_non_default_param` | `init_code_parts` (export as a Python brain) round-trips activation, scale, noise, `x0`, `alpha` (and Matsuoka's `tau_decay`) for Matsuoka / Pulse / RingAttractor layers |
| `test_every_dynamics_layer_applies_noise_and_scale` | Noise and scale come from the shared layer pipeline, so they work on every layer — `noise_std` changes Pulse / Conv2d / Delta output, and RingAttractor's output is multiplied by `scale` |
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
| `test_activation_panel_supports_sensors` | A sensor (which has no `.output` attribute) can be pinned too — `update_activation_panel` reads its current reading from `brain.<sensor.name>`, set each tick by `sim_engine.step_agents` |
| `test_side_view_drag_preserves_sensor_z` | Dragging a sensor container in the side view updates its `z`, not just its column (regression: cross-column and same-column drag handlers used to leave `sensor.z` frozen) |
| `test_paste_selection_bumps_connections_identity` | `_paste_selection` gives `circuit.connections` a fresh list identity after appending, so the `conn_id`-gated caches in `step_network` don't silently miss pasted connections (regression: `.append()` alone doesn't change `id()`) |
| `test_world_serializer_round_trip` | `save_world_file` → `load_world_file` preserves world state, including per-object/per-wall/floor texture assignments |
| `test_mujoco_build_xml_textures` | MuJoCo XML builder dedupes repeated texture files into one `<texture>`/`<material>` pair, uses `material=` for textured geometry and `rgba=` for untextured, and produces byte-identical output when no textures are used |
| `test_collision_sensor_vectorized_matches_reference_loop` | `CollisionSensor._detect_hits`'s numpy-vectorized geometry matches the original nested-loop implementation exactly, across round/square arenas and 2-point/triangle/square-polygon walls |
| `test_mujoco_collision_eligible_gating` | `_mujoco_collision_eligible` accepts any root-mounted `CollisionSensor` regardless of radius (dedicated geometry is built at exactly that sensor's own probe radius) — only a sensor mounted off the root body correctly falls back to the analytic path |
| `test_log_collision_sensor_routing_reports_correct_path` | `log_collision_sensor_routing` prints, per `CollisionSensor`, which path it actually uses and why (MuJoCo contact path, or analytic — not root-mounted) — the only way to confirm the routing decision from the running app's console instead of reading the code |
| `test_mujoco_collision_sensor_fires_on_contact_not_before` | A literal-touch `CollisionSensor` (`radius=1.0`) and a lookahead one (`radius=1.2`, the default), sampled side by side via `sim_engine.step_agents` against their own dedicated per-sector geometry, each fire exactly at their own correct threshold — lookahead fires strictly earlier than touch, never later, and neither fires from open space |
| `test_mujoco_collision_matches_analytic_randomized` | The MuJoCo per-sector-geometry collision path disagrees with the analytic reference (`CollisionSensor._detect_hits`) on fewer than 10% of randomized scenes (objects, walls, round/square arenas, varying `n`/`angle_spread`/`arc_angle`/`radius`) — the two approximate the same geometry differently (real MuJoCo collision shapes vs. discrete point-probes), so this is a tolerance check, not bit-for-bit equality |
| `test_step_agents_distance_sensor_sees_other_agent` | `step_agents` passes each agent a pre-step snapshot of the other agents' bodies, so a `DistanceSensor` sees another robot as an obstacle |
| `test_step_agents_senses_everything_before_the_brain_runs` | Within one step, all sensing (2-D fields, MuJoCo contacts, due cameras) happens before `brain.loop`, and movement after it — the brain never sees a contact one step late |
| `test_step_agents_keyboard_command_replaces_wheels_and_shows_in_motor_layer` | A keyboard/network command becomes that agent's wheel command, the brain still runs, and the command is written into the motor layer (`rules/motor_commands.md`) |
| `test_camera_fps_schedule_runs_on_sim_time` | A camera with `fps > 0` renders once per `1/fps` simulated seconds regardless of step size; `fps = 0` renders whenever sim time has advanced; `reset()` restarts the schedule |
| `test_camera_process_frame_output_formats` | A gray camera's plain output is the frame's centre row; an RGB camera's is the full CHW frame; lateralized halves use the same layout |
| `test_editor_undo_restores_state_from_before_the_edit` | An edit's undo snapshot is taken before anything changes, so weight settings and weights changed in one edit are both undone |
| `test_editor_no_change_records_no_undo_step_and_nesting_is_one_step` | A transaction that changes nothing (e.g. a cancelled dialog) records no undo step; nested transactions make one step |
| `test_editor_undo_restores_bodies_and_joints` | Undo brings bodies and joints back to their state before the edit |
| `test_editor_syncs_brain_after_add_and_rename` | After an edit the brain sees renamed and added layers by name (stale names removed), and `step_network` keeps running after a layer is appended in place |
| `test_editor_history_is_per_circuit_bounded_and_clearable` | Undo history lives on the circuit (a new editor sees it), is capped at 50 steps, and `clear_history` empties it |
| `test_network_window_remove_layer_undo_keeps_brain_runnable` | Through the real network window: removing a layer and undoing it restores the circuit and brain, toggles the Undo button, and the brain keeps running |
| `test_session_brain_entry_code_and_network_formats` | Session group entries: a network is written as `mode: network` + `network: project/file`, a code brain as `mode: code` + `module_name`; both read back to the same brain module + params, and the old `BrainGUI` form reads the same as the new one |
| `test_find_mirror_follows_lateral_pairs_only` | `find_mirror` finds lateral→lateral and camera-halves→shared-target mirrors, and ignores a plain layer whose name merely ends in `_L` |
| `test_rename_layer_renames_lateral_pair_and_weight_settings` | Renaming one half renames the pair (`foo_L`/`foo_R`), both `lateral_pair` links, the connections and the saved weight settings |
| `test_unpair_layer_turns_pair_back_into_one_layer` | `unpair_layer` keeps the edited half under the base name and removes the partner, its connections and its weight settings |
| `test_layer_edit_lateralized_off_removes_partner` | Through the layer edit: switching `lateralized` off turns `conv_L`/`conv_R` into one layer `conv` |
| `test_column_renumbering_keeps_hidden_flags_and_set_identity` | Inserting a column in the side view shifts the hidden / disabled flags with their column, and the sets stay the same objects (the app saves them) |
| `test_network_window_delete_connection_removes_mirror_and_column_disable` | Deleting a connection also deletes its mirror (camera halves → one target); disabling a column mutes only that column's layers |
| `test_new_layer_type_needs_only_its_own_file` | A layer type defined only in the test file (params, step, reset — nothing else) saves/loads, runs in `step_network` and renders in the network window, with default capabilities — no other module needs to know it exists |
| `test_lateral_helpers` | `lateral.py`: side/base/mirror/half names, `partner_layer`, `parent_sensor`, camera vs body-pair halves, `is_lateral_half` |
| `test_load_syncs_every_lateral_pair_param` | Loading an L/R pair copies the `_L` side's params to `_R` for every pair-capable layer type (Leaky2dLayer used to be skipped) |
| `test_sensors_and_layers_share_dynamics` | A sensor and a LeakyLayer fed the same input give the same output; either blank tau makes that side instantaneous, and both blank means no filtering at all — for sensors as for layers |
| `test_derivative_output_mode_is_zero_on_first_step` | `output_mode='derivative'` outputs 0 on the first step (and after reset) for layers and sensors alike |
| `test_freshness_check_satisfied_after_resave` | The "Network file outdated" check doesn't report settings that saving leaves out by design (e.g. `GradientSensor.gradient` unset), but still reports genuinely missing ones |
| `test_engines_satisfy_engine_protocol` | `MuJoCoEngine` and the test stand-in engine both provide every method `step_agents` needs (`sim_engine.Engine`), so the stand-in can't drift from the real engine |
| `test_robot_mode_runs_brain_loop_and_sends_its_command` | Robot mode runs the brain's `loop()` (so code-only brains work) and sends the command it returns to every motor layer's `robot_address`; the keyboard takes priority, is written into the motor layer, and the brain keeps running |
| `test_simulation_step_ticks_task_once_with_all_agents` | `Simulation.step` ticks the task once per step and passes every agent's position, not just the selected agent's |
| `test_headless_session_loads_all_agents_and_runs` | `session_loader.build_simulation` builds every agent group of a saved session without the GUI (multi-agent tutorial: 3 agents, all with brains), and the result can be stepped |
| `test_simulation_runs_without_qt` | Importing the simulation core (`simulation`, `sim_engine`, `agent_registry`, `session_loader`, `headless`) in a fresh interpreter never imports PySide6 or pyqtgraph |
| `test_step_agents_renders_due_cameras_only` | `step_agents` renders `fps > 0` cameras on their schedule and leaves `fps = 0` cameras to the display loop |

## `tests/test_shipped_files.py`

Backward-compatibility guard: every session in `configs/` and every network in `networks/` (run in an empty world) is loaded and simulated for 1 s with a fixed seed. One test per file (`latest_session.json`, the app's per-user state, is skipped).

| Test | Verifies |
|---|---|
| `test_shipped_file_loads_runs_and_matches_reference[<file>]` | The file loads, runs without errors or NaN, and every robot ends at the pose recorded in `tests/reference/shipped_reference.json` (tolerance 1e-6). Each reference entry stores a hash of the files it reads (session + networks); if you edit one, that entry is skipped instead of failed until you regenerate — so the test fails only when *code* changes how an unchanged file behaves |

After an intended behaviour change, regenerate the reference from `simulation/2d/`:

```bash
python tests/test_shipped_files.py --update
```

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
