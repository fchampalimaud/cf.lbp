# MotorLayer

SumLayer computation (instantaneous weighted sum) with a **robot_address** that routes output to physical hardware in real-robot mode.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of motor outputs; default 2 (left wheel, right wheel) |
| `activation` | Nonlinearity; default `linear` |
| `scale` | Output multiplier; default 1.0 |
| `robot_address` | OSC target: `ip:port/osc_path`, e.g. `192.168.0.1:2390/wheels`. Leave empty for sim-only. |

## Output

$$\text{output} = f\!\left(\sum_k W_k \cdot \text{input}_k\right) \times s$$

In **sim mode** the output drives wheel velocity or joint angle. In **robot mode** values are packed into an OSC message and sent to `robot_address` each tick.

!!! note
    Manual-control override also writes through this layer so the circuit sees what the wheels are actually doing.
