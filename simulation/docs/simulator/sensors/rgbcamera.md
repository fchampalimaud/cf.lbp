# RGBCameraSensor

![RGBCameraSensor](../../assets/figures/sensor_rgbcamera.svg)

Raycasts `width × height` pixels across the field of view and returns RGB colour (same raycasting as `GrayCameraSensor`, all 3 channels retained).

## Output shape (channels-first / CHW)

$$\text{output} \in \mathbb{R}^{3 \times H \times W} \quad \text{(flat: channel, row, col)}$$

## Lateralized mode

`lateralized=True`:

$$\text{sensor\_L} \in \mathbb{R}^{3 \times H \times (W/2 + \text{overlap})}, \quad \text{sensor\_R} \in \mathbb{R}^{3 \times H \times (W/2 + \text{overlap})}$$

Connect to a `Conv2dLayer` with `in_ch=3` (set automatically from camera mode).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 64 | Number of rays (horizontal resolution) |
| `height` | 48 | Tile rows (1 = single strip) |
| `fov` | 90° | Horizontal field of view |
| `center_angle` | 0° | Center offset from robot heading |
| `vertical_angle` | 0° | Camera tilt — positive = tilted down (ground), negative = tilted up (sky). Each image row sees a different ground distance: bottom rows closer, top rows farther. At 90° the camera points straight down and the image goes black. |
| `max_range` | 10.0 | Max ray length in world units |
| `lateralized` | False | Split output into left/right halves |
| `overlap` | 0 | Pixels past midline included in each half |
