Self driving Car Navigation (PyQt5 + DQN)

This project implements a **2D autonomous car navigation simulator** using **PyQt5** for visualization and **Deep Q-Learning (DQN)** with **PyTorch** for decision making. The agent must navigate on a valid road region, avoid off-road areas, and reach **three targets sequentially**.

**Target order:** `A1 → A2 → A3`
**Episode ends after reaching A3 or on crash.**

---

## Features

* Interactive **GUI-based simulator** (PyQt5)
* **Deep Q-Network (DQN)** agent implemented in PyTorch
* **Extra fully connected layer** added to the neural network
* Exactly **3 targets enforced**: A1, A2, A3
* **Sequential target navigation** (no looping back to A1)
* Episode termination on:

  * Crash (off-road)
  * Successful reach of A3
* Safe handling of Qt graphics objects (prevents deleted-object runtime errors)

---

## Neural Network Architecture

The DQN architecture used by the agent is:

```
Input (9)
 → FC(128) → ReLU
 → FC(256) → ReLU
 → FC(256) → ReLU
 → FC(128) → ReLU
 → FC(64)  → ReLU   ← extra layer
 → FC(Output = 5 actions)
```

---

## State, Actions, and Rewards

### State Space (9 values)

* 7 distance-based sensor readings
* Normalized angle to current target
* Normalized distance to current target

### Action Space (5 actions)

| Action | Meaning     |
| ------ | ----------- |
| 0      | Turn Left   |
| 1      | Go Straight |
| 2      | Turn Right  |
| 3      | Sharp Left  |
| 4      | Sharp Right |

### Reward Design (high level)

* `-0.1` per step (encourage efficiency)
* `+100` for reaching A1 or A2
* `+100` for reaching A3 (episode success)
* `-100` for crashing
* Additional shaping rewards based on progress and terrain

---

## Environment Rules

* The map is interpreted using **pixel brightness**:

  * **Bright pixels** → drivable road
  * **Dark pixels** → off-road (crash)
* Crash detection is done by sampling the pixel under the car center

---

## Installation

### Requirements

* Python 3.9+ (tested with 3.10–3.12)
* PyQt5
* PyTorch
* NumPy

### Install dependencies

```bash
pip install pyqt5 torch numpy
```

(Conda users may install PyQt via conda-forge.)

---

## How to Run

1. Save the main script as:

```
citymap_assignment.py
```

2. (Optional) Place a map image in the same directory:

```
city_map.png
```

If no map is found, the program automatically generates a dummy map.

3. Run the simulator:

```bash
python citymap_assignment.py
```

---

## Controls & Usage Flow

### Setup Steps (mandatory order)

1. **Left-click once** on the map to place the **car start position**
2. **Left-click three times** to place targets:

   * First click → A1
   * Second click → A2
   * Third click → A3
3. **Right-click** to confirm target placement
4. Press **SPACE** or click **START** to begin the episode

### Buttons

* **START / PAUSE / RESUME** – control simulation
* **RESET ALL** – clears environment and training state
* **LOAD MAP** – load a new map image

---

## Target Logic

* Targets must be reached **strictly in order**:

```
A1 → A2 → A3
```

* After reaching **A3**, the episode ends
* On reset, the next episode always starts again at **A1**

---

## Important Implementation Notes

### SensorItem RuntimeError Fix

Qt automatically deletes graphics items when `scene.clear()` is called.
This code safely recreates all sensor items **after** clearing the scene to avoid:

```
RuntimeError: wrapped C/C++ object of type SensorItem has been deleted
```

---

## Tunable Parameters

You can experiment with these parameters at the top of the file:

* `SPEED`
* `TURN_SPEED`
* `SHARP_TURN`
* `SENSOR_DIST`
* `BATCH_SIZE`
* `GAMMA`
* `LR`
* `TAU`
* `MAX_CONSECUTIVE_CRASHES`

Tip: If the agent struggles with sharp turns, reduce `SPEED` or increase `SHARP_TURN`.

---

## Suggested Folder Structure

```
project/
├── RL_car.py
├── city_map.png   (optional)
└── README.md
```

---

## Learning Objective

This project demonstrates:

* Deep Reinforcement Learning (DQN)
* State design using sensor fusion
* Reward shaping
* Sequential goal navigation
* Safe integration of RL logic with GUI frameworks

---

