# NeuralNav (TD3) — Stable Exploration + Hysteresis 

This project is a PyQt5-based 2D autonomous navigation simulator where a car learns to reach a **sequence of targets** (A1 → A2 → A3 …) using **TD3 (Twin Delayed DDPG)**.

It includes fixes for a common failure mode you observed: the agent doing “okay” when exploration is high, but getting worse when exploration becomes very low (e.g., “epsilon ~ 0.001”).

---

## 1) What this code does

### Core loop (high-level)

1. You click on the map to place the **car start position**
2. You click multiple times to place **targets** (A1, A2, A3…)
3. The agent runs in a loop:

   * Reads state from sensors + target direction
   * TD3 Actor outputs a **continuous action** `a ∈ [-1, 1]`
   * That action is mapped to **one of 5 discrete environment actions**
   * Environment moves the car (**physics unchanged**)
   * TD3 trains from replay buffer using Actor + Twin Critic + target smoothing

---

## 2) Important constraints this implementation respects

### ✅ Physics parameters are NOT changed

The car movement and turning behavior are controlled by:

* `SENSOR_DIST`, `SPEED`, `TURN_SPEED`, `SHARP_TURN`, etc.

These are used exactly as-is for all motion logic.

### ✅ Your provided RL hyperparameters are NOT changed

These are used unchanged:

* `BATCH_SIZE`
* `GAMMA`
* `LR`
* `TAU`
* `MAX_CONSECUTIVE_CRASHES`

---

## 3) State space & action space

### State vector (size = 9)

The agent observes:

* **7 sensor readings** (brightness/occupancy-like values sampled ahead of the car)
* **1 normalized angle-to-target**
* **1 normalized distance-to-target**

State shape:

```
[s0, s1, s2, s3, s4, s5, s6, norm_angle_to_target, norm_distance_to_target]
```

### Actions (environment uses discrete actions)

The environment executes 5 discrete actions (physics unchanged):

* `0` → left (`-TURN_SPEED`)
* `1` → straight (`0`)
* `2` → right (`+TURN_SPEED`)
* `3` → sharp left (`-SHARP_TURN`)
* `4` → sharp right (`+SHARP_TURN`)

### TD3 continuous action → discrete mapping

TD3 outputs a continuous action `a ∈ [-1, 1]`.

That value is mapped to the discrete action set using **hysteresis** (see below), to avoid flip-flopping near thresholds.

---

## 4) Why the old behavior degraded as “epsilon” got low

In your original DQN-style setup, epsilon controlled exploration.

In TD3:

* **epsilon is not the exploration driver**
* Exploration comes from adding **Gaussian noise** to the actor output.

What you observed (“bad at epsilon=0.001”) typically happens because:

* exploration disappears
* the learned policy is unstable/brittle
* action mapping is threshold-sensitive and starts oscillating

This README corresponds to the fixed version addressing that.

---

## 5) What fixes were implemented (and why they matter)

### Fix A — Stable exploration (not tied to epsilon)

**Problem:** When exploration noise becomes too small, the agent exposes its true policy — and if that policy is unstable, performance collapses.

**Fix:** The actor always gets at least a minimum exploration noise during training:

* `TD3_EXPL_NOISE_MIN` ensures noise never collapses to near-zero.

This makes training more stable and prevents the policy from becoming brittle too early.

---

### Fix B — Hysteresis-based continuous→discrete action mapping

**Problem:** If your mapping is purely threshold-based, small changes in `a` flip the discrete action:

* right ↔ sharp right
* left ↔ straight
* etc.

That leads to:

* oscillations
* jittery steering
* worse behavior when exploration is low

**Fix:** Hysteresis keeps decisions “sticky”:

* If you were already turning left, you don’t switch to straight unless the signal crosses a stronger boundary.
* If you were sharp turning, you don’t exit sharp turning unless the signal clearly weakens.

This reduces “boundary flip-flop”.

---

### Fix C — Priority replay based on real target progress

**Problem:** Early in training, “successful episodes” can be luck (noise), not true skill.
If those are oversampled, the agent learns unstable strategies.

**Fix:** An episode is prioritized only if it:

* made real target progress (e.g., reached at least the next target in the chain), AND
* ended alive

This filters out “lucky but sloppy” trajectories.

---

### Fix D — SensorItem deletion crash fix (PyQt)

**Problem:** `scene.clear()` deletes QGraphicsItems (including sensors). Later calling `setPos()` triggers:

> `RuntimeError: wrapped C/C++ object ... has been deleted`

**Fix:** Sensor items and car items are recreated after:

* `scene.clear()`
* `full_reset()`

---

## 6) How rewards work (summary)

The environment uses:

* **-100** on crash
* **+100** on reaching a target
* Small shaping rewards (based on sensor readings and distance progress)

These rewards remain consistent with your original intent.

---

## 7) How to run

### Requirements

* Python 3.x
* PyQt5
* PyTorch
* NumPy

Install dependencies:

```bash
pip install pyqt5 torch numpy
```

Run:

```bash
python your_file_name.py
```

---

## 8) How to use the simulator

### Step-by-step

1. Launch the app
2. Click on the map to place the **car start position**
3. Click to place **Target 1 (A1)**
4. Click again to place **Target 2 (A2)**
5. Click again to place **Target 3 (A3)**
6. Right-click to finish target placement
7. Press **Space** or click **START**

---

## 9) Understanding what you see on screen

### Sensor points

Green dots indicate sensor readings above threshold (detecting “safe” area)
Red dots indicate lower values (likely off-road / obstacle / unsafe)

### Target markers

Targets show their sequence number (1,2,3…)
Only the current target is “active” (pulsing)

### Stats panel

* **Epsilon(UI)**: shown for reference, not driving exploration in TD3
* **Score**: cumulative episode score

---

## 10) Troubleshooting

### “Agent was better earlier, now worse”

This can still happen due to:

* replay distribution issues
* reward shaping making it too easy to “survive” instead of reaching targets
* too many targets too close to unsafe zones

Try:

* placing targets on clear “road-like” regions
* spacing targets so the car has time to stabilize heading
* letting training run longer so actor learns stable steering

### App crashes with sensor deletion error

This code already includes the fix, but if you edit:

* ensure you never keep stale references to QGraphicsItems after `scene.clear()`

---

## 11) File structure

Single Python file containing:

* TD3 networks (`Actor`, `Critic`)
* Environment + training logic (`CarBrain`)
* UI widgets (`CarItem`, `SensorItem`, `TargetItem`, `RewardChart`)
* Main window app (`NeuralNavApp`)

---