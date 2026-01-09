"""
===============================================================================
NeuralNav - TD3 (Stable Exploration + Hysteresis) 
===============================================================================

✅ DOES NOT CHANGE your physics parameters:
  CAR_WIDTH, CAR_HEIGHT, SENSOR_DIST, SENSOR_ANGLE, SPEED, TURN_SPEED, SHARP_TURN

✅ DOES NOT CHANGE your provided RL hyperparameters:
  BATCH_SIZE, GAMMA, LR, TAU, MAX_CONSECUTIVE_CRASHES

What’s implemented (to fix “gets worse when epsilon is low”):
1) TD3 exploration noise is kept alive during training (not tied to epsilon)
2) Continuous action -> discrete action mapping now uses HYSTERESIS to prevent
   flip-flopping near thresholds (the big cause of collapse when noise is low)
3) Priority replay is based on actual target progress (not “lucky” score alone)
4) Fix for PyQt crash: SensorItem deleted -> recreate items after scene.clear/reset
===============================================================================
"""

import sys
import os
import math
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- PYQT ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QFrame, QFileDialog, QTextEdit, QGridLayout
)
from PyQt5.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QPainter, QFont, QPainterPath
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF


# ==========================================
# THEME
# ==========================================
C_BG_DARK   = QColor("#2E3440")
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A")
C_ACCENT    = QColor("#88C0D0")
C_TEXT      = QColor("#ECEFF4")
C_SUCCESS   = QColor("#A3BE8C")
C_FAILURE   = QColor("#BF616A")
C_SENSOR_ON = QColor("#A3BE8C")
C_SENSOR_OFF= QColor("#BF616A")


# ==========================================
# PHYSICS PARAMETERS — DO NOT CHANGE
# ==========================================
CAR_WIDTH = 14
CAR_HEIGHT = 8
SENSOR_DIST = 18
SENSOR_ANGLE = 50
SPEED = 2.5
TURN_SPEED = 2.2
SHARP_TURN = 12


# ==========================================
# RL HYPERPARAMETERS — DO NOT CHANGE
# ==========================================
BATCH_SIZE = 128
GAMMA = 0.99
LR = 3e-4
TAU = 0.005
MAX_CONSECUTIVE_CRASHES = 2


# ==========================================
# (Not your hyperparams) TD3 algorithm knobs
# These are required by TD3 itself and are not altering your given hyperparams.
# ==========================================
TD3_POLICY_NOISE = 0.2
TD3_NOISE_CLIP   = 0.5
TD3_POLICY_FREQ  = 2

# Stable exploration (key fix): never collapses to ~0 during training
TD3_EXPL_NOISE_STD = 0.25
TD3_EXPL_NOISE_MIN = 0.15  # minimum exploration noise (kept alive)


TARGET_COLORS = [
    QColor(0, 255, 255),
    QColor(255, 100, 255),
    QColor(0, 255, 100),
    QColor(255, 150, 0),
    QColor(100, 150, 255),
    QColor(255, 50, 150),
    QColor(150, 255, 50),
    QColor(255, 255, 0),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# TD3 Networks (classic 400/300 style)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)  # scalar action in [-1, 1]

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + 1, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + 1, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# ==========================================
# CAR BRAIN (env + TD3)
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()

        # state: 7 sensors + angle_to_target + distance_to_target
        self.state_dim = 9

        # actor/critic
        self.actor = Actor(self.state_dim).to(DEVICE)
        self.actor_target = Actor(self.state_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)  # LR unchanged

        self.critic = Critic(self.state_dim).to(DEVICE)
        self.critic_target = Critic(self.state_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)  # LR unchanged

        # replay (kept lightweight)
        self.memory = deque(maxlen=10000)
        self.priority_memory = deque(maxlen=3000)
        self.current_episode_buffer = []
        self.episode_scores = deque(maxlen=100)

        # training state
        self.td3_total_it = 0
        self.steps = 0

        # epsilon kept only for UI display (TD3 exploration uses Gaussian noise)
        self.epsilon = 1.0

        self.consecutive_crashes = 0

        # world state
        self.start_pos = QPointF(100, 100)
        self.car_pos = QPointF(100, 100)
        self.car_angle = 0

        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        self.target_pos = QPointF(200, 200)

        self.alive = True
        self.score = 0.0
        self.sensor_coords = []
        self.prev_dist = None

        # key fix: hysteresis mapping state
        self.last_disc_action = 1  # start as "straight"

        # stable exploration (not tied to epsilon)
        self.expl_noise_std = TD3_EXPL_NOISE_STD

    def set_start_pos(self, p: QPointF):
        self.start_pos = p
        self.car_pos = p

    def add_target(self, p: QPointF):
        self.targets.append(QPointF(p.x(), p.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0

    def switch_to_next_target(self) -> bool:
        # A1 -> A2 -> A3 (no looping back)
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def reset(self):
        self.alive = True
        self.score = 0.0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)

        self.current_target_idx = 0
        self.targets_reached = 0
        if self.targets:
            self.target_pos = self.targets[0]

        self.last_disc_action = 1  # reset hysteresis memory
        s, dist = self.get_state()
        self.prev_dist = dist
        return s

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        angles = [-45, -30, -15, 0, 15, 30, 45]

        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos.x() + math.cos(rad) * SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))

            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)

        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx * dx + dy * dy)

        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)

        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360

        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0

        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    # ==========================================
    # TD3 action selection (continuous)
    # Stable exploration fix: keep noise >= TD3_EXPL_NOISE_MIN during training
    # ==========================================
    def select_action_cont(self, state_np: np.ndarray, training: bool = True) -> float:
        st = torch.FloatTensor(state_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a = self.actor(st).cpu().numpy().flatten()[0]  # [-1,1]

        if training:
            # keep noise alive even late in training
            sigma = max(self.expl_noise_std, TD3_EXPL_NOISE_MIN)
            a = a + np.random.normal(0, sigma)

        return float(np.clip(a, -1.0, 1.0))

    # ==========================================
    # HYSTERESIS mapping: continuous [-1,1] -> discrete {0..4}
    # Prevents flip-flopping near thresholds (major stability fix)
    # Discrete actions:
    # 0 left, 1 straight, 2 right, 3 sharp left, 4 sharp right
    # ==========================================
    def cont_to_discrete_hysteresis(self, a_cont: float) -> int:
        a = float(np.clip(a_cont, -1.0, 1.0))

        dead = 0.15
        sharp = 0.55
        margin = 0.10  # hysteresis band (not physics)

        last = self.last_disc_action

        # If we were in a SHARP action, require stronger evidence to leave it
        if last == 3:  # sharp left
            if a < -(sharp - margin):
                self.last_disc_action = 3
                return 3
        if last == 4:  # sharp right
            if a > (sharp - margin):
                self.last_disc_action = 4
                return 4

        # If we were in NORMAL left/right, keep it unless we cross back into deadzone+margin
        if last == 0:  # left
            if a < -(dead - margin):
                self.last_disc_action = 0
                return 0
        if last == 2:  # right
            if a > (dead - margin):
                self.last_disc_action = 2
                return 2

        # Otherwise decide fresh with slightly “stickier” boundaries
        if abs(a) < dead:
            self.last_disc_action = 1
            return 1

        if a <= -sharp:
            self.last_disc_action = 3
            return 3
        if a >= sharp:
            self.last_disc_action = 4
            return 4

        if a < 0:
            self.last_disc_action = 0
            return 0
        else:
            self.last_disc_action = 2
            return 2

    # ==========================================
    # ENV STEP (PHYSICS UNCHANGED)
    # ==========================================
    def step(self, action_disc: int):
        turn = 0.0
        if action_disc == 0:
            turn = -TURN_SPEED
        elif action_disc == 1:
            turn = 0.0
        elif action_disc == 2:
            turn = TURN_SPEED
        elif action_disc == 3:
            turn = -SHARP_TURN
        elif action_disc == 4:
            turn = SHARP_TURN

        self.car_angle += turn
        rad = math.radians(self.car_angle)

        new_x = self.car_pos.x() + math.cos(rad) * SPEED
        new_y = self.car_pos.y() + math.sin(rad) * SPEED
        self.car_pos = QPointF(new_x, new_y)

        next_state, dist = self.get_state()

        reward = -0.1
        done = False

        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())

        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        elif dist < 20:
            reward = 100
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            # keep your shaping style (unchanged intent)
            reward += (1.0 - next_state[4]) * 20
            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 10
            self.prev_dist = dist

        self.score += reward
        return next_state, reward, done

    # ==========================================
    # Replay handling
    # Store (s, a_cont, r, ns, done)
    # ==========================================
    def store_experience(self, exp):
        self.current_episode_buffer.append(exp)

    def finalize_episode(self):
        """
        Priority replay fix:
        - Only prioritize if the episode actually made target progress AND ended alive.
          (reduces “lucky with noise” episodes polluting priority replay)
        """
        if not self.current_episode_buffer:
            return

        reached_progress = (self.targets_reached > 0)  # reached A1->A2 progress etc.
        if reached_progress and self.alive:
            for e in self.current_episode_buffer:
                self.priority_memory.append(e)
        else:
            for e in self.current_episode_buffer:
                self.memory.append(e)

        self.episode_scores.append(self.score)

        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0

        self.current_episode_buffer = []

    def _sample_batch(self):
        total = len(self.memory) + len(self.priority_memory)
        if total < BATCH_SIZE:
            return None

        # keep your priority-bias idea
        success_rate = len(self.priority_memory) / max(total, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)

        p_n = int(BATCH_SIZE * priority_ratio)
        r_n = BATCH_SIZE - p_n

        batch = []
        if len(self.priority_memory) >= p_n:
            batch.extend(random.sample(self.priority_memory, p_n))
        else:
            batch.extend(list(self.priority_memory))
            r_n += p_n - len(self.priority_memory)

        if len(self.memory) >= r_n:
            batch.extend(random.sample(self.memory, r_n))
        else:
            batch.extend(list(self.memory))

        if len(batch) < max(32, BATCH_SIZE // 2):
            return None

        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(DEVICE)
        a = torch.FloatTensor(np.array(a, dtype=np.float32)).unsqueeze(1).to(DEVICE)
        r = torch.FloatTensor(np.array(r, dtype=np.float32)).unsqueeze(1).to(DEVICE)
        ns = torch.FloatTensor(np.array(ns)).to(DEVICE)
        d = torch.FloatTensor(np.array(d, dtype=np.float32)).unsqueeze(1).to(DEVICE)
        return s, a, r, ns, d

    @staticmethod
    def _soft_update(net, target_net, tau):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def optimize(self) -> float:
        batch = self._sample_batch()
        if batch is None:
            return 0.0

        self.td3_total_it += 1

        state, action, reward, next_state, done = batch

        with torch.no_grad():
            noise = (torch.randn_like(action) * TD3_POLICY_NOISE).clamp(-TD3_NOISE_CLIP, TD3_NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * GAMMA * target_Q  # GAMMA unchanged

        # Critic update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed actor update
        if self.td3_total_it % TD3_POLICY_FREQ == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft updates (TAU unchanged)
            self._soft_update(self.actor, self.actor_target, TAU)
            self._soft_update(self.critic, self.critic_target, TAU)

        # epsilon decay kept only for UI display (doesn't control TD3)
        if self.epsilon > 0.001:
            self.epsilon *= 0.9995

        return float(critic_loss.item())


# ==========================================
# UI Widgets
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, C_PANEL)

        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val:
            max_val += 1

        step_x = w / (self.max_points - 1)
        points = []
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)

        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawPath(path)


class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0.0
        self.pulse_speed = 0.3
        self.is_detecting = True

    def set_detecting(self, detecting: bool):
        self.is_detecting = detecting
        self.update()

    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)

    def paint(self, painter, option, widget):
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0.0

        color = C_SENSOR_ON if self.is_detecting else C_SENSOR_OFF
        outer_alpha = int((150 if self.is_detecting else 200) * (1 - self.pulse))

        painter.setRenderHint(QPainter.Antialiasing)

        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)

        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)


class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH / 2, -CAR_HEIGHT / 2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.white)
        painter.drawRect(int(CAR_WIDTH / 2) - 2, -3, 2, 6)


class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0.0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10:
                    self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0:
                    self.growing = True

            r = 10 + self.pulse
            painter.setPen(Qt.NoPen)
            outer = QColor(self.color)
            outer.setAlpha(100)
            painter.setBrush(QBrush(outer))
            painter.drawEllipse(QPointF(0, 0), r, r)

            painter.setBrush(QBrush(self.color))
            painter.setPen(QPen(Qt.white, 2))
            painter.drawEllipse(QPointF(0, 0), 8, 8)
        else:
            dim = QColor(self.color)
            dim.setAlpha(120)
            painter.setPen(QPen(Qt.white, 1))
            painter.setBrush(QBrush(dim))
            painter.drawEllipse(QPointF(0, 0), 6, 6)

        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignCenter, str(self.number))


# ==========================================
# APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav (TD3) - Stable Exploration + Hysteresis")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)

        lbl_title = QLabel("CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)

        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)\nRight-click to finish targets")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)

        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)

        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)

        vbox.addSpacing(15)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)

        self.val_eps = QLabel("1.00")
        self.val_eps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Epsilon(UI):"), 0, 0)
        sf_layout.addWidget(self.val_eps, 0, 1)

        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Score:"), 1, 0)
        sf_layout.addWidget(self.val_rew, 1, 1)

        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Simulation state
        self.setup_state = 0
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)

        # Graphics items
        self.car_item = None
        self.sensor_items = []
        self.target_items = []

        # Load map
        self.setup_map("city_map.png")

    def log(self, msg: str):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.white)
        p.setPen(Qt.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def _rebuild_overlay_items(self):
        """
        IMPORTANT FIX: scene.clear() deletes QGraphicsItems.
        We must recreate CarItem + SensorItem objects after clear/reset.
        """
        self.car_item = CarItem()
        self.scene.addItem(self.car_item)

        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)

        self.map_img = QImage(path).convertToFormat(QImage.Format_RGB32)

        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self._rebuild_overlay_items()

        self.brain = CarBrain(self.map_img)
        self.target_items = []
        self.log("Map Loaded.")

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f:
            self.full_reset()
            self.setup_map(f)

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())

        if self.setup_state == 0:
            self.brain.set_start_pos(pt)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map -> TARGET(S)\nRight-click when done")
            return

        if self.setup_state == 1:
            if event.button() == Qt.LeftButton:
                self.brain.add_target(pt)

                idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[idx % len(TARGET_COLORS)]
                is_active = (idx == 0)
                num = len(self.brain.targets)

                ti = TargetItem(color, is_active, num)
                ti.setPos(pt)
                self.scene.addItem(ti)
                self.target_items.append(ti)

                self.lbl_status.setText(f"Targets: {num}\nRight-click to finish setup")
                self.log(f"Target #{num} added at ({pt.x():.0f}, {pt.y():.0f})")
                return

            if event.button() == Qt.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(
                        f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;"
                    )
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0

        # Remove target visuals
        for t in self.target_items:
            if t.scene() == self.scene:
                self.scene.removeItem(t)
        self.target_items = []

        # Reset brain targets
        self.brain.targets = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        self.brain.alive = True
        self.brain.score = 0.0

        # Recreate sensors (avoid deleted-object issues)
        for s in self.sensor_items:
            if s.scene() == self.scene:
                self.scene.removeItem(s)
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

        # Move car offscreen until placed
        self.car_item.setPos(QPointF(-9999, -9999))

        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)\nRight-click to finish targets")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")

        self.log("--- RESET ---")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        if self.setup_state != 2:
            return

        state, _ = self.brain.get_state()
        prev_target_idx = self.brain.current_target_idx

        # TD3 continuous action + stable exploration
        a_cont = self.brain.select_action_cont(state, training=True)

        # Hysteresis mapping to discrete (physics unchanged)
        action_disc = self.brain.cont_to_discrete_hysteresis(a_cont)

        next_state, reward, done = self.brain.step(action_disc)

        # Store TD3 experience
        self.brain.store_experience((state, a_cont, reward, next_state, float(done)))

        # TD3 update
        loss = self.brain.optimize()

        if self.brain.current_target_idx != prev_target_idx:
            self.log(f"<font color='#88C0D0'>🎯 Reached target {prev_target_idx + 1} → now going to {self.brain.current_target_idx + 1}</font>")

        self.brain.steps += 1

        if done:
            self.brain.finalize_episode()

            if self.brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='#BF616A'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting to origin...</b></font>")
                self.brain.consecutive_crashes = 0
                self.brain.reset()
            else:
                if not self.brain.alive:
                    self.log(f"<font color='#BF616A'>CRASH | Score: {self.brain.score:.0f} | Loss: {loss:.3f}</font>")
                    # restart episode but keep target progression reset
                    self.brain.reset()
                else:
                    # completed all targets
                    self.log(f"<font color='#A3BE8C'>DONE | Score: {self.brain.score:.0f} | Loss: {loss:.3f}</font>")
                    self.brain.reset()

            self.chart.update_chart(self.brain.score)

        self.update_visuals()
        self.val_eps.setText(f"{self.brain.epsilon:.3f}")
        self.val_rew.setText(f"{self.brain.score:.0f}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)

        for i, t in enumerate(self.target_items):
            t.set_active(i == self.brain.current_target_idx)

        self.scene.update()

        # Update sensors
        st, _ = self.brain.get_state()
        for i, coord in enumerate(self.brain.sensor_coords):
            if i < len(self.sensor_items):
                self.sensor_items[i].setPos(coord)
                self.sensor_items[i].set_detecting(st[i] > 0.5)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec_())
