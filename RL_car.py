"""
===============================================================================
ASSIGNMENT: AUTONOMOUS CAR NAVIGATION
===============================================================================

Updates:
1) Added a DNN with 1 extra FC layer.
2) Enforced exactly 3 targets: A1 -> A2 -> A3.
3) The car targets them sequentially; after reaching A3, the episode ends.
4) Fixed: RuntimeError "wrapped C/C++ object of type SensorItem has been deleted"
   by recreating sensor items after scene.clear().
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
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QGraphicsScene,
    QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
    QTextEdit, QGridLayout
)
from PyQt5.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QPainter,
    QPainterPath, QFont
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
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
# PHYSICS PARAMETERS
# ==========================================
CAR_WIDTH = 14
CAR_HEIGHT = 8

SENSOR_DIST = 18
SPEED = 2.7
TURN_SPEED = 2.2
SHARP_TURN = 12

# ==========================================
# RL HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 128
GAMMA = 0.99
LR = 3e-4
TAU = 0.005
MAX_CONSECUTIVE_CRASHES = 2

# Targets
TARGET_COLORS = [
    QColor(0, 255, 255),      # A1
    QColor(255, 100, 255),    # A2
    QColor(0, 255, 100),      # A3
]
NUM_TARGETS_REQUIRED = 3
TARGET_LABELS = ["A1", "A2", "A3"]


# ==========================================
# 2. NEURAL NETWORK (DNN)  ✅ extra FC layer
# ==========================================
class DrivingDQN(nn.Module):
    """
    Added 1 extra FC layer:
    input -> 128 -> 256 -> 256 -> 128 -> 64 -> out
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),   # ✅ NEW extra FC layer
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. ENV / BRAIN
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()

        # State: 7 sensors + angle_to_target + distance_to_target = 9
        self.input_dim = 9
        self.n_actions = 5  # left, straight, right, sharp_left, sharp_right

        self.policy_net = DrivingDQN(self.input_dim, self.n_actions)
        self.target_net = DrivingDQN(self.input_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self.memory = deque(maxlen=10000)
        self.priority_memory = deque(maxlen=3000)
        self.current_episode_buffer = []

        self.steps = 0
        self.epsilon = 1.0
        self.consecutive_crashes = 0

        self.start_pos = QPointF(100, 100)
        self.car_pos = QPointF(100, 100)
        self.car_angle = 0

        # Targets A1->A2->A3
        self.targets = []
        self.current_target_idx = 0
        self.target_pos = QPointF(200, 200)

        self.alive = True
        self.score = 0

        self.sensor_coords = []
        self.prev_dist = None

    def set_start_pos(self, point: QPointF):
        self.start_pos = QPointF(point.x(), point.y())
        self.car_pos = QPointF(point.x(), point.y())

    def reset(self):
        """
        Reset position + restart the sequence from A1.
        (Since after reaching A3, episode ends.)
        """
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)

        self.current_target_idx = 0
        if len(self.targets) == NUM_TARGETS_REQUIRED:
            self.target_pos = self.targets[0]

        state, dist = self.get_state()
        self.prev_dist = dist
        return state

    def add_target(self, point: QPointF) -> bool:
        if len(self.targets) >= NUM_TARGETS_REQUIRED:
            return False
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.current_target_idx = 0
            self.target_pos = self.targets[0]
        return True

    def switch_to_next_target(self) -> bool:
        """
        A1 -> A2 -> A3 (no wrap).
        Returns True if switched, False if already at last target.
        """
        if len(self.targets) == 0:
            return False
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            return True
        return False  # already at A3

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

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

    def step(self, action):
        # action -> turn amount
        if action == 0:
            turn = -TURN_SPEED
        elif action == 1:
            turn = 0
        elif action == 2:
            turn = TURN_SPEED
        elif action == 3:
            turn = -SHARP_TURN
        else:
            turn = SHARP_TURN

        self.car_angle += turn
        rad = math.radians(self.car_angle)

        self.car_pos = QPointF(
            self.car_pos.x() + math.cos(rad) * SPEED,
            self.car_pos.y() + math.sin(rad) * SPEED
        )

        next_state, dist = self.get_state()
        reward = -0.1
        done = False

        # Crash check
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False

        # Reached target
        elif dist < 20:
            reward = 100

            # If A1/A2 -> move to next, continue episode
            if self.current_target_idx < NUM_TARGETS_REQUIRED - 1:
                self.switch_to_next_target()
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                # ✅ reached A3 -> end episode
                done = True

        else:
            # shaping
            reward += (1.0 - next_state[4]) * 20
            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 10
            self.prev_dist = dist

        self.score += reward
        return next_state, reward, done

    def store_experience(self, exp):
        self.current_episode_buffer.append(exp)

    def finalize_episode(self):
        if not self.current_episode_buffer:
            return

        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0

        if self.score > 0:
            self.priority_memory.extend(self.current_episode_buffer)
        else:
            self.memory.extend(self.current_episode_buffer)

        self.current_episode_buffer = []

    def optimize(self):
        total_size = len(self.memory) + len(self.priority_memory)
        if total_size < BATCH_SIZE:
            return 0.0

        success_rate = len(self.priority_memory) / max(total_size, 1)
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

        if len(batch) < BATCH_SIZE // 2:
            return 0.0

        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns))
        d = torch.FloatTensor(d).unsqueeze(1)

        q = self.policy_net(s).gather(1, a)
        next_q = self.target_net(ns).max(1)[0].detach().unsqueeze(1)
        target = r + GAMMA * next_q * (1 - d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > 0.001:
            self.epsilon *= 0.9995

        return float(loss.item())


# ==========================================
# 4. QGraphics Items
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

        min_val, max_val = min(self.scores), max(self.scores)
        if max_val == min_val:
            max_val += 1

        step_x = w / (self.max_points - 1)
        pts = []
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            pts.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)

        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawPath(path)


class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
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
            self.pulse = 0

        if self.is_detecting:
            color = C_SENSOR_ON
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF
            outer_alpha = int(200 * (1 - self.pulse))

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
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)

        painter.setBrush(Qt.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)


class TargetItem(QGraphicsItem):
    def __init__(self, color: QColor, label: str, is_active: bool):
        super().__init__()
        self.setZValue(50)
        self.color = QColor(color)
        self.label = label
        self.is_active = is_active
        self.pulse = 0
        self.growing = True

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

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
            outer = QColor(self.color)
            outer.setAlpha(100)
            painter.setPen(Qt.NoPen)
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
        painter.drawText(QRectF(-12, -12, 24, 24), Qt.AlignCenter, self.label)


# ==========================================
# 5. APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav: Assignment")
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

        # Left panel
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)

        lbl_title = QLabel("CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)

        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGETS A1,A2,A3\n3. Right-click to finish")
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

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf = QGridLayout(stats_frame)
        sf.setContentsMargins(10, 10, 10, 10)

        self.val_eps = QLabel("1.00")
        self.val_eps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf.addWidget(QLabel("Epsilon:"), 0, 0)
        sf.addWidget(self.val_eps, 0, 1)

        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf.addWidget(QLabel("Score:"), 1, 0)
        sf.addWidget(self.val_rew, 1, 1)

        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # Right panel
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)

        # Items / state
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []

        self.setup_state = 0
        self.current_map_path = "city_map.png"

        self.setup_map(self.current_map_path)

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

    def create_sensor_items(self):
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)

        self.current_map_path = path
        self.map_img = QImage(path).convertToFormat(QImage.Format_RGB32)

        # IMPORTANT: clears & deletes items
        self.scene.clear()

        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)

        # ✅ recreate sensors after clear
        self.create_sensor_items()

        # reset placement state
        self.target_items = []
        self.setup_state = 0
        self.btn_run.setEnabled(False)
        self.btn_run.setChecked(False)
        self.btn_run.setText("▶ START (Space)")

        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGETS A1,A2,A3\n3. Right-click to finish")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        self.log("Map Loaded.")

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg *.jpeg)")
        if f:
            self.sim_timer.stop()
            self.setup_map(f)

    def full_reset(self):
        self.sim_timer.stop()
        self.chart.scores = []
        self.chart.update()
        self.log("--- RESET ---")
        self.setup_map(self.current_map_path)

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

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())

        if self.setup_state == 0:
            self.brain.set_start_pos(pt)
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click 3 targets: A1, A2, A3 (Left-click)\nRight-click when done")
            self.log(f"Car start set at ({pt.x():.0f}, {pt.y():.0f})")
            return

        if self.setup_state == 1:
            if event.button() == Qt.LeftButton:
                if len(self.brain.targets) >= NUM_TARGETS_REQUIRED:
                    self.log(f"<font color='{C_FAILURE.name()}'><b>⚠️ Only 3 targets allowed: A1, A2, A3</b></font>")
                    return

                if not self.brain.add_target(pt):
                    return

                idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[idx]
                label = TARGET_LABELS[idx]
                is_active = (idx == 0)

                t = TargetItem(color=color, label=label, is_active=is_active)
                t.setPos(pt)
                self.scene.addItem(t)
                self.target_items.append(t)

                remaining = NUM_TARGETS_REQUIRED - len(self.brain.targets)
                self.lbl_status.setText(
                    f"Targets placed: {len(self.brain.targets)}/3\n"
                    f"Remaining: {remaining}\nRight-click when all 3 are placed"
                )
                self.log(f"Target {label} added at ({pt.x():.0f}, {pt.y():.0f})")
                return

            if event.button() == Qt.RightButton:
                if len(self.brain.targets) != NUM_TARGETS_REQUIRED:
                    self.log(f"<font color='{C_FAILURE.name()}'><b>⚠️ Place exactly 3 targets (A1,A2,A3) before starting.</b></font>")
                    return

                self.setup_state = 2
                self.btn_run.setEnabled(True)
                self.lbl_status.setText("READY. Sequence: A1 → A2 → A3 (stops at A3)\nPress SPACE.")
                self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")

                self.brain.reset()
                self.update_visuals()
                return

    def game_loop(self):
        if self.setup_state != 2:
            return

        state, _ = self.brain.get_state()
        prev_target_idx = self.brain.current_target_idx

        # epsilon-greedy
        if random.random() < self.brain.epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                q = self.brain.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = int(q.argmax().item())

        next_s, rew, done = self.brain.step(action)

        self.brain.store_experience((state, action, rew, next_s, float(done)))
        self.brain.optimize()

        # log target change
        if self.brain.current_target_idx != prev_target_idx:
            frm = TARGET_LABELS[prev_target_idx]
            to = TARGET_LABELS[self.brain.current_target_idx]
            self.log(f"<font color='{C_ACCENT.name()}'><b>🎯 {frm} reached! Now targeting {to}</b></font>")

        # soft update target net
        for t_param, p_param in zip(self.brain.target_net.parameters(), self.brain.policy_net.parameters()):
            t_param.data.copy_(TAU * p_param.data + (1.0 - TAU) * t_param.data)

        self.brain.steps += 1

        if done:
            self.brain.finalize_episode()

            if not self.brain.alive:
                self.log(f"<font color='{C_FAILURE.name()}'><b>💥 CRASH! Score={self.brain.score:.0f}</b></font>")
            else:
                self.log(f"<font color='{C_SUCCESS.name()}'><b>✅ A3 REACHED! Episode complete. Score={self.brain.score:.0f}</b></font>")

            if self.brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='{C_FAILURE.name()}'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} crashes in a row! Resetting...</b></font>")
                self.brain.consecutive_crashes = 0

            self.chart.update_chart(self.brain.score)
            self.brain.reset()

        self.update_visuals()
        self.val_eps.setText(f"{self.brain.epsilon:.3f}")
        self.val_rew.setText(f"{self.brain.score:.0f}")

    def update_visuals(self):
        # ensure sensors exist
        if not self.sensor_items:
            return

        # ensure coords exist
        if len(self.brain.sensor_coords) != len(self.sensor_items):
            _ = self.brain.get_state()

        # car
        if self.car_item.scene() == self.scene:
            self.car_item.setPos(self.brain.car_pos)
            self.car_item.setRotation(self.brain.car_angle)

        # targets highlight
        for i, t in enumerate(self.target_items):
            t.set_active(i == self.brain.current_target_idx)

        # sensors
        state, _ = self.brain.get_state()
        for i, coord in enumerate(self.brain.sensor_coords):
            if i < len(self.sensor_items):
                self.sensor_items[i].setPos(coord)
                self.sensor_items[i].set_detecting(state[i] > 0.5)

        self.scene.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec_())
