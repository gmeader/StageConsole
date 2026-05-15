import sys
import json
import asyncio
import time
import os
import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QRect
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QSlider, QLabel, QProgressBar,
    QHeaderView, QFileDialog, QSpinBox, QDoubleSpinBox,
    QMessageBox, QScrollArea, QDialog, QLineEdit, QCheckBox, 
    QRadioButton, QInputDialog
)
from qasync import QEventLoop

# --- CONSTANTS ---
CONFIG_FILE = "config.json"
DMX_CHANNELS = 512
REFRESH_RATE = 0.022 

# --- DATA MODELS ---
class SceneTableModel(QAbstractTableModel):
    def __init__(self, scenes):
        super().__init__()
        self.scenes = scenes
        self.headers = ["#", "Name", "In", "Out", "Delay", "Follow"]

    def rowCount(self, parent=QModelIndex()): return len(self.scenes)
    def columnCount(self, parent=QModelIndex()): return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole: return None
        scene = self.scenes[index.row()]
        col = index.column()
        if col == 0: return f"{scene['num']:.1f}"
        if col == 1: return scene['name']
        if col == 2: return f"{scene['fin']}s"
        if col == 3: return f"{scene['fout']}s"
        if col == 4: return f"{scene['delay']}s"
        if col == 5: return "Yes" if scene['follow'] else "No"
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return None

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()

class DmxOutputModel(QAbstractTableModel):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def rowCount(self, parent=None): return 32
    def columnCount(self, parent=None): return 16

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        channel = index.row() * 16 + index.column()
        val = self.engine.current[channel]
        if role == Qt.DisplayRole: return str(int(val))
        if role == Qt.BackgroundRole:
            return QColor(0, int(val * 0.6), 0) if val > 0 else None
        if role == Qt.TextAlignmentRole: return Qt.AlignCenter
        return None

# --- DMX ENGINE ---
class DmxEngine:
    def __init__(self, controller):
        self.controller = controller
        self.current = np.zeros(DMX_CHANNELS, dtype=np.float32)
        self.start_vals = np.zeros(DMX_CHANNELS, dtype=np.float32)
        self.target_vals = np.zeros(DMX_CHANNELS, dtype=np.float32)
        self.is_running = False
        self.start_time = 0.0
        self.fade_in_time = 2.0
        self.fade_out_time = 2.0
        self.delay = 0.0

    def trigger_scene(self, scene_dict):
        self.start_vals = np.copy(self.current)
        self.target_vals = np.array(scene_dict['dmx'], dtype=np.float32)
        self.fade_in_time = max(0.01, scene_dict['fin'])
        self.fade_out_time = max(0.01, scene_dict['fout'])
        self.delay = scene_dict['delay']
        self.start_time = time.monotonic()
        self.is_running = True

    def calculate(self):
        if not self.is_running: return
        elapsed = time.monotonic() - self.start_time
        p_in = np.clip((elapsed - self.delay) / self.fade_in_time, 0, 1)
        p_out = np.clip((elapsed - self.delay) / self.fade_out_time, 0, 1)
        up_mask = self.target_vals > self.start_vals
        prog_array = np.where(up_mask, p_in, p_out)
        self.current = self.start_vals + (self.target_vals - self.start_vals) * prog_array
        
        if p_in >= 1.0 and p_out >= 1.0: 
            self.is_running = False
            self.controller.advance_next_scene()

    def get_remaining(self):
        if not self.is_running: return 0.0, 0.0
        elapsed = time.monotonic() - self.start_time
        return max(0.0, (self.fade_in_time + self.delay) - elapsed), max(0.0, (self.fade_out_time + self.delay) - elapsed)

# --- DIALOGS ---
class SceneDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scene Properties")
        layout = QVBoxLayout(self)
        self.num = QDoubleSpinBox(); self.num.setRange(1.0, 999.9); self.num.setDecimals(1)
        self.name = QLineEdit()
        self.fin = QDoubleSpinBox(); self.fin.setValue(2.0); self.fout = QDoubleSpinBox(); self.fout.setValue(2.0)
        self.delay = QDoubleSpinBox(); self.follow = QCheckBox("Auto-Follow")
        for w in [QLabel("Scene #"), self.num, QLabel("Name"), self.name, QLabel("Fade In"), self.fin, QLabel("Fade Out"), self.fout, QLabel("Delay"), self.delay, self.follow]:
            layout.addWidget(w)
        btns = QHBoxLayout(); ok = QPushButton("Save"); ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        btns.addWidget(ok); btns.addWidget(cancel); layout.addLayout(btns)

    def get_data(self):
        return {"num": self.num.value(), "name": self.name.text(), "fin": self.fin.value(), "fout": self.fout.value(), "delay": self.delay.value(), "follow": self.follow.isChecked()}

class ClickLabel(QLabel):
    def __init__(self, text, index, controller):
        super().__init__(text)
        self.index = index
        self.controller = controller
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("color: #A80; font-size: 11px;")

    def mousePressEvent(self, event):
        text, ok = QInputDialog.getText(self, "Channel Label", f"Label for Channel {self.index + 1}:", text=self.text())
        if ok:
            self.setText(text)
            self.controller.channel_labels[self.index] = text

# --- WINDOWS ---
class PlaybackWindow(QMainWindow):
    def __init__(self, engine, controller):
        super().__init__(); self.setWindowTitle("Playback"); self.resize(300, 250); self.engine, self.controller = engine, controller
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        self.info = QLabel("NEXT: --"); self.info.setStyleSheet("font-weight: bold; font-size: 14px; color: #31A6F3;")
        self.lbl_in = QLabel("In: 0.0s"); self.bar_in = QProgressBar()
        self.lbl_out = QLabel("Out: 0.0s"); self.bar_out = QProgressBar()
        self.go_btn = QPushButton("GO"); self.go_btn.setFixedHeight(60); self.go_btn.setStyleSheet("background: green; color: white; font-size: 18px;")
        self.go_btn.clicked.connect(self.controller.trigger_playback)
        for w in [self.info, self.lbl_in, self.bar_in, self.lbl_out, self.bar_out, self.go_btn]: layout.addWidget(w)

    def refresh_ui(self):
        if self.controller.scenes and self.controller.next_scene_idx < len(self.controller.scenes):
            nxt = self.controller.scenes[self.controller.next_scene_idx]
            self.info.setText(f"NEXT: {nxt['num']} ({nxt['name']})")
        else:
            self.info.setText("NEXT: End of Show")
        rem_in, rem_out = self.engine.get_remaining()
        self.lbl_in.setText(f"In: {rem_in:.1f}s"); self.lbl_out.setText(f"Out: {rem_out:.1f}s")
        if self.engine.is_running:
            el = time.monotonic() - self.engine.start_time
            self.bar_in.setValue(int(np.clip((el-self.engine.delay)/self.engine.fade_in_time, 0, 1)*100))
            self.bar_out.setValue(int(np.clip((el-self.engine.delay)/self.engine.fade_out_time, 0, 1)*100))
        else:
            self.bar_in.setValue(0); self.bar_out.setValue(0)

class FaderWindow(QMainWindow):
    def __init__(self, engine, controller):
        super().__init__(); self.setWindowTitle("Faders"); self.resize(950, 420); self.engine, self.controller = engine, controller
        mb = self.menuBar(); mb.setNativeMenuBar(False)
        
        file_m = mb.addMenu("&File")
        file_m.addAction("Open Show", self.controller.open_show_dialog)
        file_m.addAction("Save Show", self.controller.save_show_dialog)
        file_m.addSeparator()
        file_m.addAction("Quit", QApplication.instance().quit)
        
        config_m = mb.addMenu("&Config")
        config_m.addAction("Save Windows", self.controller.save_config) # New Save Action
        port_m = config_m.addMenu("Output Port")
        for p in ["COM1", "COM3", "/dev/ttyUSB0", "Enttec Pro"]:
            port_m.addAction(p, lambda pt=p: self.controller.set_port(pt))
        
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        
        mode_layout = QHBoxLayout()
        self.live_radio = QRadioButton("LIVE"); self.blind_radio = QRadioButton("BLIND")
        self.live_radio.setChecked(True); self.live_radio.toggled.connect(self.mode_changed)
        mode_layout.addWidget(self.live_radio); mode_layout.addWidget(self.blind_radio); mode_layout.addStretch()
        layout.addLayout(mode_layout)

        self.scroll = QScrollArea(); self.fw = QWidget(); self.fh = QHBoxLayout(self.fw)
        self.faders = []
        for i in range(16):
            v = QVBoxLayout(); lbl_val = QLabel("0"); sl = QSlider(Qt.Vertical); sl.setRange(0, 255)
            lbl_num = QLabel(str(i+1))
            lbl_user = ClickLabel(f"CH {i+1}", i, self.controller)
            sl.valueChanged.connect(lambda v, l=lbl_val, idx=i: self.update_ch(idx, v, l))
            v.addWidget(lbl_val, alignment=Qt.AlignCenter); v.addWidget(sl)
            v.addWidget(lbl_num, alignment=Qt.AlignCenter); v.addWidget(lbl_user, alignment=Qt.AlignCenter)
            self.fh.addLayout(v); self.faders.append((sl, lbl_val, lbl_num, lbl_user))
        
        self.scroll.setWidget(self.fw); self.scroll.setWidgetResizable(True)
        self.pg = QSpinBox(); self.pg.setRange(1, 32); self.pg.valueChanged.connect(self.sync_faders)
        layout.addWidget(self.scroll); layout.addWidget(QLabel("Page:")); layout.addWidget(self.pg)

    def mode_changed(self):
        self.controller.mode = "LIVE" if self.live_radio.isChecked() else "BLIND"
        if self.controller.mode == "LIVE": self.controller.fader_buffer = np.copy(self.engine.current)
        self.sync_faders()

    def update_ch(self, idx, val, lbl):
        lbl.setText(str(val)); g_idx = (self.pg.value()-1)*16 + idx
        self.controller.fader_buffer[g_idx] = val
        if self.controller.mode == "LIVE": self.engine.current[g_idx] = val

    def sync_faders(self):
        for i, (sl, lbl_val, lbl_num, lbl_user) in enumerate(self.faders):
            g_idx = (self.pg.value()-1)*16 + i
            lbl_num.setText(str(g_idx+1))
            lbl_user.setText(self.controller.channel_labels[g_idx])
            lbl_user.index = g_idx
            val = self.controller.fader_buffer[g_idx]
            sl.blockSignals(True); sl.setValue(int(val)); sl.blockSignals(False)
            lbl_val.setText(str(int(val)))

# --- CONTROLLER ---
class DmxController:
    def __init__(self):
        self.engine = None 
        self.scenes = [] 
        self.channel_labels = [f"CH {i+1}" for i in range(DMX_CHANNELS)]
        self.scene_model = SceneTableModel(self.scenes)
        self.next_scene_idx = 0
        self.mode = "LIVE"
        self.fader_buffer = np.zeros(DMX_CHANNELS, dtype=np.float32)
        self.current_port = "None"
        
        # Window References
        self.win_f = None
        self.win_p = None
        self.win_s = None
        self.win_o = None

    def set_port(self, port):
        self.current_port = port
        self.save_config()

    def save_config(self):
        """Saves Port and Window Geometries to config.json"""
        config = {"port": self.current_port, "windows": {}}
        
        wins = {
            "faders": self.win_f,
            "playback": self.win_p,
            "scenes": self.win_s,
            "monitor": self.win_o
        }
        
        for name, win in wins.items():
            if win:
                geom = win.geometry()
                config["windows"][name] = [geom.x(), geom.y(), geom.width(), geom.height()]
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def load_config(self):
        """Loads Port and applies Window Geometries"""
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.current_port = config.get("port", "None")
                
                win_data = config.get("windows", {})
                wins = {
                    "faders": self.win_f,
                    "playback": self.win_p,
                    "scenes": self.win_s,
                    "monitor": self.win_o
                }
                
                for name, win in wins.items():
                    if name in win_data and win:
                        x, y, w, h = win_data[name]
                        win.setGeometry(QRect(x, y, w, h))
        except Exception as e:
            print(f"Error loading config: {e}")

    def add_scene(self):
        dlg = SceneDialog()
        if dlg.exec():
            d = dlg.get_data(); d["dmx"] = self.fader_buffer.tolist()
            self.scenes[:] = [s for s in self.scenes if s['num'] != d['num']]
            self.scenes.append(d); self.scenes.sort(key=lambda x: x['num'])
            self.scene_model.refresh()

    def load_scene_to_faders(self, index):
        if 0 <= index < len(self.scenes):
            self.fader_buffer = np.array(self.scenes[index]['dmx'], dtype=np.float32)
            if self.mode == "LIVE": self.engine.current = np.copy(self.fader_buffer)
            if self.win_f: self.win_f.sync_faders()

    def set_next_by_index(self, index):
        if 0 <= index < len(self.scenes): self.next_scene_idx = index

    def advance_next_scene(self):
        if self.next_scene_idx < len(self.scenes) - 1: self.next_scene_idx += 1

    def trigger_playback(self):
        if self.scenes and self.next_scene_idx < len(self.scenes):
            self.engine.trigger_scene(self.scenes[self.next_scene_idx])

    def save_show_dialog(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save Show", "", "Show Files (*.json)")
        if path:
            show_data = {"scenes": self.scenes, "labels": self.channel_labels}
            with open(path, 'w') as f: json.dump(show_data, f)

    def open_show_dialog(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open Show", "", "Show Files (*.json)")
        if path:
            with open(path, 'r') as f: 
                data = json.load(f)
                self.scenes.clear(); self.scenes.extend(data.get("scenes", []))
                default_labels = [f"CH {i+1}" for i in range(DMX_CHANNELS)]
                labels = data.get("labels", default_labels)
                self.channel_labels[:len(labels)] = labels
            self.next_scene_idx = 0; self.scene_model.refresh()
            if self.win_f: self.win_f.sync_faders()

# --- MAIN ---
async def main_loop(engine, out_model, win_p, win_f, ctrl):
    while True:
        engine.calculate()
        if ctrl.mode == "LIVE":
            ctrl.fader_buffer = np.copy(engine.current)
            win_f.sync_faders()
        out_model.layoutChanged.emit()
        win_p.refresh_ui()
        await asyncio.sleep(REFRESH_RATE)

if __name__ == "__main__":
    app = QApplication(sys.argv); loop = QEventLoop(app); asyncio.set_event_loop(loop)
    
    ctrl = DmxController()
    eng = DmxEngine(ctrl)
    ctrl.engine = eng
    out_model = DmxOutputModel(eng)

    # Initialize Faders
    win_f = FaderWindow(eng, ctrl); ctrl.win_f = win_f
    
    # Initialize Playback
    win_p = PlaybackWindow(eng, ctrl); ctrl.win_p = win_p
    
    # Initialize Scene Manager
    win_s = QMainWindow(); win_s.setWindowTitle("Scene Manager"); win_s.resize(500, 400)
    sw = QWidget(); sl = QVBoxLayout(sw); tv = QTableView(); tv.setModel(ctrl.scene_model)
    tv.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tv.clicked.connect(lambda idx: ctrl.set_next_by_index(idx.row()))
    tv.doubleClicked.connect(lambda idx: ctrl.load_scene_to_faders(idx.row()))
    btn = QPushButton("Save Look"); btn.clicked.connect(ctrl.add_scene)
    sl.addWidget(tv); sl.addWidget(btn); win_s.setCentralWidget(sw); ctrl.win_s = win_s

    # Initialize Monitor
    win_o = QMainWindow(); win_o.setWindowTitle("Monitor"); win_o.resize(600, 400)
    otv = QTableView(); otv.setModel(out_model); win_o.setCentralWidget(otv)
    otv.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); ctrl.win_o = win_o

    # Apply Config (Windows positions + Port)
    ctrl.load_config()

    win_f.show(); win_p.show(); win_s.show(); win_o.show()
    
    loop.create_task(main_loop(eng, out_model, win_p, win_f, ctrl))
    loop.run_forever()