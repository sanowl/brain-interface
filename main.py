from __future__ import annotations
import os, time, logging, asyncio, threading, json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
import secrets

try:
    import mne
    import pyedflib
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE or pyedflib not available. Install with: pip install mne pyedflib")

def set_global_seed(seed: int = 42):
    secrets.SystemRandom().seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("BCI")

class CognitiveState(str, Enum):
    FOCUSED = "focused"
    STUCK = "stuck"
    DESIGNING = "designing"
    DEBUGGING = "debugging"
    TESTING = "testing"
    SEARCHING = "searching"

ALL_STATES: List[str] = [s.value for s in CognitiveState]

@dataclass(frozen=True)
class StateMetadata:
    description: str
    typical_duration_sec: float
    intervention_priority: int

STATE_METADATA: Dict[CognitiveState, StateMetadata] = {
    CognitiveState.FOCUSED: StateMetadata("Deep focus", 15.0, 1),
    CognitiveState.STUCK: StateMetadata("Problem‑solving block", 3.0, 5),
    CognitiveState.DESIGNING: StateMetadata("Architecting solution", 10.0, 2),
    CognitiveState.DEBUGGING: StateMetadata("Bug hunt", 8.0, 3),
    CognitiveState.TESTING: StateMetadata("Writing / running tests", 5.0, 2),
    CognitiveState.SEARCHING: StateMetadata("Information lookup", 2.0, 3),
}

class COGBCILoader:
    """Loader for COG-BCI Database (cognitive workload tasks)"""

    def __init__(self, data_path: str, sampling_rate: int = 250):
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.task_mapping = {
            'nback_0': CognitiveState.FOCUSED,
            'nback_1': CognitiveState.DEBUGGING,
            'nback_2': CognitiveState.STUCK,
            'matb_easy': CognitiveState.FOCUSED,
            'matb_hard': CognitiveState.TESTING,
            'pvt': CognitiveState.SEARCHING,
            'flanker': CognitiveState.DESIGNING,
            'rest': CognitiveState.FOCUSED
        }

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load COG-BCI dataset"""
        if not MNE_AVAILABLE:
            raise ImportError("MNE required for real EEG loading. Install with: pip install mne")

        log.info("Loading COG-BCI dataset...")
        eeg_files = list(self.data_path.glob("**/*.edf")) + list(self.data_path.glob("**/*.bdf"))

        if not eeg_files:
            log.error(f"No EEG files found in {self.data_path}")
            log.info("Expected file structure:")
            log.info("  data_path/")
            log.info("    ├── sub-01/")
            log.info("    │   ├── ses-01/")
            log.info("    │   │   └── eeg/")
            log.info("    │   │       └── sub-01_ses-01_task-nback_eeg.edf")
            log.info("    └── ...")
            raise FileNotFoundError("COG-BCI dataset not found")

        all_data, all_labels, all_users = [], [], []

        for i, eeg_file in enumerate(eeg_files[:10]):
            try:
                raw = mne.io.read_raw_edf(str(eeg_file), preload=True, verbose=False)

                task = self._extract_task_from_filename(eeg_file.name)
                if task not in self.task_mapping:
                    log.warning(f"Unknown task '{task}' in file {eeg_file.name}")
                    continue

                processed_data = self._preprocess_raw(raw)

                epochs = self._create_epochs(processed_data, epoch_length=2.0)

                for epoch in epochs:
                    all_data.append(epoch)
                    all_labels.append(ALL_STATES.index(self.task_mapping[task].value))
                    all_users.append(i)

                log.info(f"Loaded {len(epochs)} epochs from {eeg_file.name}")

            except Exception as e:
                log.warning(f"Failed to load {eeg_file}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid EEG data could be loaded")

        X = np.stack(all_data)
        y = np.array(all_labels, dtype="int64")
        users = np.array(all_users, dtype="int64")

        log.info(f"Loaded COG-BCI: {X.shape[0]} epochs, {len(np.unique(users))} subjects")
        return X, y, users

    def _extract_task_from_filename(self, filename: str) -> str:
        """Extract task name from COG-BCI filename"""
        filename = filename.lower()
        if 'nback' in filename:
            if '0back' in filename or 'nback_0' in filename:
                return 'nback_0'
            elif '1back' in filename or 'nback_1' in filename:
                return 'nback_1'
            elif '2back' in filename or 'nback_2' in filename:
                return 'nback_2'
        elif 'matb' in filename:
            if 'easy' in filename or 'low' in filename:
                return 'matb_easy'
            elif 'hard' in filename or 'high' in filename:
                return 'matb_hard'
        elif 'pvt' in filename:
            return 'pvt'
        elif 'flanker' in filename:
            return 'flanker'
        elif 'rest' in filename:
            return 'rest'
        return 'unknown'

    def _preprocess_raw(self, raw: 'mne.io.Raw') -> np.ndarray:
        """Preprocess real EEG data"""
        if raw.info['sfreq'] != self.sampling_rate:
            raw.resample(self.sampling_rate)

        raw.pick_types(eeg=True, exclude='bads')

        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)
        raw.notch_filter(freqs=60.0, verbose=False)

        data, _ = raw[:]

        data = data.T
        target_channels = 64

        if data.shape[1] < target_channels:
            padding = np.zeros((data.shape[0], target_channels - data.shape[1]))
            data = np.concatenate([data, padding], axis=1)
        elif data.shape[1] > target_channels:
            data = data[:, :target_channels]

        return data.astype('float32')

    def _create_epochs(self, data: np.ndarray, epoch_length: float = 2.0) -> List[np.ndarray]:
        """Create fixed-length epochs from continuous data"""
        epoch_samples = int(epoch_length * self.sampling_rate)
        epochs = []

        for start in range(0, data.shape[0] - epoch_samples, epoch_samples // 2):
            epoch = data[start:start + epoch_samples]
            if epoch.shape[0] == epoch_samples:
                epochs.append(epoch)

        return epochs

class PhysioNetLoader:
    """Backup loader for PhysioNet Motor Imagery dataset"""

    def __init__(self, data_path: str, sampling_rate: int = 160):
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.task_mapping = {
            'rest': CognitiveState.FOCUSED,
            'motor': CognitiveState.DEBUGGING,
        }

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load PhysioNet Motor Imagery dataset"""
        if not MNE_AVAILABLE:
            raise ImportError("MNE required for real EEG loading")

        log.info("Loading PhysioNet Motor Imagery dataset...")

        edf_files = list(self.data_path.glob("**/*.edf"))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found in {self.data_path}")

        all_data, all_labels, all_users = [], [], []

        for i, edf_file in enumerate(edf_files[:20]):
            try:
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

                processed_data = self._preprocess_physionet(raw)

                task = 'motor' if 'task' in edf_file.name else 'rest'

                epochs = self._create_epochs(processed_data, epoch_length=2.0)

                for epoch in epochs:
                    all_data.append(epoch)
                    all_labels.append(ALL_STATES.index(self.task_mapping[task].value))
                    all_users.append(i)

                log.info(f"Loaded {len(epochs)} epochs from {edf_file.name}")

            except Exception as e:
                log.warning(f"Failed to load {edf_file}: {e}")
                continue

        X = np.stack(all_data)
        y = np.array(all_labels, dtype="int64")
        users = np.array(all_users, dtype="int64")

        log.info(f"Loaded PhysioNet: {X.shape[0]} epochs, {len(np.unique(users))} subjects")
        return X, y, users

    def _preprocess_physionet(self, raw: 'mne.io.Raw') -> np.ndarray:
        """Preprocess PhysioNet data"""
        raw.pick_types(eeg=True, exclude='bads')
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)
        raw.notch_filter(freqs=60.0, verbose=False)

        data, _ = raw[:]
        data = data.T

        if data.shape[1] < 64:
            padding = np.zeros((data.shape[0], 64 - data.shape[1]))
            data = np.concatenate([data, padding], axis=1)
        elif data.shape[1] > 64:
            data = data[:, :64]

        return data.astype('float32')

    def _create_epochs(self, data: np.ndarray, epoch_length: float = 2.0) -> List[np.ndarray]:
        """Create epochs for PhysioNet data"""
        epoch_samples = int(epoch_length * self.sampling_rate)
        epochs = []

        for start in range(0, data.shape[0] - epoch_samples, epoch_samples):
            epoch = data[start:start + epoch_samples]
            if epoch.shape[0] == epoch_samples:
                epochs.append(epoch)

        return epochs

class EnhancedBCISimulator:
    """Fallback simulator if real datasets unavailable"""

    def __init__(self, n_channels: int = 64, sampling_rate: int = 250, window_sec: float = 2.0, global_seed: int | None = None):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.window_size = int(window_sec * sampling_rate)
        self._rng = np.random.default_rng(global_seed)

    def _baseline(self, user_id: int) -> Dict[str, Any]:
        rng = np.random.default_rng(user_id)
        return {
            "alpha": rng.uniform(8, 12),
            "beta": rng.uniform(13, 30),
            "gamma": rng.uniform(30, 50),
            "noise": rng.uniform(0.15, 0.35),
            "weights": rng.uniform(0.3, 1.8, self.n_channels),
            "artefact_rate": rng.uniform(0.1, 0.3),
            "circadian_phase": rng.uniform(0, 2*np.pi),
        }

    def _add_wave(self, t: np.ndarray, freq: float, amp: float, w: np.ndarray) -> np.ndarray:
        wave = np.sin(2 * np.pi * freq * t)[:, None]
        return wave * amp * w[None, :]

    def _artefacts(self, data: np.ndarray, base: Dict, rng: np.random.Generator) -> np.ndarray:
        """More realistic artifacts"""
        rate = base["artefact_rate"]
        T, C = data.shape

        n_blinks = rng.poisson(rate * T / 50)
        for _ in range(min(n_blinks, 8)):
            if T < 50: continue
            s = rng.integers(0, T - 40)
            blink_profile = rng.uniform(100, 300) * np.exp(-np.arange(40)/8)
            for ch in range(min(8, C)):
                data[s:s+40, ch] += blink_profile
        muscle_samples = int(rate * 0.15 * T * C)
        if muscle_samples > 0:
            t_idx = rng.integers(0, T, muscle_samples)
            c_idx = rng.integers(0, C, muscle_samples)
            muscle_noise = rng.normal(0, 30, muscle_samples)
            data[t_idx, c_idx] += muscle_noise
        t = np.arange(T) / self.sampling_rate
        line_noise = 3.0 * np.sin(2*np.pi*60*t)
        data += line_noise[:, None]

        return data

    def simulate(self, state: str, base: Dict, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self._rng
        T, C = self.window_size, self.n_channels
        data = rng.standard_normal((T, C)) * base["noise"]

        t = np.arange(T)/self.sampling_rate
        w = base["weights"]
        if state == CognitiveState.FOCUSED.value:
            data += self._add_wave(t, base["alpha"], 0.4, w)
            data += self._add_wave(t, 6.0, 0.2, w)

        elif state == CognitiveState.STUCK.value:
            data[:, :min(8, C)] += self._add_wave(t, 6.0, 0.8, w[:min(8, C)])
            data += self._add_wave(t, base["alpha"], 0.2, w)

        elif state == CognitiveState.DESIGNING.value:
            data += self._add_wave(t, base["gamma"], rng.uniform(0.2, 0.4), w)
            bursts = rng.random(T) < 0.05
            data[bursts] *= 1.3

        elif state == CognitiveState.DEBUGGING.value:
            data += self._add_wave(t, base["beta"], 0.6, w)

        elif state == CognitiveState.TESTING.value:
            m = (np.sin(2*np.pi*base["alpha"]*t) + np.sin(2*np.pi*base["beta"]*t))
            data += 0.3*m[:,None]*w

        elif state == CognitiveState.SEARCHING.value:
            data += self._add_wave(t, 15.0, 0.4, w)
            n_bursts = rng.integers(1, 3)
            burst_positions = rng.integers(0, max(1, T-20), size=n_bursts)
            for p in burst_positions:
                end_pos = min(p+20, T)
                data[p:end_pos] *= 1.2

        data = self._artefacts(data, base, rng)
        data *= 0.7 + 0.3*np.cos(base["circadian_phase"])
        return data.astype("float32")

    def generate_dataset(self, n_users: int = 20, samples_per_user: int = 30, imbalance: Dict[str, float] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if imbalance is None:
            p = np.full(len(ALL_STATES), 1/len(ALL_STATES))
        else:
            p = np.array([imbalance.get(s,0) for s in ALL_STATES])
            p /= p.sum()

        X, y, u = [], [], []
        log.info(f"Generating realistic simulated data ({n_users} users, {samples_per_user} samples each)...")

        for user in range(n_users):
            base = self._baseline(user)
            rng = np.random.default_rng(self._rng.integers(0,2**32))
            states = rng.choice(ALL_STATES, samples_per_user, p=p)
            for s in states:
                X.append(self.simulate(s, base, rng))
                y.append(ALL_STATES.index(s))
                u.append(user)

        return np.stack(X), np.array(y), np.array(u)

class EEGDataManager:
    """Manages loading from real datasets or simulation fallback"""

    def __init__(self, prefer_real_data: bool = True):
        self.prefer_real_data = prefer_real_data

    def load_best_available_dataset(self, cogbci_path: Optional[str] = None, physionet_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Load the best available dataset"""

        if self.prefer_real_data and cogbci_path and MNE_AVAILABLE:
            try:
                if Path(cogbci_path).exists():
                    loader = COGBCILoader(cogbci_path)
                    X, y, users = loader.load_dataset()
                    return X, y, users, "COG-BCI"
                else:
                    log.warning(f"COG-BCI path not found: {cogbci_path}")
            except Exception as e:
                log.warning(f"Failed to load COG-BCI: {e}")

        if self.prefer_real_data and physionet_path and MNE_AVAILABLE:
            try:
                if Path(physionet_path).exists():
                    loader = PhysioNetLoader(physionet_path)
                    X, y, users = loader.load_dataset()
                    return X, y, users, "PhysioNet"
                else:
                    log.warning(f"PhysioNet path not found: {physionet_path}")
            except Exception as e:
                log.warning(f"Failed to load PhysioNet: {e}")

        log.info("Using enhanced simulation (more realistic than before)")
        simulator = EnhancedBCISimulator(global_seed=2025)
        X, y, users = simulator.generate_dataset(n_users=15, samples_per_user=25)
        return X, y, users, "Enhanced Simulation"

class AdvancedFeatureExtractor:
    def __init__(self, sr: int = 250):
        self.sr = sr
        self.scaler = StandardScaler()

    def _notch(self, batch: np.ndarray) -> np.ndarray:
        """More robust notch filtering"""
        fft = np.fft.fft(batch, axis=1)
        freqs = np.fft.fftfreq(batch.shape[1], 1/self.sr)
        fft_clean = fft.copy()

        for f0 in (60, 120, 180):
            mask = np.abs(freqs - f0) < 0.5
            fft_clean[:, mask] *= 0.1

        return np.real(np.fft.ifft(fft_clean, axis=1))

    def _spectral(self, batch: np.ndarray) -> np.ndarray:
        clean = self._notch(batch)
        fft = np.fft.rfft(clean, axis=1)
        power = np.abs(fft)**2 + 1e-12

        freqs = np.fft.rfftfreq(batch.shape[1], 1/self.sr)
        bands = [(1,4),(4,8),(8,12),(12,30),(30,50)]
        total = power.sum(axis=1, keepdims=True) + 1e-10
        feats = []

        for lo,hi in bands:
            m = (freqs>=lo)&(freqs<=hi)
            if m.any():
                bp = power[:, m, :].mean(axis=1)
                feats.append(bp)
                feats.append(bp/total.squeeze(1))
            else:
                zeros = np.zeros((batch.shape[0], batch.shape[2]))
                feats.extend([zeros, zeros])

        return np.concatenate(feats, axis=1)

    @staticmethod
    def _temporal(batch: np.ndarray) -> np.ndarray:
        eps = 1e-12
        mean = batch.mean(axis=1)
        std = batch.std(axis=1) + eps
        var = batch.var(axis=1)

        if batch.shape[1] > 1:
            activity = np.abs(np.diff(batch, axis=1)).mean(axis=1)
        else:
            activity = np.zeros_like(mean)

        if batch.shape[1] > 2:
            d1 = np.diff(batch, axis=1)
            d2 = np.diff(d1, axis=1)
            hj_mob = d1.std(axis=1) / (std + eps)
            hj_comp = d2.std(axis=1) / (d1.std(axis=1) + eps)
        else:
            hj_mob = np.ones_like(mean)
            hj_comp = np.ones_like(mean)

        centered = batch - mean[:, None, :]
        skew = (centered**3).mean(axis=1) / (std**3 + eps)
        kurt = (centered**4).mean(axis=1) / (std**4 + eps)

        return np.concatenate([mean, std, var, activity, hj_mob, hj_comp, skew, kurt], axis=1)

    def transform(self, batch: np.ndarray) -> np.ndarray:
        if len(batch.shape) != 3:
            raise ValueError(f"Expected 3D input (N,T,C), got shape {batch.shape}")

        spec = self._spectral(batch)
        temp = self._temporal(batch)
        return np.concatenate([spec, temp], axis=1)

class UniversalCognitiveModel:
    def __init__(self, extractor: AdvancedFeatureExtractor, n_states: int = len(ALL_STATES)):
        self.extractor = extractor
        self.n_states = n_states
        dummy = np.zeros((1, 500, 64), dtype=np.float32)
        self.input_dim = extractor.transform(dummy).shape[1]
        self.model = self._build()
        self._scaler_fitted = False

    def _build(self) -> keras.Model:
        m = keras.Sequential([
            keras.layers.Input((self.input_dim,)),
            keras.layers.Dense(512,'relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256,'relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(128,'relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self.n_states,'softmax')
        ])
        m.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        return m

    @staticmethod
    def _split(X,y,u,ratio=0.2):
        users=np.unique(u); rng=np.random.default_rng(0); rng.shuffle(users)
        n_t=int(max(1,len(users)*ratio)); test=users[:n_t]
        tr=~np.isin(u,test); te=~tr
        return X[tr],y[tr],X[te],y[te]

    def fit(self, X_raw,y,user_ids,epochs=15,batch=32):
        feats = self.extractor.transform(X_raw)
        Xtr,ytr,Xte,yte = self._split(feats,y,user_ids)

        scaler=self.extractor.scaler
        Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
        self._scaler_fitted=True

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]

        self.model.fit(Xtr,ytr,
                       validation_data=(Xte,yte),
                       epochs=epochs,batch_size=batch,
                       callbacks=callbacks,
                       verbose=2)

    def predict(self, sample: np.ndarray) -> Dict[str, Any]:
        if not self._scaler_fitted:
            raise RuntimeError("Model not trained yet.")

        if len(sample.shape) != 2:
            raise ValueError(f"Expected 2D input (T,C), got shape {sample.shape}")

        try:
            feats = self.extractor.transform(sample[None,...])
            feats = self.extractor.scaler.transform(feats)
            p = self.model.predict(feats, verbose=0)[0]
            idx = int(np.argmax(p))
            return {
                "state": ALL_STATES[idx],
                "conf": float(p[idx]),
                "probs": {s: float(pp) for s, pp in zip(ALL_STATES, p)}
            }
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return {"state": "unknown", "conf": 0.0, "probs": {}}

class PersonalisedModel:
    def __init__(self, base: UniversalCognitiveModel):
        self.base = base
        self.personal = None

    def adapt(self, X,y,epochs=8,lr=1e-4,freeze_to=-2):
        try:
            self.personal = keras.models.clone_model(self.base.model)
            self.personal.set_weights(self.base.model.get_weights())
            for layer in self.personal.layers[:freeze_to]:
                layer.trainable=False
            self.personal.compile(optimizer=keras.optimizers.Adam(lr),
                                  loss="sparse_categorical_crossentropy",
                                  metrics=["accuracy"])
            feats=self.base.extractor.transform(X)
            feats=self.base.extractor.scaler.transform(feats)
            self.personal.fit(feats,y,epochs=epochs,batch_size=min(8,len(feats)),verbose=0)
            log.info(f"Personalization completed with {len(feats)} samples")
        except Exception as e:
            log.error(f"Adaptation failed: {e}")
            self.personal = None

    def predict(self, sample: np.ndarray):
        try:
            model = self.personal or self.base.model
            feats = self.base.extractor.transform(sample[None,...])
            feats = self.base.extractor.scaler.transform(feats)
            p = model.predict(feats,verbose=0)[0]
            idx = int(np.argmax(p))
            return {
                "state": ALL_STATES[idx],
                "conf": float(p[idx]),
                "model_type": "personalized" if self.personal else "universal"
            }
        except Exception as e:
            log.error(f"Personalized prediction error: {e}")
            return {"state": "unknown", "conf": 0.0, "model_type": "error"}

class CodingAssistanceEngine:
    def __init__(self):
        self.hist: List[Tuple[str,float,datetime]] = []

    def assist(self, state:str, conf:float)->List[str]:
        if conf<0.5:
            return ["⚠️ Low confidence, building confidence..."]

        self.hist.append((state,conf,datetime.now()))
        if state==CognitiveState.STUCK.value:
            return ["Try rubber‑duck debugging 🦆",
                    "Add debug prints 🔍",
                    "Take a 5‑minute break ☕"]
        if state==CognitiveState.DEBUGGING.value:
            return ["Use binary‑search debugging 🪓",
                    "Create minimal reproduction 📦"]
        if state==CognitiveState.DESIGNING.value:
            return ["Sketch UML diagram 📐",
                    "Apply SOLID principles 🧩"]
        if state==CognitiveState.TESTING.value:
            return ["Write edge‑case tests ✅",
                    "Consider parameterized tests 📊"]
        if state==CognitiveState.SEARCHING.value:
            return ["Google with specific terms 🔎",
                    "Check official docs 📚"]
        return ["Great focus! 🚀"]

class SignalBuffer:
    def __init__(self, max_samples: int, n_channels: int):
        self.buf = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        self.n_channels = n_channels

    def add(self, s: np.ndarray):
        if s.shape != (self.n_channels,):
            raise ValueError(f"Expected shape ({self.n_channels},), got {s.shape}")
        with self.lock:
            self.buf.append(s.copy())

    def ready(self, win: int) -> bool:
        return len(self.buf) >= win

    def window(self, win: int) -> np.ndarray:
        with self.lock:
            if len(self.buf) < win:
                raise ValueError(f"Not enough samples: {len(self.buf)} < {win}")
            return np.array(list(self.buf)[-win:])

class RealTimeBCI:
    def __init__(self, model: PersonalisedModel, win: int = 500, interval: float = 1.0):
        self.model = model
        self.win = win
        self.interval = interval
        self.buf = SignalBuffer(2000, 64)
        self.engine = CodingAssistanceEngine()
        self.running = False

    async def run(self):
        self.running = True
        log.info("Real-time processing started...")

        while self.running:
            if self.buf.ready(self.win):
                try:
                    sample = self.buf.window(self.win)
                    pred = self.model.predict(sample)
                    tips = self.engine.assist(pred["state"], pred["conf"])

                    model_type = pred.get("model_type", "unknown")
                    log.info(f"🧠 {pred['state']:<9} conf={pred['conf']:.2f} [{model_type}] | " +
                            " | ".join(tips))

                except Exception as e:
                    log.error(f"Processing error: {e}")

            await asyncio.sleep(self.interval)

    def stop(self):
        self.running = False


def main():
    set_global_seed(2025)
    log.info("=== BCI System with Real EEG Dataset Loading ===")
    COGBCI_PATH = "/path/to/cogbci/dataset"
    PHYSIONET_PATH = "/path/to/physionet/eegmmidb"

    data_manager = EEGDataManager(prefer_real_data=True)
    X, y, users, dataset_name = data_manager.load_best_available_dataset(
        cogbci_path=COGBCI_PATH,
        physionet_path=PHYSIONET_PATH
    )

    log.info(f"Using dataset: {dataset_name}")
    log.info(f"Data shape: {X.shape}, Users: {len(np.unique(users))}")

    log.info("Training universal model...")
    extractor = AdvancedFeatureExtractor(sr=250)
    uni = UniversalCognitiveModel(extractor)
    uni.fit(X, y, users, epochs=10 if dataset_name == "Enhanced Simulation" else 15)

    val_acc = max(uni.model.history.history.get('val_accuracy', [0]))
    log.info(f"Final validation accuracy: {val_acc:.1%}")

    if dataset_name != "Enhanced Simulation":
        log.info("🎉 Using REAL EEG data! Expect 60-75% accuracy (realistic)")
    else:
        log.info("📊 Using enhanced simulation (more realistic than before)")

    log.info("Creating personalization data...")
    if dataset_name == "Enhanced Simulation":
        sim = EnhancedBCISimulator(global_seed=2025)
        new_base = sim._baseline(999)
        calib, lab = [], []
        for idx, state in enumerate(ALL_STATES):
            for _ in range(6):
                calib.append(sim.simulate(state, new_base))
                lab.append(idx)
    else:
        calib, lab = [], []
        user_indices = np.where(users == users[0])[0][:36]
        for i, idx in enumerate(user_indices):
            calib.append(X[idx])
            lab.append(y[idx])

    calib = np.stack(calib)
    lab = np.array(lab)

    perso = PersonalisedModel(uni)
    perso.adapt(calib, lab, epochs=6)

    rt = RealTimeBCI(perso, win=X.shape[1], interval=0.8)

    async def producer():
        log.info("Starting realistic data stream...")

        if dataset_name == "Enhanced Simulation":
            sim = EnhancedBCISimulator(global_seed=2025)
            new_base = sim._baseline(999)
            rng = np.random.default_rng(7)

            for window_idx in range(6):
                current_state = rng.choice(ALL_STATES)
                log.info(f"📡 Simulating realistic {current_state} state...")

                window_data = sim.simulate(current_state, new_base, rng)

                for t in range(window_data.shape[0]):
                    sample = window_data[t]
                    rt.buf.add(sample)
                    await asyncio.sleep(1/250)
        else:
            log.info("📡 Streaming real EEG data...")
            for i in range(min(6, len(X))):
                real_state = ALL_STATES[y[i]]
                log.info(f"📡 Streaming real {real_state} data...")

                for t in range(X[i].shape[0]):
                    sample = X[i][t]
                    rt.buf.add(sample)
                    await asyncio.sleep(1/250)

    async def orchestrate():
        consumer = asyncio.create_task(rt.run())
        await producer()
        rt.stop()
        await consumer

    log.info("Starting real-time demo...")
    try:
        asyncio.run(orchestrate())
    except KeyboardInterrupt:
        log.info("Demo interrupted by user")
    except Exception as e:
        log.error(f"Demo error: {e}")

    log.info("✅ Demo completed!")
    log.info("🔧 To use real data:")
    log.info("  1. Download COG-BCI: https://www.nature.com/articles/s41597-022-01898-y")
    log.info("  2. Or PhysioNet: https://archive.physionet.org/pn4/eegmmidb/")
    log.info("  3. pip install mne pyedflib")
    log.info("  4. Update COGBCI_PATH and PHYSIONET_PATH in main()")

if __name__ == "__main__":
    main()
