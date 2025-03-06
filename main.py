import pygame
import numpy as np
import pyaudio
from scipy.fft import fft
from collections import deque
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import abc
import math


@dataclass
class AudioConfig:
    """音频配置类"""
    CHUNK: int = 2048
    FORMAT: int = pyaudio.paFloat32
    CHANNELS: int = 1
    RATE: int = 44100


@dataclass
class VisualizerConfig:
    """可视化配置类"""
    # 窗口大小预设
    WINDOW_SIZES = [
        (800, 600),  # 默认大小
        (1024, 768),  # 中等大小
        (1280, 720),  # 720p
        (1920, 1080),  # 1080p
    ]

    WIDTH: int = WINDOW_SIZES[0][0]
    HEIGHT: int = WINDOW_SIZES[0][1]
    FPS: int = 60
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    HIGH_FREQ_COLOR: Tuple[int, int, int] = (255, 50, 50)
    LOW_FREQ_COLOR: Tuple[int, int, int] = (50, 255, 50)
    UI_COLOR: Tuple[int, int, int] = (200, 200, 200)
    FONT_SIZE: int = 16
    FONT_PATH: str = "C:\\Windows\\Fonts\\msyh.ttc"

    # 音符灯带配置
    LIGHT_WIDTH: int = 40
    LIGHT_HEIGHT: int = 20
    LIGHT_SPACING: int = 2
    LIGHT_FADE_SPEED: float = 0.1  # 灯光淡出速度

    # 地面投影配置
    GROUND_Y: int = 550  # 地面高度
    BASE_RADIUS: int = 60  # 基础半径
    MAX_RADIUS: int = 120  # 最大半径
    GLOW_LAYERS: int = 5  # 发光层数

    # 激光粒子配置
    PARTICLE_LIFETIME: float = 1.0  # 粒子生命周期（秒）
    PARTICLE_SPEED: float = 200.0  # 粒子速度
    PARTICLE_SIZE: int = 3  # 粒子大小
    MAX_PARTICLES: int = 100  # 最大粒子数
    PARTICLE_ALPHA: int = 128  # 粒子透明度


class LaserParticle:
    """激光粒子类"""

    def __init__(self, x: float, y: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = VisualizerConfig.PARTICLE_LIFETIME
        self.max_lifetime = VisualizerConfig.PARTICLE_LIFETIME
        self.size = VisualizerConfig.PARTICLE_SIZE
        self.alpha = VisualizerConfig.PARTICLE_ALPHA
        # 随机方向
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(VisualizerConfig.PARTICLE_SPEED * 0.5, VisualizerConfig.PARTICLE_SPEED)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)

    def update(self, dt: float) -> bool:
        """更新粒子状态，返回是否存活"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface):
        """绘制粒子"""
        alpha = int(self.alpha * (self.lifetime / self.max_lifetime))
        color = (*self.color, alpha)
        pos = (int(self.x), int(self.y))
        pygame.draw.circle(surface, color, pos, self.size)


class ParticleSystem:
    """粒子系统类"""

    def __init__(self):
        self.particles: List[LaserParticle] = []
        self.surface = pygame.Surface((VisualizerConfig.WIDTH, VisualizerConfig.HEIGHT), pygame.SRCALPHA)
        # 添加能量系统
        self.energy = 0.0  # 当前能量
        self.max_energy = 1.0  # 最大能量
        self.energy_decay = 0.95  # 能量衰减率
        self.energy_threshold = 0.2  # 降低能量阈值
        self.particle_energy_cost = 0.05  # 降低粒子能量消耗
        self.last_energy_update = time.time()
        # 添加相对峰值检测
        self.last_peak_time = 0
        self.peak_cooldown = 0.05  # 降低峰值冷却时间
        self.peak_threshold = 0.3  # 降低峰值检测阈值
        self.last_values = []  # 存储最近的值用于相对峰值检测
        self.history_size = 5  # 历史值数量

    def update(self, dt: float):
        """更新所有粒子和能量系统"""
        # 更新现有粒子
        self.particles = [p for p in self.particles if p.update(dt)]

        # 更新能量系统
        current_time = time.time()
        if current_time - self.last_energy_update >= 0.016:  # 约60fps
            # 能量衰减
            self.energy *= self.energy_decay
            self.last_energy_update = current_time

    def add_particles(self, x: float, y: float, color: Tuple[int, int, int], energy: float):
        """添加新粒子"""
        current_time = time.time()

        # 更新历史值
        self.last_values.append(energy)
        if len(self.last_values) > self.history_size:
            self.last_values.pop(0)

        # 检查是否在峰值冷却时间内
        if current_time - self.last_peak_time < self.peak_cooldown:
            return

        # 相对峰值检测
        if len(self.last_values) >= 3:
            # 检查当前值是否比前后值都大
            if (self.last_values[-1] > self.last_values[-2] and
                    self.last_values[-1] > self.last_values[-3] and
                    self.last_values[-1] > self.peak_threshold):

                # 更新峰值时间
                self.last_peak_time = current_time

                # 更新系统能量
                self.energy = min(self.max_energy, self.energy + energy * 0.5)

                # 检查是否有足够能量生成粒子
                if self.energy >= self.energy_threshold:
                    # 根据系统能量和峰值强度决定添加的粒子数量
                    num_particles = int(self.energy * energy * 20)  # 增加粒子数量

                    # 计算粒子扩散范围
                    spread_range = int(30 * energy)  # 增加扩散范围

                    for _ in range(num_particles):
                        if len(self.particles) >= VisualizerConfig.MAX_PARTICLES:
                            break
                        # 消耗能量生成粒子
                        if self.energy >= self.particle_energy_cost:
                            # 在峰值位置周围随机生成粒子
                            spread_x = x + np.random.uniform(-spread_range, spread_range)
                            spread_y = y + np.random.uniform(-spread_range, spread_range)
                            self.particles.append(LaserParticle(spread_x, spread_y, color))
                            self.energy -= self.particle_energy_cost

    def draw(self, surface: pygame.Surface):
        """绘制所有粒子"""
        self.surface.fill((0, 0, 0, 0))  # 清除上一帧

        # 根据系统能量调整粒子效果
        energy_factor = self.energy / self.max_energy

        for particle in self.particles:
            # 根据系统能量调整粒子大小和透明度
            particle.size = int(VisualizerConfig.PARTICLE_SIZE * (1 + energy_factor))
            particle.alpha = int(VisualizerConfig.PARTICLE_ALPHA * (0.5 + energy_factor * 0.5))
            particle.draw(self.surface)

        surface.blit(self.surface, (0, 0))

    def resize(self, width: int, height: int):
        """调整表面大小"""
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)


class Block:
    """区块基类"""

    def __init__(self, data: np.ndarray, timestamp: float, previous_hash: int):
        self.data = data
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self._calculate_hash()

    def _calculate_hash(self) -> int:
        """计算区块哈希值"""
        return hash(f"{self.timestamp}{self.data.tobytes()}{self.previous_hash}")


class AudioBlock(Block):
    """音频数据区块"""

    def get_energy(self, freq_mask: np.ndarray) -> float:
        """计算特定频率范围的能量"""
        masked_data = self.data[freq_mask]
        return np.max(masked_data) if len(masked_data) > 0 else 0


class Blockchain:
    """区块链基类"""

    def __init__(self, max_blocks: int = 10):
        self.chain = deque(maxlen=max_blocks)

    def add_block(self, data: np.ndarray):
        """添加新区块"""
        previous_hash = self.chain[-1].hash if self.chain else 0
        new_block = AudioBlock(data, time.time(), previous_hash)
        self.chain.append(new_block)


class AudioProcessor:
    """音频处理器"""

    def __init__(self, config: AudioConfig):
        """初始化音频处理器

        Args:
            config: 音频配置对象
        """
        print("\n=== 初始化音频处理器 ===")
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.device_index = None
        self.last_error_time = 0
        self.error_count = 0
        self.max_errors = 3
        self.error_cooldown = 1.0
        self.last_device_index = None
        print("正在搜索可用音频设备...")
        self.available_devices = self._get_available_devices()
        print(f"找到 {len(self.available_devices)} 个可用设备")

        # 初始化音频流
        print("正在初始化音频流...")
        self._initialize_audio_stream()
        print("音频流初始化完成")

        self.blockchain = Blockchain()
        # 添加平滑缓冲
        self.smooth_buffer = {'high': 0.0, 'low': 0.0}
        self.energy_weights = {'high': 0.5, 'low': 0.5}  # 初始权重均等
        self.weight_smooth_factor = 0.1  # 权重平滑因子
        self.current_note = None  # 当前检测到的音符
        self.pitch_confidence = 0.0  # 音高检测的置信度

    def _initialize_audio_stream(self):
        """初始化音频流"""
        try:
            print("\n开始初始化音频流...")
            # 选择最佳输入设备
            self.device_index = self._get_best_input_device()
            print(f"选择的设备索引: {self.device_index}")

            # 如果设备索引与上次相同，尝试使用其他设备
            if self.device_index == self.last_device_index:
                print("设备索引与上次相同，尝试使用其他设备...")
                self.device_index = self._get_alternative_device()
                print(f"切换到新设备索引: {self.device_index}")

            # 验证音频参数
            print("验证音频参数...")
            if not self.audio.is_format_supported(
                    rate=self.config.RATE,
                    input_channels=self.config.CHANNELS,
                    input_format=self.config.FORMAT,
                    input_device=self.device_index
            ):
                raise RuntimeError("音频设备不支持当前配置")

            print("关闭旧的音频流...")
            # 关闭旧的音频流
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"关闭旧音频流时出错: {e}")

            print("打开新的音频流...")
            # 打开新的音频流
            self.stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.config.CHUNK,
                stream_callback=None
            )

            if not self.stream.is_active():
                raise RuntimeError("无法激活音频流")

            # 记录当前设备索引
            self.last_device_index = self.device_index
            print(f"音频流初始化成功，使用设备索引 {self.device_index}")

            # 重置错误计数
            self.error_count = 0
            self.last_error_time = 0

        except Exception as e:
            print(f"\n初始化音频流时出错: {e}")
            if self.stream:
                try:
                    self.stream.close()
                except:
                    pass
            raise RuntimeError(f"无法打开音频流: {e}")

    def _get_alternative_device(self) -> int:
        """获取替代的音频输入设备"""
        try:
            # 遍历所有设备
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    # 跳过当前设备和不可用的设备
                    if (i != self.last_device_index and
                            device_info['maxInputChannels'] > 0 and
                            device_info['index'] != self.last_device_index):
                        return i
                except:
                    continue
        except:
            pass
        # 如果没有找到替代设备，返回默认输入设备
        try:
            return self.audio.get_default_input_device_info()['index']
        except:
            return 0  # 返回第一个设备作为最后的尝试

    def _handle_audio_error(self):
        """处理音频错误，尝试重新初始化音频流"""
        current_time = time.time()

        # 检查是否在冷却时间内
        if current_time - self.last_error_time < self.error_cooldown:
            return False

        self.last_error_time = current_time
        self.error_count += 1

        # 如果错误次数超过阈值，尝试重新初始化
        if self.error_count >= self.max_errors:
            try:
                # 尝试使用不同的设备
                self.device_index = self._get_alternative_device()
                self._initialize_audio_stream()
                return True
            except Exception as e:
                print(f"重新初始化音频流失败: {e}")
                # 如果重新初始化失败，尝试重新创建 PyAudio 实例
                try:
                    if self.audio:
                        self.audio.terminate()
                    self.audio = pyaudio.PyAudio()
                    self._initialize_audio_stream()
                    return True
                except Exception as e2:
                    print(f"重新创建 PyAudio 实例失败: {e2}")
                    return False

        return False

    def _get_best_input_device(self) -> int:
        """选择最佳输入设备"""
        try:
            print("\n正在搜索最佳输入设备...")

            # 打印所有设备信息
            print("\n可用设备列表:")
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    print(f"设备 {i}: {device_info['name']}")
                    print(f"  最大输入通道: {device_info['maxInputChannels']}")
                    print(f"  最大输出通道: {device_info['maxOutputChannels']}")
                    print(f"  默认采样率: {device_info['defaultSampleRate']}")
                except Exception as e:
                    print(f"获取设备 {i} 信息时出错: {e}")

            # 尝试获取默认输入设备
            try:
                default_input = self.audio.get_default_input_device_info()
                print(f"\n默认输入设备: {default_input['name']}")
            except Exception as e:
                print(f"获取默认输入设备失败: {e}")

            # 首先尝试使用立体声混音设备
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    if (device_info['maxInputChannels'] > 0 and
                            any(keyword in device_info['name'].lower()
                                for keyword in ['立体声混音', 'stereo mix', 'wave', 'what u hear'])):
                        print(f"\n选择系统音频捕获设备: {device_info['name']}")
                        return i
                except:
                    continue

            # 然后尝试使用默认输入设备
            try:
                default_input = self.audio.get_default_input_device_info()
                if default_input and default_input['maxInputChannels'] > 0:
                    print(f"\n使用默认输入设备: {default_input['name']}")
                    return int(default_input['index'])
            except:
                pass

            # 最后尝试使用第一个可用的输入设备
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        print(f"\n使用可用输入设备: {device_info['name']}")
                        return i
                except:
                    continue

            print("\n警告: 未找到可用的音频输入设备，将使用默认设备")
            return 0

        except Exception as e:
            print(f"\n选择音频设备时出错: {e}")
            return 0

    def _detect_pitch(self, data_np: np.ndarray) -> Tuple[float, float]:
        """使用自相关函数检测音高

        Returns:
            Tuple[float, float]: (频率, 置信度)
        """
        # 归一化信号
        data_np = data_np - np.mean(data_np)
        if np.max(np.abs(data_np)) < 1e-6:
            return 0.0, 0.0

        # 计算自相关函数
        corr = np.correlate(data_np, data_np, mode='full')
        corr = corr[len(corr) // 2:]

        # 寻找峰值
        min_period = int(self.config.RATE / 2000)  # 最高频率 2000Hz
        max_period = int(self.config.RATE / 50)  # 最低频率 50Hz
        peaks = []

        for i in range(min_period, min(len(corr), max_period)):
            if corr[i] > corr[i - 1] and corr[i] > corr[i + 1]:
                peaks.append((i, corr[i]))

        if not peaks:
            return 0.0, 0.0

        # 选择最高的峰值
        period, peak_val = max(peaks, key=lambda x: x[1])
        frequency = self.config.RATE / period
        confidence = peak_val / corr[0]  # 归一化置信度

        return frequency, confidence

    def _frequency_to_note(self, frequency: float) -> str:
        """将频率转换为音符名称"""
        if frequency < 20:  # 低于可听范围
            return "Unknown"

        # 计算最接近的音符
        a4_freq = 440.0
        steps = round(12 * np.log2(frequency / a4_freq))
        note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        note_idx = (steps + 9) % 12  # A4 对应的索引偏移
        octave = 4 + (steps + 9) // 12

        return f"{note_names[note_idx]}{octave}"

    def _calculate_dynamic_weights(self, fft_data: np.ndarray, freqs: np.ndarray, freq_split: float) -> None:
        """计算高低频的动态权重

        基于频率分布特征动态调整权重
        """
        try:
            # 分离高低频
            low_mask = freqs < freq_split
            high_mask = freqs >= freq_split

            # 计算高低频段的能量总和
            low_energy = np.sum(fft_data[low_mask])
            high_energy = np.sum(fft_data[high_mask])
            total_energy = low_energy + high_energy

            if total_energy > 1e-6:  # 避免除零
                # 基础权重：能量比例
                base_high_weight = high_energy / total_energy
                base_low_weight = low_energy / total_energy

                # 计算频率分布的集中度（使用标准差）
                if np.any(low_mask):
                    low_std = np.std(fft_data[low_mask])
                else:
                    low_std = 0

                if np.any(high_mask):
                    high_std = np.std(fft_data[high_mask])
                else:
                    high_std = 0

                # 根据分布集中度调整权重
                concentration_factor = 0.5
                high_weight = base_high_weight * (1 + concentration_factor * high_std)
                low_weight = base_low_weight * (1 + concentration_factor * low_std)

                # 归一化权重
                total_weight = high_weight + low_weight
                if total_weight > 0:
                    high_weight /= total_weight
                    low_weight /= total_weight

                    # 应用平滑
                    self.energy_weights['high'] = (self.weight_smooth_factor * high_weight +
                                                   (1 - self.weight_smooth_factor) * self.energy_weights['high'])
                    self.energy_weights['low'] = (self.weight_smooth_factor * low_weight +
                                                  (1 - self.weight_smooth_factor) * self.energy_weights['low'])
        except Exception as e:
            print(f"计算动态权重时出错: {e}")

    def process_audio(self, freq_split: float, difficulty: float, history_weight: float) -> Tuple[float, float]:
        """处理音频数据并返回高低频能量"""
        try:
            # 检查音频流是否有效
            if not self.stream or not self.stream.is_active():
                if self._handle_audio_error():
                    return self.smooth_buffer['low'], self.smooth_buffer['high']
                return self.smooth_buffer['low'], self.smooth_buffer['high']

            # 获取音频数据
            try:
                data = self.stream.read(self.config.CHUNK, exception_on_overflow=False)
            except Exception as e:
                if "Stream closed" in str(e) or "Unanticipated host error" in str(e):
                    if self._handle_audio_error():
                        return self.smooth_buffer['low'], self.smooth_buffer['high']
                return self.smooth_buffer['low'], self.smooth_buffer['high']

            data_np = np.frombuffer(data, dtype=np.float32)

            if len(data_np) == 0:
                if self._handle_audio_error():
                    return self.smooth_buffer['low'], self.smooth_buffer['high']
                return self.smooth_buffer['low'], self.smooth_buffer['high']

            # 检测音高
            frequency, confidence = self._detect_pitch(data_np)
            if confidence > 0.5:  # 只在置信度较高时更新音符
                self.current_note = self._frequency_to_note(frequency)
                self.pitch_confidence = confidence

            # FFT分析
            fft_data = np.abs(fft(data_np)[:self.config.CHUNK // 2])
            freqs = np.fft.fftfreq(self.config.CHUNK, 1 / self.config.RATE)[:self.config.CHUNK // 2]

            # 计算动态权重
            self._calculate_dynamic_weights(fft_data, freqs, freq_split)

            # 频率分离
            low_mask = freqs < freq_split
            high_mask = freqs >= freq_split

            # 添加到区块链
            self.blockchain.add_block(fft_data)

            # 计算新的能量值（应用动态权重）
            low_energy = self._calculate_energy(low_mask, difficulty, history_weight) * self.energy_weights['low']
            high_energy = self._calculate_energy(high_mask, difficulty, history_weight) * self.energy_weights['high']

            # 应用平滑
            smooth_factor = 0.3  # 平滑因子
            self.smooth_buffer['low'] = smooth_factor * low_energy + (1 - smooth_factor) * self.smooth_buffer['low']
            self.smooth_buffer['high'] = smooth_factor * high_energy + (1 - smooth_factor) * self.smooth_buffer['high']

            return self.smooth_buffer['low'], self.smooth_buffer['high']

        except Exception as e:
            print(f"处理音频数据时出错: {e}")
            if self._handle_audio_error():
                return self.smooth_buffer['low'], self.smooth_buffer['high']
            return self.smooth_buffer['low'], self.smooth_buffer['high']

    def _calculate_energy(self, freq_mask: np.ndarray, difficulty: float, history_weight: float) -> float:
        """计算能量值"""
        if not self.blockchain.chain:
            return 0.0

        try:
            # 当前能量
            current = self.blockchain.chain[-1].get_energy(freq_mask)

            # 历史能量
            historical = 0.0
            weight = 1.0
            total_weight = 0.0

            for block in reversed(self.blockchain.chain):
                energy = block.get_energy(freq_mask)
                historical += energy * weight
                total_weight += weight
                weight *= history_weight

            historical /= total_weight if total_weight > 0 else 1

            # 应用难度系数
            energy = history_weight * current + (1 - history_weight) * historical
            threshold = 1.0 / (difficulty * 2)
            return min(1.0, max(0, energy - threshold) * difficulty)

        except Exception as e:
            # 计算出错时返回0
            return 0.0

    def cleanup(self):
        """清理资源"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except:
            pass  # 忽略清理时的错误

    def _get_available_devices(self) -> List[Tuple[int, str]]:
        """获取所有可用的音频输入设备"""
        devices = []
        seen_names = set()

        try:
            # 首先尝试获取默认输入设备
            try:
                default_input = self.audio.get_default_input_device_info()
                if default_input and default_input['maxInputChannels'] > 0:
                    name = default_input['name']
                    seen_names.add(name)
                    devices.append((int(default_input['index']), f"{name} (默认)"))
            except:
                pass

            # 然后查找立体声混音设备
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    name = device_info['name']

                    # 跳过已经添加的设备名称
                    if name in seen_names:
                        continue

                    # 检查是否是有效的输入设备
                    if device_info['maxInputChannels'] > 0:
                        # 优先添加立体声混音设备
                        if any(keyword in name.lower() for keyword in
                               ['立体声混音', 'stereo mix', 'wave', 'what u hear']):
                            seen_names.add(name)
                            devices.append((i, f"{name} (系统音频)"))
                except:
                    continue

            # 最后添加其他输入设备
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    name = device_info['name']

                    # 跳过已经添加的设备名称
                    if name in seen_names:
                        continue

                    # 检查是否是有效的输入设备
                    if device_info['maxInputChannels'] > 0:
                        # 过滤掉一些可能有问题的设备
                        if not any(keyword in name.lower() for keyword in ['dummy', 'null', 'empty']):
                            seen_names.add(name)
                            devices.append((i, name))
                except:
                    continue

        except Exception as e:
            print(f"枚举音频设备时出错: {e}")

        if not devices:
            # 如果没有找到任何设备，添加一个默认设备
            devices.append((0, "默认输入设备"))

        return devices

    def set_input_device(self, device_index: int):
        """设置音频输入设备"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

            self.device_index = device_index
            self._initialize_audio_stream()
            return True
        except Exception as e:
            print(f"设置音频设备失败: {e}")
            return False


class UIControl(abc.ABC):
    """UI控件基类"""

    @abc.abstractmethod
    def draw(self, surface: pygame.Surface):
        pass

    @abc.abstractmethod
    def handle_event(self, event: pygame.event.Event):
        pass


class Slider(UIControl):
    """滑动条控件"""

    def __init__(self, rect: pygame.Rect, label: str, min_val: float, max_val: float, initial: float):
        self.rect = rect
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.handle_width = 20
        self.handle_height = 30
        self.is_dragging = False

    def draw(self, surface: pygame.Surface):
        # 绘制背景
        pygame.draw.rect(surface, VisualizerConfig.UI_COLOR, self.rect)

        # 绘制滑块
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * (
                self.rect.width - self.handle_width)
        handle_rect = pygame.Rect(handle_x, self.rect.y - (self.handle_height - self.rect.height) // 2,
                                  self.handle_width, self.handle_height)
        pygame.draw.rect(surface, (255, 255, 255), handle_rect)

        # 绘制标签
        try:
            # 尝试使用系统字体
            font = pygame.font.Font(VisualizerConfig.FONT_PATH, VisualizerConfig.FONT_SIZE)
        except:
            try:
                # 尝试使用系统默认中文字体
                font = pygame.font.SysFont("microsoftyahei", VisualizerConfig.FONT_SIZE)
            except:
                # 如果都失败，使用默认字体
                font = pygame.font.Font(None, VisualizerConfig.FONT_SIZE)

        try:
            text = font.render(f"{self.label}: {self.value:.2f}", True, (255, 255, 255))
            surface.blit(text, (self.rect.x, self.rect.y - 25))
        except:
            print(f"渲染文本失败: {self.label}")

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_dragging = True
                self._update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            self._update_value(event.pos[0])

    def _update_value(self, x: int):
        relative_x = (x - self.rect.x) / self.rect.width
        self.value = self.min_val + (self.max_val - self.min_val) * max(0, min(1, relative_x))


class Dropdown(UIControl):
    """下拉菜单控件"""

    def __init__(self, rect: pygame.Rect, label: str, options: List[str], initial: int = 0):
        self.rect = rect
        self.label = label
        self.options = options
        self.selected_index = initial
        self.is_open = False
        self.handle_height = 30
        self.option_height = 25
        self.max_visible_options = 5

    def draw(self, surface: pygame.Surface):
        # 绘制背景
        pygame.draw.rect(surface, VisualizerConfig.UI_COLOR, self.rect)

        # 绘制当前选中的选项
        try:
            font = pygame.font.Font(VisualizerConfig.FONT_PATH, VisualizerConfig.FONT_SIZE)
        except:
            try:
                font = pygame.font.SysFont("microsoftyahei", VisualizerConfig.FONT_SIZE)
            except:
                font = pygame.font.Font(None, VisualizerConfig.FONT_SIZE)

        # 绘制标签
        label_text = font.render(self.label, True, (255, 255, 255))
        surface.blit(label_text, (self.rect.x, self.rect.y - 25))

        # 绘制当前选中的选项
        current_text = font.render(self.options[self.selected_index], True, (255, 255, 255))
        surface.blit(current_text, (self.rect.x + 5, self.rect.y + 5))

        # 如果下拉菜单打开，绘制选项列表
        if self.is_open:
            # 计算可见选项的范围
            start_idx = max(0, min(self.selected_index - self.max_visible_options // 2,
                                   len(self.options) - self.max_visible_options))
            visible_options = self.options[start_idx:start_idx + self.max_visible_options]

            # 绘制选项背景
            options_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height,
                                       self.rect.width, len(visible_options) * self.option_height)
            pygame.draw.rect(surface, (70, 70, 70), options_rect)

            # 绘制选项
            for i, option in enumerate(visible_options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height + i * self.option_height,
                                          self.rect.width, self.option_height)
                if i + start_idx == self.selected_index:
                    pygame.draw.rect(surface, (100, 100, 100), option_rect)
                option_text = font.render(option, True, (255, 255, 255))
                surface.blit(option_text, (option_rect.x + 5, option_rect.y + 5))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
            elif self.is_open:
                # 检查是否点击了选项
                click_y = event.pos[1]
                if (self.rect.y + self.rect.height <= click_y <=
                        self.rect.y + self.rect.height + len(self.options) * self.option_height):
                    start_idx = max(0, min(self.selected_index - self.max_visible_options // 2,
                                           len(self.options) - self.max_visible_options))
                    option_index = (click_y - (self.rect.y + self.rect.height)) // self.option_height
                    if 0 <= option_index < len(self.options):
                        self.selected_index = start_idx + option_index
                        self.is_open = False
        elif event.type == pygame.MOUSEBUTTONUP:
            if not self.rect.collidepoint(event.pos) and self.is_open:
                self.is_open = False


class WaveVisualizer:
    """波形可视化器"""

    def __init__(self):
        try:
            pygame.init()
            pygame.font.init()

            self.config = VisualizerConfig()
            self.current_size_index = 0  # 当前窗口大小的索引
            self.screen = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption("音频可视化")

            self.audio = AudioProcessor(AudioConfig())
            self.clock = pygame.time.Clock()
            self.running = True
            self.show_controls = False
            self.show_waves = True  # 波形默认显示
            self.show_lights = True  # 音符灯带默认显示
            self.show_ground = True  # 地面投影默认显示
            self.show_particles = True  # 粒子效果默认显示

            # 预计算 X 坐标数组，避免重复计算
            self.x_coords = np.arange(self.config.WIDTH)

            # 创建控制器
            self._create_controls()

            # 预创建波形表面
            self.wave_surface = pygame.Surface((self.config.WIDTH, self.config.HEIGHT))

            # 预创建地面投影表面（带alpha通道）
            self.ground_surface = pygame.Surface((self.config.WIDTH, self.config.HEIGHT), pygame.SRCALPHA)

            # 初始化波形数据
            self.wave_high = np.zeros(self.config.WIDTH, dtype=np.int32)
            self.wave_low = np.zeros(self.config.WIDTH, dtype=np.int32)

            # 初始化音符颜色映射
            self._init_note_colors()

            # 初始化灯光状态
            self.light_states = {}

            # 地面投影状态
            self.current_ground_color = (0, 0, 0)
            self.target_ground_color = (0, 0, 0)
            self.ground_energy = 0.0
            self.color_transition_speed = 0.1

            # 初始化粒子系统
            self.particle_system = ParticleSystem()
            self.last_time = time.time()

        except Exception as e:
            print(f"初始化失败: {e}")
            pygame.quit()
            raise

    def _create_controls(self):
        """创建UI控件"""
        self._update_controls_position()

    def _init_note_colors(self):
        """初始化音符到颜色的映射"""
        # 使用HSV颜色空间来生成均匀分布的颜色
        self.note_colors = {
            'C': (255, 0, 0),  # 红
            'C#': (255, 127, 0),  # 橙
            'D': (255, 255, 0),  # 黄
            'D#': (127, 255, 0),  # 黄绿
            'E': (0, 255, 0),  # 绿
            'F': (0, 255, 127),  # 青绿
            'F#': (0, 255, 255),  # 青
            'G': (0, 127, 255),  # 青蓝
            'G#': (0, 0, 255),  # 蓝
            'A': (127, 0, 255),  # 蓝紫
            'A#': (255, 0, 255),  # 紫
            'B': (255, 0, 127)  # 紫红
        }

    def _draw_note_lights(self):
        """绘制音符灯带"""
        light_width = self.config.LIGHT_WIDTH
        light_height = self.config.LIGHT_HEIGHT
        spacing = self.config.LIGHT_SPACING
        total_width = (light_width + spacing) * len(self.note_colors)
        start_x = (self.config.WIDTH - total_width) // 2

        # 更新灯光状态
        current_note = self.audio.current_note
        if current_note and self.audio.pitch_confidence > 0.5:
            note_name = current_note[:-1]  # 移除八度数字
            self.light_states[note_name] = 1.0  # 完全亮起

        # 绘制所有灯光
        for i, (note, base_color) in enumerate(self.note_colors.items()):
            # 获取当前亮度，如果不存在则为0
            brightness = self.light_states.get(note, 0.0)

            # 计算实际颜色（考虑亮度）
            color = tuple(int(c * brightness) for c in base_color)

            # 绘制灯光
            x = start_x + i * (light_width + spacing)
            rect = pygame.Rect(x, 10, light_width, light_height)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)  # 边框

            # 淡出效果
            if brightness > 0:
                self.light_states[note] = max(0, brightness - self.config.LIGHT_FADE_SPEED)

    def _draw_ground_projection(self, high_energy: float, low_energy: float):
        """绘制地面投影效果"""
        # 清除上一帧
        self.ground_surface.fill((0, 0, 0, 0))

        # 获取当前音符和能量
        current_note = self.audio.current_note
        energy = max(high_energy, low_energy)

        # 更新目标颜色
        if current_note and self.audio.pitch_confidence > 0.5:
            note_name = current_note[:-1]
            self.target_ground_color = self.note_colors.get(note_name, (0, 0, 0))

        # 平滑过渡到目标颜色
        self.current_ground_color = tuple(
            int(self.current_ground_color[i] +
                (self.target_ground_color[i] - self.current_ground_color[i]) *
                self.color_transition_speed)
            for i in range(3)
        )

        # 平滑能量变化
        self.ground_energy = self.ground_energy * 0.9 + energy * 0.1

        # 计算椭圆参数
        center_x = self.config.WIDTH // 2
        center_y = self.config.GROUND_Y
        base_radius = max(1, self.config.BASE_RADIUS)  # 确保最小半径为1
        max_radius = max(base_radius, self.config.MAX_RADIUS)

        # 根据能量计算当前半径
        current_radius = base_radius + (max_radius - base_radius) * self.ground_energy

        # 绘制多层发光效果
        for i in range(self.config.GLOW_LAYERS):
            layer_radius = max(1, current_radius * (1 - i / self.config.GLOW_LAYERS))  # 确保最小半径为1
            alpha = int(255 * (1 - i / self.config.GLOW_LAYERS) * self.ground_energy)
            color = (*self.current_ground_color, alpha)

            # 计算椭圆矩形
            width = max(2, layer_radius * 4)  # 确保最小宽度为2
            height = max(2, layer_radius * 0.6)  # 确保最小高度为2
            x = center_x - width / 2
            y = center_y - height / 2

            # 确保矩形在屏幕范围内且尺寸有效
            if (width > 0 and height > 0 and
                    x >= 0 and x + width <= self.config.WIDTH and
                    y >= 0 and y + height <= self.config.HEIGHT):
                try:
                    pygame.draw.ellipse(
                        self.ground_surface,
                        color,
                        (int(x), int(y), int(width), int(height))
                    )
                except ValueError as e:
                    print(f"椭圆绘制错误: {e}, 参数: x={x}, y={y}, w={width}, h={height}")
                    continue

    def run(self):
        """主循环"""
        try:
            while self.running:
                self._handle_events()
                self._update()
                self._draw()
                self.clock.tick(self.config.FPS)
        finally:
            self.cleanup()

    def _handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.show_controls = not self.show_controls
                elif event.key == pygame.K_w:
                    self.show_waves = not self.show_waves
                elif event.key == pygame.K_l:
                    self.show_lights = not self.show_lights
                elif event.key == pygame.K_g:
                    self.show_ground = not self.show_ground
                elif event.key == pygame.K_p:
                    self.show_particles = not self.show_particles
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    size_index = event.key - pygame.K_1
                    if size_index < len(self.config.WINDOW_SIZES):
                        self.current_size_index = size_index
                        new_width, new_height = self.config.WINDOW_SIZES[size_index]
                        self.config.WIDTH = new_width
                        self.config.HEIGHT = new_height
                        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                        self.wave_surface = pygame.Surface((new_width, new_height))
                        self.ground_surface = pygame.Surface((new_width, new_height), pygame.SRCALPHA)
                        self.x_coords = np.arange(new_width)
                        self.wave_high = np.zeros(new_width, dtype=np.int32)
                        self.wave_low = np.zeros(new_width, dtype=np.int32)
                        self.particle_system.resize(new_width, new_height)
                        self._update_controls_position()
            elif event.type == pygame.VIDEORESIZE:
                self.config.WIDTH = event.w
                self.config.HEIGHT = event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.wave_surface = pygame.Surface((event.w, event.h))
                self.ground_surface = pygame.Surface((event.w, event.h), pygame.SRCALPHA)
                self.x_coords = np.arange(event.w)
                self.wave_high = np.zeros(event.w, dtype=np.int32)
                self.wave_low = np.zeros(event.w, dtype=np.int32)
                self.particle_system.resize(event.w, event.h)
                self._update_controls_position()

            if self.show_controls:
                for control in self.controls.values():
                    control.handle_event(event)
                    # 处理设备选择
                    if isinstance(control, Dropdown) and control.label == "音频设备":
                        if not control.is_open and control.selected_index != self.audio.device_index:
                            device_index = self.audio.available_devices[control.selected_index][0]
                            if self.audio.set_input_device(device_index):
                                print(f"已切换到音频设备: {control.options[control.selected_index]}")

    def _update(self):
        """更新状态"""
        # 获取音频数据
        low_energy, high_energy = self.audio.process_audio(
            self.controls['freq_split'].value,
            self.controls['difficulty'].value,
            self.controls['history'].value
        )

        # 使用 numpy 向量化操作生成波形数据
        t = pygame.time.get_ticks() / 1000
        amplitude = 100 * self.controls['amplitude'].value
        frequency = 0.2 * self.controls['width'].value

        # 计算波形
        phase = 2 * np.pi * (self.x_coords * frequency + t)
        self.wave_high = (self.config.HEIGHT / 2 + amplitude * np.sin(phase) * high_energy).astype(np.int32)
        self.wave_low = (self.config.HEIGHT / 2 + amplitude * np.cos(phase) * low_energy).astype(np.int32)

        # 更新粒子系统
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.particle_system.update(dt)

        # 根据音频能量密度添加粒子
        if self.show_particles:
            # 计算能量密度
            energy_density = (high_energy + low_energy) / 2

            # 根据能量密度决定是否生成粒子
            if energy_density > 0.2:  # 能量密度阈值
                # 计算粒子生成位置
                center_x = self.config.WIDTH // 2
                center_y = self.config.HEIGHT // 2

                # 根据能量密度决定粒子数量
                num_particles = int(energy_density * 50)  # 基础粒子数

                # 生成高频粒子
                for _ in range(num_particles):
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(0, 200 * energy_density)  # 根据能量调整扩散范围
                    x = center_x + distance * np.cos(angle)
                    y = center_y + distance * np.sin(angle)
                    self.particle_system.add_particles(
                        x, y,
                        self.config.HIGH_FREQ_COLOR,
                        energy_density
                    )

                # 生成低频粒子
                for _ in range(num_particles):
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(0, 150 * energy_density)  # 低频粒子扩散范围稍小
                    x = center_x + distance * np.cos(angle)
                    y = center_y + distance * np.sin(angle)
                    self.particle_system.add_particles(
                        x, y,
                        self.config.LOW_FREQ_COLOR,
                        energy_density
                    )

    def _draw(self):
        """绘制画面"""
        self.screen.fill(self.config.BACKGROUND_COLOR)

        # 获取当前音频能量
        low_energy = self.audio.smooth_buffer['low']
        high_energy = self.audio.smooth_buffer['high']

        # 绘制波形（移到地面投影之前）
        if self.show_waves:
            self.wave_surface.fill(self.config.BACKGROUND_COLOR)

            points_high = np.column_stack((self.x_coords, self.wave_high))
            points_low = np.column_stack((self.x_coords, self.wave_low))

            pygame.draw.aalines(self.wave_surface, self.config.HIGH_FREQ_COLOR, False, points_high, True)
            pygame.draw.aalines(self.wave_surface, self.config.LOW_FREQ_COLOR, False, points_low, True)

            self.screen.blit(self.wave_surface, (0, 0))

        # 绘制地面投影（在波形之后）
        if self.show_ground:
            self._draw_ground_projection(high_energy, low_energy)
            self.screen.blit(self.ground_surface, (0, 0))

        # 绘制粒子效果
        if self.show_particles:
            self.particle_system.draw(self.screen)

        # 绘制音符灯带（始终在最上层）
        if self.show_lights:
            self._draw_note_lights()

        # 绘制控制界面（在最上层）
        if self.show_controls:
            self._draw_controls()

        pygame.display.flip()

    def _draw_controls(self):
        """绘制控制界面"""
        # 绘制半透明背景
        overlay = pygame.Surface((self.config.WIDTH, self.config.HEIGHT))
        overlay.fill((50, 50, 50))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))

        # 绘制控件
        for control in self.controls.values():
            control.draw(self.screen)

    def cleanup(self):
        """清理资源"""
        self.audio.cleanup()
        pygame.quit()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

    def _update_controls_position(self):
        """更新控制器位置"""
        base_x = (self.config.WIDTH - 600) // 2
        base_y = self.config.HEIGHT - 300  # 增加一点空间给设备选择器

        # 获取可用设备列表
        device_options = [f"{name}" for _, name in self.audio.available_devices]
        if not device_options:
            device_options = ["无可用设备"]

        offset_x = 300  # 你可以根据需要调整这个值
        self.controls = {
            'width': Slider(pygame.Rect(base_x, base_y + 50, 300, 15), "波浪宽度", 0.1, 1.0,
                            self.controls['width'].value if hasattr(self, 'controls') else 0.5),
            'amplitude': Slider(pygame.Rect(base_x, base_y + 90, 300, 15), "波浪振幅", 0.1, 1.0,
                                self.controls['amplitude'].value if hasattr(self, 'controls') else 0.5),
            'difficulty': Slider(pygame.Rect(base_x, base_y + 130, 300, 15), "难度系数", 1.0, 5.0,
                                 self.controls['difficulty'].value if hasattr(self, 'controls') else 2.0),
            'freq_split': Slider(pygame.Rect(base_x, base_y + 170, 300, 15), "频率分割", 500, 2000,
                                 self.controls['freq_split'].value if hasattr(self, 'controls') else 1000),
            'history': Slider(pygame.Rect(base_x, base_y + 210, 300, 15), "历史权重", 0.1, 0.9,
                              self.controls['history'].value if hasattr(self, 'controls') else 0.7),
            'device': Dropdown(pygame.Rect(base_x + offset_x, base_y, 300, 30), "音频数据入口", device_options)
        }


if __name__ == "__main__":
    try:
        with WaveVisualizer() as visualizer:
            visualizer.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
