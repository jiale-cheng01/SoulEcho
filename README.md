# SoulEcho - 音频可视化程序

一个基于 Python 的实时音频可视化程序，能够捕获系统音频并创建动态的视觉效果。

## 功能特点

- 实时音频捕获和分析
- 动态粒子效果
- 频谱可视化
- 音符灯带显示
- 地面投影效果
- 可调节的参数控制
- 多设备音频输入支持

## 系统要求

- Python 3.8+
- Windows 10/11
- VB-CABLE Virtual Audio Device（用于系统音频捕获）

## 依赖项

```
pygame
numpy
pyaudio
scipy
```

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/你的用户名/wavesoul.git
cd wavesoul
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装 VB-CABLE Virtual Audio Device（用于系统音频捕获）：
   - 从 [VB-CABLE 官网](https://vb-audio.com/Cable/) 下载并安装
   - 在 Windows 声音设置中将 CABLE Input 设为默认播放设备
   - 在程序中选择 CABLE Output 作为输入设备

## 使用方法

1. 运行程序：
```bash
python main.py
```

2. 界面控制：
   - 使用滑块调节各种参数：
     - 波形宽度
     - 振幅
     - 难度系数
     - 频率分割
     - 历史权重
   - 从下拉菜单选择音频输入设备
   - 按 F 键切换UI界面
   - 按 ESC 键退出程序

## 配置说明

### 音频配置 (AudioConfig)
- CHUNK: 音频数据块大小
- FORMAT: 音频格式 (Float32)
- CHANNELS: 声道数
- RATE: 采样率

### 可视化配置 (VisualizerConfig)
- 窗口大小预设
- 颜色设置
- 粒子系统参数
- 灯带效果参数
- 地面投影参数

## 代码结构

- `main.py`: 主程序文件
  - AudioProcessor: 音频处理
  - WaveVisualizer: 可视化实现
  - ParticleSystem: 粒子系统
  - UIControl: 用户界面控件

## 许可证

MIT License

## 作者

[jiale-cheng01]

## 致谢

- PyAudio
- Pygame
- NumPy
- SciPy

## 开发计划

- [ ] 3D效果支持
- [ ] 更多视觉效果
- [ ] 性能优化
- [ ] 更多交互功能

## 贡献

欢迎提交Issue和Pull Request！ 
