# ai_music
博客代码：快过年了，搞个AI作曲，用TensorFlow训练midi文件

## 新增内容（2026-08-04）
- 新增 PyTorch 实现脚本 `midi_main_pytorch.py`（已提交到当前分支），功能与原 TensorFlow 实现等价：读取 datasets/ 中的 MIDI 文件，训练 LSTM 模型或生成 MIDI（默认运行生成）。
- 添加 `requirements.txt`（在仓库根目录），列出运行 PyTorch 版本所需的主要依赖项。

## 如何运行（PyTorch 版本）
1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. 训练模型：

```bash
python midi_main_pytorch.py --train --epochs 50 --batch-size 64 --lr 0.01
```

训练会把最好的 checkpoint 保存到 `model/model.pth`。

3. 生成 MIDI（预测）：

```bash
python midi_main_pytorch.py --predict --num-predictions 600
```

默认不带参数会执行预测并把生成结果写为 `out_pytorch.midi`。

## 说明与建议
- PyTorch 版本使用的 checkpoint 存为 `model/model.pth`（与原 TensorFlow `.ckpt` 格式不兼容）。
- 如果数据集很大，当前的 Dataset 会一次性构建所有滑动窗口；可根据需要改为按需生成以节省内存。
- 建议根据你本地环境安装合适的 `torch` 版本（CPU/GPU）。

---
原 README 内容保留并扩展说明了 PyTorch 版本的使用方法。
