# ai_music
博客代码：快过年了，搞个AI作曲，用TensorFlow训练midi文件

## 新增内容（2026-08-04 / 2026-08-05）
- 新增 PyTorch 实现脚本 `midi_main_pytorch.py`（已提交到当前分支），功能与原 TensorFlow 实现等价：读取 `datasets/` 中的 MIDI 文件，训练模型或生成 MIDI。
- 多次迭代改进：加入 LSTM 改进版、Transformer 编码器、逐步自回归生成支持（通过 Transformer 解码器式设计的自回归调用）、以及 pitch 采样控制（temperature、top-k、top-p）。
- 添加 `requirements.txt`（在仓库根目录），列出运行 PyTorch 版本所需的主要依赖项。

## 如何运行（PyTorch 版本）
1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

注意：`requirements.txt` 中包含通用条目（pretty_midi、torch、numpy）。建议按你的平台（CPU / CUDA）从 https://pytorch.org 获取并安装合适的 `torch` wheel 以获得最佳性能。

2. 训练模型（示例）

- 使用 Transformer（示例超参）：

```bash
python midi_main_pytorch.py --train --model-type transformer --epochs 50 --batch-size 64 --lr 0.005 \
  --d-model 128 --nhead 8 --num-layers 4 --embed-dim 64
```

- 使用 LSTM（回退实现）：

```bash
python midi_main_pytorch.py --train --model-type lstm --epochs 50 --batch-size 64 --lr 0.005 \
  --lstm-embed-dim 32 --hidden-size 256 --lstm-num-layers 2
```

训练会把最好的 checkpoint 保存到 `model/model.pth`（包含 model_meta 与 dataset_stats）。

3. 生成 MIDI（预测 / 采样示例）

默认生成（与历史行为兼容）：

```bash
python midi_main_pytorch.py --predict
```

使用采样控制（temperature、top-k、top-p）以平衡多样性与质量：

```bash
# 更确定的采样（较少随机）：
python midi_main_pytorch.py --predict --temperature 0.8 --top-k 12 --model-type transformer

# 更随机的采样（更多多样性）：
python midi_main_pytorch.py --predict --temperature 1.2 --top-p 0.9 --model-type transformer
```

常用采样配置示例：
- temperature < 1.0: 更保守（更确定）
- temperature > 1.0: 更随机（更多多样性）
- top-k: 只在概率最高的 k 个 token 中采样（0 表示禁用）
- top-p: nucleus 采样，限制累积概率到 p（0.0 表示禁用）

生成的文件名为 `out_pytorch.midi`，checkpoint 存为 `model/model.pth`（默认被 `.gitignore` 忽略）。

## Smoke-test（快速验证）
为了快速验证训练与生成流程可以在本地或 CI 做一次最小试验：

```bash
# 快速 smoke-test：少量 epoch 和小 batch
python midi_main_pytorch.py --train --epochs 1 --batch-size 8 --lr 0.005 --model-type transformer
# 然后运行一次生成（默认采样）：
python midi_main_pytorch.py --predict --model-type transformer
```

Smoke-test 可以确认代码在你的环境中能正常跑通而不需要长时间训练。

## 说明与建议
- checkpoint 与数据统计：Checkpoint 中包含 `dataset_stats`（step_mean/step_std/duration_mean/duration_std），predict 会优先使用这些统计进行反归一化。
- 采样策略建议：常见组合是 temperature (0.7-1.2) + top-k (8-20) 或 top-p (0.8-0.95)。多试几组超参以找到你想要的音色/多样性平衡。
- 数据处理：目前 Dataset 会一次性构建所有滑动窗口；如果你的训练集非常大，建议改成按需生成窗口以节省内存。
- 进一步改进建议（非必须）：把 pitch 与时长联合 embedding、加入节奏/节拍信息、使用 Transformer 解码器的逐步自回归实现以提高生成一致性。

---
原 README 内容保留并扩展说明了 PyTorch 版本的使用方法与采样参数示例。
