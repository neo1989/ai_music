"""
单元测试套件：midi_main_pytorch_improved.py

测试覆盖：
- RhythmAnalyzer (时间签名分析、节拍量化、强度提取)
- JointPitchDurationEmbedding
- PositionalEncoding
- ImprovedMidiSequenceDataset
- ImprovedAutoregressiveTransformer
- 工具函数 (采样、过滤、损失)
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from collections import namedtuple
import sys
import os

# 导入被测试模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midi_main_pytorch_improved import (
    RhythmAnalyzer,
    JointPitchDurationEmbedding,
    RhythmAwareCachedTransformerLayer,
    ImprovedAutoregressiveTransformer,
    ImprovedMidiSequenceDataset,
    PositionalEncoding,
    generate_causal_mask,
    top_k_top_p_filtering,
    sample_from_logits,
    MSEWithPositivePressure,
)


# Mock Note 类用于测试
Note = namedtuple('Note', ['start', 'end', 'pitch'])


class TestRhythmAnalyzer(unittest.TestCase):
    """测试 RhythmAnalyzer 类"""
    
    def setUp(self):
        self.analyzer = RhythmAnalyzer(tempo=120.0)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.analyzer.tempo, 120.0)
        self.assertAlmostEqual(self.analyzer.beat_duration, 0.5, places=5)
    
    def test_different_tempos(self):
        """测试不同的速度"""
        analyzer_60 = RhythmAnalyzer(tempo=60.0)
        analyzer_240 = RhythmAnalyzer(tempo=240.0)
        
        self.assertAlmostEqual(analyzer_60.beat_duration, 1.0, places=5)
        self.assertAlmostEqual(analyzer_240.beat_duration, 0.25, places=5)
    
    def test_analyze_time_signature_empty_notes(self):
        """测试空音符列表"""
        result = self.analyzer.analyze_time_signature([])
        self.assertEqual(result, (4, 4))
    
    def test_analyze_time_signature_single_note(self):
        """测试单个音符"""
        notes = [Note(start=0.0, end=0.5, pitch=60)]
        result = self.analyzer.analyze_time_signature(notes)
        self.assertEqual(result, (4, 4))
    
    def test_analyze_time_signature_uniform_intervals(self):
        """测试均匀分布的音符（应识别为4/4）"""
        # 创建间隔均匀的音符序列
        notes = [Note(start=i*0.5, end=i*0.5+0.25, pitch=60) for i in range(8)]
        result = self.analyzer.analyze_time_signature(notes)
        # 间隔都是0.5（base_unit）
        self.assertIn(result[1], [4, 8])  # denominator 应是 4 或 8
    
    def test_analyze_time_signature_triple_meter(self):
        """测试三拍子节奏"""
        # 创建3拍子的音符序列（间隔比例 3:3:3）
        intervals = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5]
        notes = [Note(start=t, end=t+0.25, pitch=60) for t in intervals]
        result = self.analyzer.analyze_time_signature(notes)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0], 0)
        self.assertGreater(result[1], 0)
    
    def test_analyze_time_signature_with_midi_notes(self):
        """测试使用 pretty_midi 风格的 Note 对象"""
        notes = [
            Note(start=0.0, end=0.25, pitch=60),
            Note(start=0.5, end=0.75, pitch=64),
            Note(start=1.0, end=1.25, pitch=67),
            Note(start=1.5, end=1.75, pitch=72),
        ]
        result = self.analyzer.analyze_time_signature(notes)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_quantize_to_beat_basic(self):
        """测试节拍量化"""
        # beat_duration = 0.5, grid = 16
        # beat_unit = 0.5 / 16 = 0.03125
        result = self.analyzer.quantize_to_beat(0.0, grid=16)
        self.assertEqual(result, 0)  # round(0 / 0.03125) = 0, max(1, 0) = 1，但 0 会被 round 捕获
        
        result = self.analyzer.quantize_to_beat(0.25, grid=16)
        self.assertGreaterEqual(result, 1)  # 至少为 1
    
    def test_quantize_to_beat_grid_values(self):
        """测试不同的网格值"""
        for grid in [8, 16, 32]:
            result = self.analyzer.quantize_to_beat(0.5, grid=grid)
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 1)
    
    def test_extract_beat_position(self):
        """测试节拍位置提取"""
        # beat_duration = 0.5
        result = self.analyzer.extract_beat_position(0.0, grid=16)
        self.assertAlmostEqual(result, 0.0, places=5)
        
        result = self.analyzer.extract_beat_position(0.25, grid=16)
        self.assertAlmostEqual(result, 0.5, places=5)
        
        result = self.analyzer.extract_beat_position(0.5, grid=16)
        self.assertAlmostEqual(result, 0.0, places=5)  # 循环
    
    def test_extract_beat_position_range(self):
        """测试节拍位置在 0.0-1.0 范围内"""
        for t in np.linspace(0, 2.0, 20):
            result = self.analyzer.extract_beat_position(t, grid=16)
            self.assertGreaterEqual(result, 0.0)
            self.assertLess(result, 1.0)
    
    def test_extract_beat_strength_downbeat(self):
        """测试强下拍（downbeat）"""
        # quantized_step % 16 == 0
        result = self.analyzer.extract_beat_strength(0, grid=16)
        self.assertEqual(result, 1.0)
        
        result = self.analyzer.extract_beat_strength(16, grid=16)
        self.assertEqual(result, 1.0)
    
    def test_extract_beat_strength_on_beat(self):
        """测试弱拍（on-beat）"""
        # quantized_step % 16 == 4, 8, 12
        result = self.analyzer.extract_beat_strength(4, grid=16)
        self.assertEqual(result, 0.7)
        
        result = self.analyzer.extract_beat_strength(8, grid=16)
        self.assertEqual(result, 0.7)
    
    def test_extract_beat_strength_off_beat(self):
        """测试弱拍（off-beat）"""
        # quantized_step % 16 不是 0 或 4 的倍数
        result = self.analyzer.extract_beat_strength(1, grid=16)
        self.assertEqual(result, 0.3)
        
        result = self.analyzer.extract_beat_strength(5, grid=16)
        self.assertEqual(result, 0.3)


class TestJointPitchDurationEmbedding(unittest.TestCase):
    """测试 JointPitchDurationEmbedding"""
    
    def setUp(self):
        self.embed_dim = 64
        self.vocab_size = 128
        self.duration_buckets = 16
        self.embedding = JointPitchDurationEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_duration=4.0,
            duration_buckets=self.duration_buckets
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.embedding.vocab_size, self.vocab_size)
        self.assertEqual(self.embedding.embed_dim, self.embed_dim)
        self.assertEqual(self.embedding.duration_buckets, self.duration_buckets)
    
    def test_duration_to_bucket(self):
        """测试 duration 到 bucket 的转换"""
        durations = torch.tensor([[0.0, 1.0, 2.0, 4.0]], dtype=torch.float32)
        buckets = self.embedding._duration_to_bucket(durations)
        
        self.assertEqual(buckets.shape, durations.shape)
        self.assertTrue(torch.all(buckets >= 0))
        self.assertTrue(torch.all(buckets < self.duration_buckets))
    
    def test_duration_to_bucket_edge_cases(self):
        """测试 bucket 转换的边界情况"""
        # 超过最大值的 duration
        durations = torch.tensor([[8.0]], dtype=torch.float32)
        buckets = self.embedding._duration_to_bucket(durations)
        self.assertEqual(buckets[0, 0].item(), self.duration_buckets - 1)
        
        # 最小值
        durations = torch.tensor([[0.0]], dtype=torch.float32)
        buckets = self.embedding._duration_to_bucket(durations)
        self.assertEqual(buckets[0, 0].item(), 0)
    
    def test_forward_shape(self):
        """测试前向传播的输出形状"""
        batch_size = 2
        seq_len = 32
        
        pitch_idxs = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        durations = torch.rand(batch_size, seq_len) * 4.0
        
        output = self.embedding(pitch_idxs, durations)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.embed_dim))
    
    def test_forward_single_sequence(self):
        """测试单个序列的前向传播"""
        batch_size = 1
        seq_len = 1
        
        pitch_idxs = torch.tensor([[60]], dtype=torch.long)
        durations = torch.tensor([[1.0]], dtype=torch.float32)
        
        output = self.embedding(pitch_idxs, durations)
        
        self.assertEqual(output.shape, (1, 1, self.embed_dim))
    
    def test_embedding_is_differentiable(self):
        """测试 embedding 是否可微"""
        pitch_idxs = torch.tensor([[60, 64, 67]], dtype=torch.long)
        durations = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, requires_grad=True)
        
        output = self.embedding(pitch_idxs, durations)
        loss = output.sum()
        loss.backward()
        
        # joint_proj 的参数应该有梯度
        self.assertIsNotNone(self.embedding.joint_proj.weight.grad)


class TestPositionalEncoding(unittest.TestCase):
    """测试 PositionalEncoding"""
    
    def setUp(self):
        self.d_model = 128
        self.pos_enc = PositionalEncoding(d_model=self.d_model)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.pos_enc.pe.shape[1], self.d_model)
    
    def test_forward_shape(self):
        """测试前向传播的形状"""
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, self.d_model)
        output = self.pos_enc(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_positional_encoding_values(self):
        """测试位置编码的值"""
        # 相同位置的编码应该相同
        x1 = torch.zeros(1, 1, self.d_model)
        x2 = torch.zeros(1, 1, self.d_model)
        
        output1 = self.pos_enc(x1)
        output2 = self.pos_enc(x2)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_different_lengths(self):
        """测试不同长度的序列"""
        for seq_len in [1, 16, 32, 64]:
            x = torch.randn(1, seq_len, self.d_model)
            output = self.pos_enc(x)
            self.assertEqual(output.shape[1], seq_len)


class TestCausalMask(unittest.TestCase):
    """测试 causal mask 生成"""
    
    def test_causal_mask_shape(self):
        """测试 causal mask 的形状"""
        sz = 32
        mask = generate_causal_mask(sz)
        self.assertEqual(mask.shape, (sz, sz))
    
    def test_causal_mask_values(self):
        """测试 causal mask 的值"""
        sz = 4
        mask = generate_causal_mask(sz)
        
        # 上三角形应该是 -inf
        for i in range(sz):
            for j in range(i+1, sz):
                self.assertTrue(torch.isinf(mask[i, j]))
        
        # 下三角形和对角线应该是 0
        for i in range(sz):
            for j in range(i+1):
                self.assertEqual(mask[i, j].item(), 0.0)
    
    def test_causal_mask_device(self):
        """测试 causal mask 的设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            mask = generate_causal_mask(8, device=device)
            self.assertEqual(mask.device.type, 'cuda')


class TestTopKTopPFiltering(unittest.TestCase):
    """测试 top-k 和 top-p 采样过滤"""
    
    def test_top_k_filtering(self):
        """测试 top-k 过滤"""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        filtered = top_k_top_p_filtering(logits, top_k=3)
        
        # 应该保留最高的3个值
        self.assertTrue(torch.isfinite(filtered[1]))  # 5.0
        self.assertTrue(torch.isfinite(filtered[4]))  # 4.0
        self.assertTrue(torch.isfinite(filtered[2]))  # 3.0
        self.assertTrue(torch.isinf(filtered[0]))     # 1.0
        self.assertTrue(torch.isinf(filtered[3]))     # 2.0
    
    def test_top_p_filtering(self):
        """测试 top-p 过滤"""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], dtype=torch.float32)
        filtered = top_k_top_p_filtering(logits, top_p=0.5)
        
        # 应该保留累积概率不超过 50% 的最高值
        finite_count = torch.isfinite(filtered).sum().item()
        self.assertGreater(finite_count, 0)
        self.assertLess(finite_count, 5)
    
    def test_combined_filtering(self):
        """测试结合 top-k 和 top-p"""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], dtype=torch.float32)
        filtered = top_k_top_p_filtering(logits, top_k=3, top_p=0.5)
        
        finite_count = torch.isfinite(filtered).sum().item()
        self.assertGreater(finite_count, 0)


class TestSampleFromLogits(unittest.TestCase):
    """测试从 logits 采样"""
    
    def test_sample_from_logits(self):
        """测试基本采样"""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], dtype=torch.float32)
        
        for _ in range(10):
            sample = sample_from_logits(logits, temperature=1.0)
            self.assertIsInstance(sample, int)
            self.assertGreaterEqual(sample, 0)
            self.assertLess(sample, len(logits))
    
    def test_temperature_effect(self):
        """测试温度参数的效果"""
        logits = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32)
        
        # 温度应该大于0
        with self.assertRaises(ValueError):
            sample_from_logits(logits, temperature=0.0)
        
        with self.assertRaises(ValueError):
            sample_from_logits(logits, temperature=-1.0)
    
    def test_high_temperature(self):
        """测试高温度下的采样"""
        logits = torch.tensor([0.0, 100.0, 0.0], dtype=torch.float32)
        
        # 高温度应该使采样更均匀
        samples = []
        for _ in range(100):
            sample = sample_from_logits(logits, temperature=10.0)
            samples.append(sample)
        
        # 应该有多个不同的采样值
        unique_samples = len(set(samples))
        self.assertGreater(unique_samples, 1)


class TestMSEWithPositivePressure(unittest.TestCase):
    """测试带正压力的 MSE 损失"""
    
    def setUp(self):
        self.loss_fn = MSEWithPositivePressure(pressure=10.0, mean=0.0, std=1.0)
    
    def test_forward_shape(self):
        """测试前向传播的形状"""
        pred = torch.randn(32, requires_grad=True)
        target = torch.randn(32)
        
        loss = self.loss_fn(pred, target)
        
        self.assertEqual(loss.dim(), 0)  # 标量
    
    def test_positive_pressure(self):
        """测试正压力效果"""
        # 负值应该产生额外的损失
        pred_negative = torch.tensor([-1.0, -2.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0])
        
        loss_negative = self.loss_fn(pred_negative, target)
        
        # 应该有正压力项
        self.assertGreater(loss_negative.item(), 0.0)
    
    def test_no_pressure_for_positive(self):
        """测试正值不产生额外压力"""
        pred_positive = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0])
        
        loss_positive = self.loss_fn(pred_positive, target)
        
        # 损失应该主要是 MSE
        self.assertGreater(loss_positive.item(), 0.0)
    
    def test_different_pressures(self):
        """测试不同的压力值"""
        pred = torch.tensor([-1.0, -2.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0])
        
        loss_fn_high = MSEWithPositivePressure(pressure=100.0, mean=0.0, std=1.0)
        loss_fn_low = MSEWithPositivePressure(pressure=1.0, mean=0.0, std=1.0)
        
        loss_high = loss_fn_high(pred, target)
        loss_low = loss_fn_low(pred, target)
        
        self.assertGreater(loss_high.item(), loss_low.item())


class TestImprovedMidiSequenceDataset(unittest.TestCase):
    """测试 MIDI 序列数据集"""
    
    def setUp(self):
        # 创建模拟的 MIDI 输入数据
        self.midi_inputs = [
            [60.0, 0.1, 0.5, 0.0, 1.0],  # [pitch, step, duration, beat_pos, beat_strength]
            [64.0, 0.1, 0.5, 0.25, 0.7],
            [67.0, 0.1, 0.5, 0.5, 0.3],
            [72.0, 0.1, 0.5, 0.75, 0.3],
        ] * 10  # 重复以获得足够的数据
    
    def test_dataset_init(self):
        """测试数据集初始化"""
        dataset = ImprovedMidiSequenceDataset(self.midi_inputs)
        
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.window, 32 + 1)  # seq_length + 1
    
    def test_dataset_getitem(self):
        """测试获取单个数据"""
        dataset = ImprovedMidiSequenceDataset(self.midi_inputs)
        
        if len(dataset) > 0:
            (pitch_inputs, durs_in, rhythm_features), targets = dataset[0]
            
            self.assertEqual(pitch_inputs.dtype, torch.long)
            self.assertEqual(durs_in.dtype, torch.float32)
            self.assertEqual(rhythm_features.dtype, torch.float32)
            
            self.assertEqual(rhythm_features.shape[-1], 2)  # beat_pos, beat_strength
    
    def test_normalization(self):
        """测试数据标准化"""
        dataset_normalized = ImprovedMidiSequenceDataset(self.midi_inputs, normalize=True)
        dataset_unnormalized = ImprovedMidiSequenceDataset(self.midi_inputs, normalize=False)
        
        self.assertNotEqual(dataset_normalized.step_std, dataset_unnormalized.step_std)
    
    def test_statistics_computation(self):
        """测试统计量计算"""
        dataset = ImprovedMidiSequenceDataset(self.midi_inputs)
        
        self.assertGreater(dataset.step_mean, 0.0)
        self.assertGreater(dataset.duration_mean, 0.0)
        self.assertGreater(dataset.step_std, 0.0)
        self.assertGreater(dataset.duration_std, 0.0)


class TestRhythmAwareCachedTransformerLayer(unittest.TestCase):
    """测试节奏感知的 Transformer 层"""
    
    def setUp(self):
        self.d_model = 128
        self.nhead = 8
        self.layer = RhythmAwareCachedTransformerLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=256,
            dropout=0.1
        )
    
    def test_forward_shape(self):
        """测试前向传播的形状"""
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, self.d_model)
        output, k, v = self.layer(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_forward_with_rhythm_weights(self):
        """测试带节奏权重的前向传播"""
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, self.d_model)
        rhythm_weights = torch.rand(batch_size, seq_len)
        
        output, k, v = self.layer(x, rhythm_weights=rhythm_weights)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_forward_with_causal_mask(self):
        """测试带因果掩码的前向传播"""
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, self.d_model)
        causal_mask = generate_causal_mask(seq_len)
        
        output, k, v = self.layer(x, causal_mask=causal_mask)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_forward_step(self):
        """测试增量解码步"""
        batch_size = 2
        
        x_new = torch.randn(batch_size, 1, self.d_model)
        
        output, k, v = self.layer.forward_step(x_new, None, None, None)
        
        self.assertEqual(output.shape, x_new.shape)
        self.assertIsNotNone(k)
        self.assertIsNotNone(v)


class TestImprovedAutoregressiveTransformer(unittest.TestCase):
    """测试改进的自回归 Transformer"""
    
    def setUp(self):
        self.model = ImprovedAutoregressiveTransformer(
            vocab_size=128,
            embed_dim=64,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            duration_buckets=16
        )
    
    def test_model_init(self):
        """测试模型初始化"""
        self.assertEqual(self.model.vocab_size, 128)
        self.assertEqual(self.model.d_model, 128)
    
    def test_forward_shape(self):
        """测试前向传播的形状"""
        batch_size = 2
        seq_len = 32
        
        pitch_idxs = torch.randint(0, 128, (batch_size, seq_len))
        durations = torch.rand(batch_size, seq_len) * 4.0
        rhythm_features = torch.rand(batch_size, seq_len, 2)
        
        outputs, kv_cache = self.model(pitch_idxs, durations, rhythm_features)
        
        self.assertEqual(outputs['pitch'].shape, (batch_size, seq_len, 128))
        self.assertEqual(outputs['step'].shape, (batch_size, seq_len))
        self.assertEqual(outputs['duration'].shape, (batch_size, seq_len))
    
    def test_init_kv_cache(self):
        """测试 KV 缓存初始化"""
        batch_size = 1
        seq_len = 32
        
        pitch_idxs = torch.randint(0, 128, (batch_size, seq_len))
        durations = torch.rand(batch_size, seq_len) * 4.0
        rhythm_features = torch.rand(batch_size, seq_len, 2)
        
        kv_cache = self.model.init_kv_cache(pitch_idxs, durations, rhythm_features)
        
        self.assertIsInstance(kv_cache, list)
        self.assertEqual(len(kv_cache), 2)  # num_layers = 2
    
    def test_generate_step_with_cache(self):
        """测试使用缓存的生成步"""
        batch_size = 1
        
        # 初始化缓存
        pitch_seed = torch.randint(0, 128, (batch_size, 32))
        durations_seed = torch.rand(batch_size, 32) * 4.0
        rhythm_seed = torch.rand(batch_size, 32, 2)
        kv_cache = self.model.init_kv_cache(pitch_seed, durations_seed, rhythm_seed)
        
        # 生成一步
        last_pitch = torch.tensor([60], dtype=torch.long)
        last_duration = torch.tensor([0.5], dtype=torch.float32)
        last_rhythm = torch.tensor([[0.5, 0.7]], dtype=torch.float32)
        
        pitch_logits, step_pred, dur_pred, new_kv = self.model.generate_step_with_cache(
            last_pitch, last_duration, last_rhythm, 0, kv_cache
        )
        
        self.assertEqual(pitch_logits.shape, (batch_size, 128))
        self.assertEqual(step_pred.shape, (batch_size,))
        self.assertEqual(dur_pred.shape, (batch_size,))
        self.assertEqual(len(new_kv), 2)
    
    def test_model_device(self):
        """测试模型设备转换"""
        device = torch.device('cpu')
        self.model.to(device)
        
        # 所有参数应该在指定设备上
        for param in self.model.parameters():
            self.assertEqual(param.device.type, device.type)
    
    def test_model_training_mode(self):
        """测试模型训练/评估模式"""
        self.model.train()
        self.assertTrue(self.model.training)
        
        self.model.eval()
        self.assertFalse(self.model.training)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_rhythm_analyzer_full_pipeline(self):
        """测试 RhythmAnalyzer 的完整流程"""
        analyzer = RhythmAnalyzer(tempo=120.0)
        
        # 创建模拟的音符
        notes = [Note(start=i*0.5, end=i*0.5+0.25, pitch=60+i) for i in range(8)]
        
        # 分析拍号
        time_sig = analyzer.analyze_time_signature(notes)
        self.assertIsInstance(time_sig, tuple)
        
        # 分析每个音符的节奏特征
        for note in notes:
            beat_pos = analyzer.extract_beat_position(note.start, grid=16)
            quantized = analyzer.quantize_to_beat(note.start, grid=16)
            beat_strength = analyzer.extract_beat_strength(quantized, grid=16)
            
            self.assertGreaterEqual(beat_pos, 0.0)
            self.assertLess(beat_pos, 1.0)
            self.assertGreaterEqual(beat_strength, 0.3)
            self.assertLessEqual(beat_strength, 1.0)
    
    def test_model_forward_backward(self):
        """测试模型的前向和反向传播"""
        model = ImprovedAutoregressiveTransformer(
            vocab_size=128,
            embed_dim=64,
            d_model=128,
            nhead=8,
            num_layers=1,
            dim_feedforward=256,
            dropout=0.1
        )
        
        batch_size = 2
        seq_len = 16
        
        pitch_idxs = torch.randint(0, 128, (batch_size, seq_len))
        durations = torch.rand(batch_size, seq_len) * 4.0
        rhythm_features = torch.rand(batch_size, seq_len, 2)
        
        outputs, _ = model(pitch_idxs, durations, rhythm_features)
        
        # 计算损失
        loss = outputs['pitch'].sum() + outputs['step'].sum() + outputs['duration'].sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestEdgeCases(unittest.TestCase):
    """边界情况和错误处理测试"""
    
    def test_rhythm_analyzer_with_none_input(self):
        """测试 None 输入"""
        analyzer = RhythmAnalyzer()
        
        # 应该返回默认拍号
        result = analyzer.analyze_time_signature(None)
        self.assertEqual(result, (4, 4))
    
    def test_dataset_with_small_input(self):
        """测试数据集的小输入"""
        small_midi = [[60.0, 0.1, 0.5, 0.0, 1.0]]
        
        # 应该处理或跳过太小的输入
        dataset = ImprovedMidiSequenceDataset(small_midi)
        self.assertEqual(len(dataset), 0)  # 不足以形成窗口
    
    def test_embedding_with_zero_duration(self):
        """测试零 duration 的 embedding"""
        embedding = JointPitchDurationEmbedding(128, 64)
        
        pitch = torch.tensor([[60]], dtype=torch.long)
        duration = torch.tensor([[0.0]], dtype=torch.float32)
        
        output = embedding(pitch, duration)
        self.assertEqual(output.shape, (1, 1, 64))
    
    def test_model_with_single_element_batch(self):
        """测试单元素 batch"""
        model = ImprovedAutoregressiveTransformer(
            vocab_size=128,
            embed_dim=64,
            d_model=128,
            nhead=8,
            num_layers=1,
            dim_feedforward=256
        )
        
        pitch = torch.randint(0, 128, (1, 1))
        duration = torch.rand(1, 1)
        rhythm = torch.rand(1, 1, 2)
        
        outputs, _ = model(pitch, duration, rhythm)
        
        self.assertEqual(outputs['pitch'].shape[0], 1)


class TestRandomness(unittest.TestCase):
    """随机性测试"""
    
    def test_sampling_randomness(self):
        """测试采样的随机性"""
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        
        samples = [sample_from_logits(logits) for _ in range(100)]
        
        # 应该有多个不同的采样值
        unique_samples = len(set(samples))
        self.assertGreater(unique_samples, 1)
    
    def test_model_parameter_initialization(self):
        """测试模型参数初始化"""
        model1 = ImprovedAutoregressiveTransformer(
            vocab_size=128,
            embed_dim=64,
            d_model=128,
            nhead=8,
            num_layers=1
        )
        model2 = ImprovedAutoregressiveTransformer(
            vocab_size=128,
            embed_dim=64,
            d_model=128,
            nhead=8,
            num_layers=1
        )
        
        # 两个模型的参数应该不同（随机初始化）
        different_params = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2):
                different_params = True
                break
        
        self.assertTrue(different_params)


if __name__ == '__main__':
    unittest.main()
