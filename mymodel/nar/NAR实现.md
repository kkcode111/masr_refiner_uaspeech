**可直接落地**的“非因果解码器（Refiner）”实现方案。它满足你现在的目标：

* 输入：`[B, L, 512]`，即 **Wenet 因果 decoder 最后一层 hidden**
* 输出：`[B, L, V]`，即 **恒等长度的文本预测分布**
* 任务 1：直接做 `hidden -> text Y`
* 任务 2：做 **hidden-level MLM**，即对 hidden 的部分位置做 mask，只预测 mask 位置的文本 id
* 非因果：内部用 **full self-attention**，不使用 causal mask

这个模块本质上不是 BERT，也不是 teacher-student 蒸馏，而是：

> **基于因果 decoder hidden 的非因果纠错/补全模块**

---

# 一、整体思路

你的主干已经有：

[
H_{dec} \in \mathbb{R}^{B \times L \times 512}
]

这里的 `H_dec` 来自 Wenet decoder 最后一层 `after_norm` 后、`output_layer` 前的 hidden。

然后新增一个非因果 Refiner：

[
H_{ref} = \mathrm{Refiner}(H_{dec})
]

输出词表分布：

[
P = \mathrm{Softmax}(W H_{ref})
]

训练时做两个任务：

## 任务 1：恒等长度文本预测

直接让 Refiner 输出对齐到真实文本 `Y`：

[
L_{ref} = CE(P, Y)
]

---

## 任务 2：hidden-level MLM

对 `H_dec` 的部分位置进行 mask，得到 `H_dec^{mask}`，再输入 Refiner，要求恢复被 mask 位置的正确 token：

[
L_{mlm} = CE(P_{mask}, Y_{mask})
]

这里只在 mask 位置计算损失。

---

# 二、为什么这个设计适合你

相比之前 BERT + KL 方案，这个设计有几个优势：

1. **输入空间一致**
   输入就是 Wenet decoder hidden，不存在 teacher/student 条件空间不匹配

2. **保留声学信息**
   hidden 已经融合了 encoder_out 的声学信息，而不是纯文本建模

3. **非因果补偿明确**
   因果 decoder 无法看未来；Refiner 可以利用整句左右信息修正局部错误

4. **推理路径清晰**
   attention rescoring 后选出 `best_hyp`，重新喂 Wenet decoder 得到 hidden，再做 Refiner 输出 final_hyp

---

# 三、模块结构建议

推荐结构：

* 输入投影：`512 -> 512` 或 `512 -> 256`
* 2~4 层 TransformerEncoder（非因果）
* LayerNorm
* 输出层：`hidden -> vocab_size`

推荐参数：

* `input_dim = 512`
* `hidden_dim = 512`
* `num_layers = 2`
* `num_heads = 4 or 8`
* `ffn_dim = 2048`
* `dropout = 0.1`

先从 **2 层** 开始最稳。

---

# 四、输入输出对齐

假设：

* decoder hidden：`decoder_hidden`，shape `[B, U+1, 512]`
* 原始文本：`ys_pad`，shape `[B, U]`

则取：

```python
hidden_text = decoder_hidden[:, 1:U+1, :]   # 去掉 sos，对齐到真实文本长度
target_text = ys_pad                        # [B, U]
```

这样 Refiner 输入和文本标签严格对齐。

---

# 五、完整 PyTorch 实现

下面给你一个可直接用的版本。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonCausalRefiner(nn.Module):
    """
    非因果解码器 / Refiner
    输入:  [B, L, 512]  (来自 Wenet decoder hidden)
    输出:  [B, L, vocab_size]
    任务:
        1) hidden -> text
        2) hidden-level MLM
    """
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        mlm_prob: float = 0.10,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlm_prob = mlm_prob

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # 可学习 mask 向量（hidden-level [MASK]）
        self.mask_embed = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_embed, mean=0.0, std=0.02)

        # 非因果 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm 更稳
        )
        self.refiner = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def _build_padding_mask(self, lengths: torch.Tensor, max_len: int):
        """
        返回 True 表示 padding，需要被 mask 掉
        shape: [B, L]
        """
        device = lengths.device
        seq_range = torch.arange(max_len, device=device).unsqueeze(0)  # [1, L]
        return seq_range >= lengths.unsqueeze(1)  # [B, L]

    def _apply_hidden_mlm_mask(self, hidden: torch.Tensor, lengths: torch.Tensor):
        """
        对 hidden 的部分位置进行 mask
        hidden: [B, L, D]
        lengths: [B]
        """
        B, L, D = hidden.shape
        device = hidden.device

        padding_mask = self._build_padding_mask(lengths, L)   # True表示padding
        valid_mask = ~padding_mask

        # 采样 MLM 位置
        prob = torch.full((B, L), self.mlm_prob, device=device)
        masked_indices = torch.bernoulli(prob).bool() & valid_mask

        # 保证每条样本至少 mask 一个有效位置（可选但推荐）
        for b in range(B):
            if masked_indices[b].sum() == 0:
                valid_pos = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(1)
                if len(valid_pos) > 0:
                    rand_idx = valid_pos[torch.randint(len(valid_pos), (1,), device=device)]
                    masked_indices[b, rand_idx] = True

        hidden_masked = hidden.clone()
        hidden_masked[masked_indices] = self.mask_embed.unsqueeze(0)

        return hidden_masked, masked_indices

    def forward(
        self,
        hidden: torch.Tensor,         # [B, L, 512]
        target: torch.Tensor,         # [B, L]
        lengths: torch.Tensor,        # [B]
        compute_ref_loss: bool = True,
        compute_mlm_loss: bool = True,
    ):
        """
        返回:
            logits_ref: [B, L, V]   直接文本预测
            logits_mlm: [B, L, V]   MLM预测
            loss_ref
            loss_mlm
        """
        result = {
            "logits_ref": None,
            "logits_mlm": None,
            "loss_ref": torch.tensor(0.0, device=hidden.device),
            "loss_mlm": torch.tensor(0.0, device=hidden.device),
        }

        B, L, _ = hidden.shape
        padding_mask = self._build_padding_mask(lengths, L)   # [B, L]

        # ========== 任务1：直接 hidden -> text ==========
        if compute_ref_loss:
            x = self.input_proj(hidden)               # [B, L, D]
            x = self.refiner(x, src_key_padding_mask=padding_mask)
            x = self.final_norm(x)
            logits_ref = self.output_layer(x)         # [B, L, V]
            result["logits_ref"] = logits_ref

            loss_ref = F.cross_entropy(
                logits_ref.reshape(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=-100
            )
            result["loss_ref"] = loss_ref

        # ========== 任务2：hidden-level MLM ==========
        if compute_mlm_loss:
            hidden_masked, masked_indices = self._apply_hidden_mlm_mask(hidden, lengths)

            x_mlm = self.input_proj(hidden_masked)
            x_mlm = self.refiner(x_mlm, src_key_padding_mask=padding_mask)
            x_mlm = self.final_norm(x_mlm)
            logits_mlm = self.output_layer(x_mlm)
            result["logits_mlm"] = logits_mlm

            mlm_labels = target.clone()
            mlm_labels[~masked_indices] = -100   # 只在 mask 位置计算loss

            loss_mlm = F.cross_entropy(
                logits_mlm.reshape(-1, self.vocab_size),
                mlm_labels.reshape(-1),
                ignore_index=-100
            )
            result["loss_mlm"] = loss_mlm

        return result

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor, lengths: torch.Tensor):
        """
        推理时用：直接输出修正后的 token ids
        """
        B, L, _ = hidden.shape
        padding_mask = self._build_padding_mask(lengths, L)

        x = self.input_proj(hidden)
        x = self.refiner(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        logits = self.output_layer(x)         # [B, L, V]
        pred = torch.argmax(logits, dim=-1)   # [B, L]
        return pred, logits
```

---

# 六、训练时怎么接入 Wenet

你现在需要从 decoder 取出：

* `decoder_hidden`: `[B, U+1, 512]`
* `ys_pad`: `[B, U]`
* `ys_pad_lens`: `[B]`

然后：

```python
U = ys_pad.size(1)
hidden_text = decoder_hidden[:, 1:U+1, :]   # 去掉sos，对齐文本
target_text = ys_pad.clone()

# padding位置改成 -100，便于CE忽略
target_text[target_text == self.ignore_id] = -100

refiner_result = self.refiner(
    hidden=hidden_text,
    target=target_text,
    lengths=ys_pad_lens,
    compute_ref_loss=True,
    compute_mlm_loss=True
)

loss_ref = refiner_result["loss_ref"]
loss_mlm = refiner_result["loss_mlm"]
```

总损失建议：

```python
loss = loss_ctc + loss_att + alpha * loss_ref + beta * loss_mlm
```

推荐初始权重：

* `alpha = 0.2`
* `beta = 0.05`

先从这个量级开始，不要太大。

---

# 七、推理时怎么用

你现在仍然保留：

1. CTC prefix beam search
2. attention rescoring
3. 选出 `best_hyp`

然后：

1. 把 `best_hyp` 重新组织成 `hyps_pad, hyps_lens`
2. 再跑一遍 **Wenet 因果 decoder**
3. 拿到 `decoder_hidden`
4. 取 `decoder_hidden[:, 1:U+1, :]`
5. 输入 Refiner
6. 输出 `final_hyp`

即：

```text
CTC prefix beam
    ↓
attention rescoring
    ↓
best_hyp
    ↓
Wenet causal decoder (re-run)
    ↓
decoder hidden
    ↓
Non-causal Refiner
    ↓
final_hyp
```

---

# 八、这个方案和你之前 BERT 路线的本质区别

之前：

* BERT 输入 GT 文本
* teacher/student 条件空间不一致
* KL 对齐不稳定

现在：

* 输入就是 decoder hidden
* 任务就是 `hidden -> y`
* MLM 也是在 hidden 空间做局部遮挡恢复
* 没有 teacher/student 条件错位问题

所以这个方案更适合你。

---

# 九、推荐你现在的执行流程

先做最小可运行版本：

1. decoder 返回 hidden
2. 接上这个 Refiner
3. 训练时只开 `loss_ref`
4. 先看 CER/WER 是否改善
5. 再加入 `loss_mlm`

也就是说，一开始先不要两个任务全开，先验证：

> hidden -> y 的非因果补偿是否有效

如果这个方向有效，再加 MLM 增强鲁棒性。


