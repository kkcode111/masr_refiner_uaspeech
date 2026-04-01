下面给你一个**在原始 `attention_rescoring` 基础上接入 Refiner 的版本**。
目标是保持你现在的解码主流程不变：

```text
CTC prefix beam search
    ↓
attention rescoring
    ↓
best_hyp
    ↓
重新喂给 Wenet causal decoder 得到 hidden
    ↓
Non-causal Refiner
    ↓
final_hyp
```

也就是说：

* **原 attention rescoring 保留**
* **Refiner 作为 rescoring 后的二阶段修正**
* **不改你现有的 CTC + decoder 打分逻辑**

---

# 一、你需要新增的模型接口

因为你现在的 `get_decoder_out()` 只能返回：

```python
decoder_out, r_decoder_out
```

而 Refiner 需要的是：

```python
decoder_hidden
```

所以首先要在 model 里新增一个接口，例如：

```python
def get_decoder_hidden(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
) -> torch.Tensor:
    ...
```

这个接口逻辑和 `get_decoder_out()` 基本一致，只是最后返回 hidden。

---

# 二、建议的 decoder 改动方式

你当前 `TransformerDecoder.forward()` 里是：

```python
x, _ = self.embed(tgt)
for layer in self.decoders:
    x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
if self.normalize_before:
    x = self.after_norm(x)
if self.use_output_layer:
    x = self.output_layer(x)
return x, torch.tensor(0.0), olens
```

建议改成：

```python
x, _ = self.embed(tgt)
for layer in self.decoders:
    x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)

if self.normalize_before:
    hidden = self.after_norm(x)
else:
    hidden = x

if self.use_output_layer:
    logits = self.output_layer(hidden)
else:
    logits = hidden

return logits, hidden, olens
```

这样：

* `logits` 用于原 attention rescoring
* `hidden` 用于 Refiner

---

# 三、BiTransformerDecoder 同步改动

你现在 `BiTransformerDecoder.forward()` 是：

```python
l_x, _, olens = self.left_decoder(...)
r_x = torch.zeros([1])
if reverse_weight > 0.0:
    r_x, _, olens = self.right_decoder(...)
return l_x, r_x, olens
```

建议改成：

```python
l_x, l_hidden, olens = self.left_decoder(memory, memory_mask, ys_in_pad, ys_in_lens)
r_x = torch.zeros([1], device=memory.device)
r_hidden = torch.zeros([1], device=memory.device)

if reverse_weight > 0.0:
    r_x, r_hidden, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad, ys_in_lens)

return l_x, r_x, olens, l_hidden, r_hidden
```

---

# 四、在 model 中新增 `get_decoder_hidden()`

下面给你一个可以直接照着写的版本。

```python
@torch.jit.export
def get_decoder_hidden(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
) -> torch.Tensor:
    """
    获取 decoder output_layer 前的 hidden，用于 Refiner
    Args:
        hyps: 已经加了 sos 的候选序列, shape [num_hyps, max_len]
        hyps_lens: 每条候选长度, shape [num_hyps]
        encoder_out: 单条样本的 encoder 输出, shape [1, T, D]
        reverse_weight: 是否使用反向 decoder
    Returns:
        decoder_hidden: [num_hyps, max_len, hidden_dim]
    """
    assert encoder_out.size(0) == 1
    num_hyps = hyps.size(0)
    assert hyps_lens.size(0) == num_hyps

    encoder_out = encoder_out.repeat(num_hyps, 1, 1)
    encoder_mask = torch.ones(
        num_hyps,
        1,
        encoder_out.size(1),
        dtype=torch.bool,
        device=encoder_out.device
    )

    r_hyps_lens = hyps_lens - 1
    r_hyps = hyps[:, 1:]
    max_len = torch.max(r_hyps_lens)
    index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
    seq_len_expand = r_hyps_lens.unsqueeze(1)
    seq_mask = seq_len_expand > index_range
    index = (seq_len_expand - 1) - index_range
    index = index * seq_mask
    r_hyps = torch.gather(r_hyps, 1, index)
    r_hyps = torch.where(seq_mask, r_hyps, self.eos)
    r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)

    decoder_out, r_decoder_out, _, decoder_hidden, r_decoder_hidden = self.decoder(
        encoder_out, encoder_mask, hyps, hyps_lens, r_hyps, reverse_weight
    )

    return decoder_hidden
```

这里默认返回 **左向 decoder hidden**，因为你的 Refiner 主要接在主 decoder 后面。

---

# 五、给 Refiner 加一个推理接口

假设你已经有：

```python
self.refiner = NonCausalRefiner(...)
```

那么在 model 里建议再加一个接口：

```python
@torch.no_grad()
def refine_hyp(
        self,
        hyp: torch.Tensor,          # [1, L]
        hyp_len: torch.Tensor,      # [1]
        encoder_out: torch.Tensor,  # [1, T, D]
) -> torch.Tensor:
    """
    对单条 best_hyp 做 Refiner 修正
    """
    # 先补 sos
    hyp_in, _ = add_sos_eos(hyp, self.sos, self.eos, self.ignore_id)
    hyp_in_len = hyp_len + 1

    decoder_hidden = self.get_decoder_hidden(
        hyp_in,
        hyp_in_len,
        encoder_out,
        reverse_weight=0.0
    )   # [1, L+1, D]

    # 去掉 sos，对齐到真实文本长度
    hidden_text = decoder_hidden[:, 1:hyp_len.item()+1, :]   # [1, L, D]

    refined_pred, refined_logits = self.refiner.decode(
        hidden_text,
        hyp_len
    )

    return refined_pred[0]   # [L]
```

---

# 六、在原 attention_rescoring 基础上接入 Refiner

下面是你原函数的**增强版**。
我会只在最后选出 `best_hyp` 之后，加一步 Refiner。

---

## 改造版 `attention_rescoring_with_refiner`

```python
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from masr.decoders.ctc_prefix_beam_search import ctc_prefix_beam_search
from masr.model_utils.utils.common import add_sos_eos


def attention_rescoring_with_refiner(
        model,
        ctc_probs: torch.Tensor,
        ctc_lens: torch.Tensor,
        encoder_outs: torch.Tensor,
        encoder_lens: torch.Tensor,
        num_workers: int = 4,
        beam_size: int = 10,
        blank_id: int = 0,
        ctc_weight: float = 0.3,
        reverse_weight: float = 0.5,
        use_refiner: bool = True,
) -> List:
    """
    attention rescoring + Refiner 二阶段纠错
    """
    device = encoder_outs.device
    batch_size = encoder_outs.shape[0]
    sos, eos, ignore_id = model.sos_symbol(), model.eos_symbol(), model.ignore_symbol()

    _, hyps_list = ctc_prefix_beam_search(
        ctc_probs=ctc_probs,
        ctc_lens=ctc_lens,
        num_workers=num_workers,
        blank_id=blank_id,
        beam_size=beam_size
    )
    assert len(hyps_list[0]) == beam_size

    results = []
    raw_results = []

    for b in range(batch_size):
        hyps = hyps_list[b]
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)

        hyp_list = []
        for hyp in hyps:
            hyp_content = hyp[0]
            if len(hyp_content) == 0:
                hyp_content = (blank_id,)
            hyp_content = torch.tensor(hyp_content, device=device, dtype=torch.int64)
            hyp_list.append(hyp_content)

        hyps_pad = pad_sequence(hyp_list, True, ignore_id)
        hyps_lens = torch.tensor(
            [len(hyp[0]) if len(hyp[0]) > 0 else 1 for hyp in hyps],
            device=device,
            dtype=torch.int64
        )

        # 加 sos/eos
        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, ignore_id)
        hyps_lens = hyps_lens + 1

        # 原 attention rescoring
        decoder_out, r_decoder_out = model.get_decoder_out(
            hyps_pad, hyps_lens, encoder_out, reverse_weight
        )

        best_score = -float('inf')
        best_index = 0

        for i, hyp in enumerate(hyps):
            score = 0.0
            hyp_tokens = hyp[0]
            if len(hyp_tokens) == 0:
                hyp_tokens = (blank_id,)

            for j, w in enumerate(hyp_tokens):
                score += decoder_out[i][j][w]

            score += decoder_out[i][len(hyp_tokens)][eos]

            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp_tokens):
                    r_score += r_decoder_out[i][len(hyp_tokens) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp_tokens)][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight

            score += hyp[1] * ctc_weight

            if score > best_score:
                best_score = score
                best_index = i

        # 原 rescoring 最优结果
        best_hyp_tokens = hyps[best_index][0]
        if len(best_hyp_tokens) == 0:
            best_hyp_tokens = (blank_id,)
        raw_results.append(list(best_hyp_tokens))

        # ===== 新增：Refiner 二阶段修正 =====
        if use_refiner:
            best_hyp_tensor = torch.tensor(
                best_hyp_tokens,
                device=device,
                dtype=torch.int64
            ).unsqueeze(0)   # [1, L]
            best_hyp_len = torch.tensor(
                [len(best_hyp_tokens)],
                device=device,
                dtype=torch.int64
            )

            refined_pred = model.refine_hyp(
                best_hyp_tensor,
                best_hyp_len,
                encoder_out
            )   # [L]

            results.append(refined_pred.tolist())
        else:
            results.append(list(best_hyp_tokens))

    return results, raw_results
```

---

# 七、这个版本做了什么

这个版本返回两个结果：

* `results`：Refiner 修正后的最终输出
* `raw_results`：原 attention rescoring 的输出

这样你可以直接做对比实验：

* 只看 rescoring
* rescoring + refiner

---

# 八、你现在最推荐的实验顺序

建议你先做这三组实验：

### 1. baseline

```text
CTC prefix beam + attention rescoring
```

### 2. baseline + Refiner（只 loss_ref）

```text
CTC prefix beam + attention rescoring + Refiner
```

### 3. baseline + Refiner + hidden-level MLM

```text
CTC prefix beam + attention rescoring + Refiner(带 MLM)
```

这样你能清楚判断：

* 非因果 Refiner 本身有没有用
* MLM 是否进一步提升

---

# 九、最小总结

你要的不是替换原 `attention_rescoring()`，而是：

> 在它选出 `best_hyp` 后，再把 `best_hyp` 重新送入 Wenet 因果 decoder 得到 hidden，
> 再用非因果 Refiner 输出最终文本。

这和你原来的 rescoring 完全兼容。



