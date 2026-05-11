# Refiner：基于 Decoder Hidden 的非因果语义纠错模块

## 1. 方法动机

Wenet 解码器采用自回归因果建模方式，在第 (t) 个位置仅依赖历史上下文与声学特征进行预测，其本质可表示为：

[
P(y_t \mid y_{<t}, X)
]

其中，(X) 表示声学输入，(y_{<t}) 表示当前位置之前的文本历史。该建模方式保证了解码过程的可实现性，但在构音障碍语音识别场景下，局部发音异常容易导致早期 token 预测错误，并通过自回归链条向后传播，进而造成句级语义偏移。由于因果解码无法利用未来上下文进行纠偏，因此仅依赖原始 decoder 往往难以充分恢复全局语义一致性。

为此，本文在因果 decoder 之后引入一个**非因果语义纠错模块 Refiner**。该模块不改变原有 CTC 与自回归解码结构，而是以 decoder 最后一层隐藏表示为输入，通过双向上下文建模对整句进行语义修正。与外部语言模型蒸馏不同，Refiner 直接作用于 Wenet decoder 的隐藏状态空间，避免了 teacher-student 条件空间不一致所带来的优化冲突，更适合构音障碍语音识别中“声学失真显著、但语言结构相对稳定”的任务特点。

---

## 2. 总体结构

Refiner 的输入不是文本 token 本身，也不是词表概率分布，而是 Wenet decoder 在输出层前的隐藏表示。设 decoder 最终隐藏表示为：

[
H_{dec} \in \mathbb{R}^{B \times (L+1) \times D}
]

其中，(B) 为 batch size，(L) 为目标文本长度，额外的 1 对应 `<sos>` 位置，(D) 为隐藏维度。实际训练时，为与原始文本序列对齐，去除 `<sos>` 位置后得到：

[
H = H_{dec}[:, 1:L+1, :] \in \mathbb{R}^{B \times L \times D}
]

Refiner 以该隐藏表示为输入，经过若干层非因果 Transformer Encoder，对每个位置同时建模左右上下文，输出精炼后的隐藏表示，再映射到词表空间得到最终预测分布：

[
H_{ref} = \mathrm{Refiner}(H)
]

[
P_{ref} = \mathrm{Softmax}(W_o H_{ref})
]

其中，(W_o) 为输出映射矩阵。由于 Refiner 采用 full self-attention，不受因果 mask 约束，因此每一位置的预测均可利用全句上下文，从而实现对局部错误的全局语义修正。

---

## 3. 直接文本预测任务

Refiner 的第一项任务是基于 decoder hidden 直接恢复真实文本序列。设真实标签为：

[
Y = (y_1, y_2, \dots, y_L)
]

则 Refiner 的基本优化目标为：

[
L_{ref} = \mathrm{CE}(P_{ref}, Y)
]

该损失直接约束 Refiner 学习从“因果解码隐藏表示”到“正确文本序列”的映射关系。由于输入隐藏表示已经包含声学信息与历史上下文信息，Refiner 学到的并不是纯文本语言模型，而是一种**基于声学-文本联合表示的非因果语义重分配能力**。其作用在于对因果 decoder 已经产生的局部偏差进行句级修正，使输出更加符合整句语义一致性。

---

## 4. Hidden-Level MLM 辅助任务

仅使用直接文本预测任务时，Refiner 有可能倾向于学习“恒等映射”，即简单地沿用 decoder hidden 中已有的信息，而对局部错误位置的纠偏能力不足。为增强 Refiner 的补全与恢复能力，本文进一步引入 hidden-level MLM 辅助任务。

具体而言，在输入隐藏序列 (H) 上随机选取一部分有效位置进行掩码。设掩码位置集合为 (\mathcal{M})，则对被选中的隐藏向量用一个可学习的 mask 向量 (e_{mask}) 进行替换，得到掩码后的输入：

[
\tilde{H}*i =
\begin{cases}
e*{mask}, & i \in \mathcal{M} \
H_i, & i \notin \mathcal{M}
\end{cases}
]

将掩码后的隐藏表示送入 Refiner，得到对应的输出分布 (P_{mlm})。仅在被 mask 的位置上计算交叉熵损失：

[
L_{mlm} = \mathrm{CE}(P_{mlm}, Y_{mask})
]

其中，(Y_{mask}) 表示仅保留 mask 位置标签、其余位置忽略的监督信号。该任务的目标是迫使模型利用左右上下文恢复被遮挡位置的正确 token，从而显式增强非因果补全能力。

需要说明的是，这里的掩码对象不是 token id，而是 decoder hidden，因此该机制本质上属于**hidden-level masked reconstruction**，而不是传统的 token-level BERT MLM。相比直接对文本 token 建模，它更贴合当前任务，因为输入空间与主模型完全一致，且保留了原始 decoder hidden 中的声学信息。

---

## 5. 损失函数设计

Refiner 模块与主 ASR 模型联合训练时，整体目标函数写为：

[
L = L_{ctc} + L_{att} + \alpha L_{ref} + \beta L_{mlm}
]

其中，(L_{ctc}) 为 CTC 损失，(L_{att}) 为原始自回归解码损失，(L_{ref}) 为 Refiner 的直接文本预测损失，(L_{mlm}) 为 hidden-level MLM 辅助损失，(\alpha) 与 (\beta) 为相应权重系数。

在实际训练中，推荐先仅引入 (L_{ref}) 验证非因果纠错模块本身是否有效，在确认 CER/WER 有改善后，再加入 (L_{mlm}) 作为补充监督，以避免在初期引入过多优化因素影响主任务收敛。

---

## 6. 训练阶段流程

训练阶段的执行过程如下：

首先，输入语音经过 encoder 得到声学表示；随后因果 decoder 在 teacher forcing 条件下生成文本对应的隐藏表示 (H_{dec}) 与原始解码输出。然后取去除 `<sos>` 后的隐藏表示 (H) 作为 Refiner 输入，一方面直接通过 (L_{ref}) 学习 hidden 到真实文本的映射，另一方面在随机掩码后通过 (L_{mlm}) 学习局部缺失恢复能力。最终 Refiner 的损失与原有 CTC、Attention 损失共同反向传播，从而实现因果声学建模与非因果语义建模的联合优化。

---

## 7. 推理阶段流程

推理阶段保持 Wenet 原有的解码流程不变，仍可采用 CTC prefix beam search 与 attention rescoring 进行初步预测。设经过 attention rescoring 后选出的最优候选为 `best_hyp`，则将其重新组织为带 `<sos>` 的输入序列，再送入原有 **因果 decoder** 得到对应的 decoder hidden。随后去除 `<sos>` 位置，将对齐后的 hidden 输入 Refiner，得到最终修正后的文本输出。

因此，推理阶段的流程可以概括为：

```text
CTC prefix beam search
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

需要强调的是，Refiner 输入的不是 `best_hyp` token 序列本身，而是由该候选序列重新经过 **Wenet 因果 decoder** 产生的隐藏表示。这样可以确保 Refiner 的输入仍然保留声学上下文信息，而不是退化为纯文本纠错。

---

## 8. 方法特点与优势

与此前基于 BERT 的 KL 后监督方案相比，Refiner 具有以下优势。首先，Refiner 直接作用于 decoder hidden，与主模型处于同一表示空间，不存在 teacher 与 student 条件空间不一致的问题，因此训练更稳定。其次，Refiner 保留了 encoder-decoder cross-attention 引入的声学信息，不会像纯文本语言模型那样仅依赖语言统计进行猜测，更适合构音障碍语音中声学异常显著的场景。再次，Refiner 通过 full attention 在整句范围内进行信息交互，能够有效弥补因果 decoder 无法利用未来上下文进行纠偏的缺陷。最后，Refiner 采用恒等长度建模，不涉及插入和删除操作，实现简单，便于与现有 attention rescoring 框架兼容。

---

## 9. 推荐实现配置

在实现中，Refiner 建议采用 2~4 层 Transformer Encoder 作为主干结构，隐藏维度与 decoder hidden 一致，推荐设为 512，注意力头数可取 4 或 8，前馈层维度可设置为 2048，dropout 建议为 0.1。对于 hidden-level MLM，掩码比例可设为 0.05~0.10，并保证每条样本至少 mask 一个有效位置。mask 向量建议采用可学习参数，而非随机噪声，以便模型稳定学习“该位置为缺失信息”的统一语义标记。

损失权重方面，可首先设置 (\alpha=0.2)，(\beta=0)，待 Refiner 主任务验证有效后，再将 (\beta) 设为 0.05 进行增强训练。

---

## 10. 方法本质总结

Refiner 的本质并不是一个独立语言模型，也不是外部知识蒸馏教师，而是一个建立在 Wenet decoder hidden 之上的**非因果语义纠错模块**。它通过对因果 decoder 隐藏表示进行整句级非因果重建与补全，使模型在保持原有解码框架不变的前提下，获得更强的句级语义一致性建模能力，从而提高构音障碍语音识别在复杂发音条件下的鲁棒性与文本可读性。
