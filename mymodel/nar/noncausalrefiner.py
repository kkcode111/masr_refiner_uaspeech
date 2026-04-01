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

        # 输入投影 - 始终使用 Linear 以确保类型转换为 float
        self.input_proj = nn.Linear(input_dim, hidden_dim)

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
        对 hidden 的部分位置进行 错误部分的mask
        hidden: [B, L, D]
        lengths: [B]
        """
        B, L, D = hidden.shape
        device = hidden.device

        padding_mask = self._build_padding_mask(lengths, L).to(device)   # True表示padding
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
        predict: torch.Tensor,
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
        target = target.long()
        result = {
            "logits_ref": None,
            "logits_mlm": None,
            "loss_ref": torch.tensor(0.0, device=hidden.device),
            "loss_mlm": torch.tensor(0.0, device=hidden.device),
        }

        B, L, _ = hidden.shape
        padding_mask = self._build_padding_mask(lengths, L).to(hidden.device)   # [B, L]
        # print(hidden.dtype)
        # print(target.dtype)
        # ========== 任务1：直接 hidden -> text ==========
        if compute_ref_loss:
            # 确保 hidden 是 float 类型
            hidden = hidden.float()
            x = self.input_proj(hidden)               # [B, L, D]
            x = self.refiner(x, src_key_padding_mask=padding_mask).to(hidden.device)
            x = self.final_norm(x)
            logits_ref = self.output_layer(x)         # [B, L, V]
            result["logits_ref"] = logits_ref

            loss_ref = F.cross_entropy(
                logits_ref.reshape(-1, self.vocab_size).float(),
                target.reshape(-1).long(),
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
         #=================任务3 只进行错误预测的修正
        error_mask = (predict != target) & ~padding_mask  # 只在非padding的错误位置

        # 对错误位置的 hidden 进行 mask
        hidden_error_masked = hidden.clone()
        hidden_error_masked[error_mask] = self.mask_embed.unsqueeze(0)

        # 通过 refiner 预测
        x_error = self.input_proj(hidden_error_masked)
        x_error = self.refiner(x_error, src_key_padding_mask=padding_mask)
        x_error = self.final_norm(x_error)
        logits_error = self.output_layer(x_error)

        # 只在错误位置计算 loss
        error_labels = target.clone()
        error_labels[~error_mask] = -100  # 非错误位置不计算loss

        loss_error = F.cross_entropy(
            logits_error.reshape(-1, self.vocab_size),
            error_labels.reshape(-1),
            ignore_index=-100
        )
        result["loss_error"] = loss_error
        result["logits_error"] = logits_error
        result["error_mask"] = error_mask

        return result

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor, lengths: torch.Tensor):
        """
        推理时用：直接输出修正后的 token ids
        """
        B, L, _ = hidden.shape
        padding_mask = self._build_padding_mask(lengths, L).to(hidden.device)

        x = self.input_proj(hidden)
        x = self.refiner(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        logits = self.output_layer(x)         # [B, L, V]
        pred = torch.argmax(logits, dim=-1)   # [B, L]
        return pred, logits