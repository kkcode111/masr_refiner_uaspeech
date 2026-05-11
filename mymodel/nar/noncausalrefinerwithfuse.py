import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticPreprocessor(nn.Module):
    """
    使用 decoder hidden 作为 query，对 encoder_out 做 cross-attention，
    得到 token 对齐的声学补偿表示。
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_gate = use_gate

        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_gate:
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.alpha = nn.Parameter(torch.tensor(0.5))

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def _convert_encoder_mask(self, encoder_mask):
        """
        转为 nn.MultiheadAttention 所需的 key_padding_mask:
        [B, T], True 表示 padding
        """
        if encoder_mask is None:
            return None

        if encoder_mask.dim() == 3:
            # Wenet风格 [B, 1, T], True表示有效位置
            key_padding_mask = ~encoder_mask.squeeze(1)
        elif encoder_mask.dim() == 2:
            # [B, T], True表示有效位置
            key_padding_mask = ~encoder_mask
        else:
            raise ValueError(f"Unsupported encoder_mask shape: {encoder_mask.shape}")

        return key_padding_mask

    def forward(self, hidden, encoder_out, encoder_mask=None):
        """
        hidden:      [B, L, D]
        encoder_out: [B, T, D]
        encoder_mask:[B, 1, T] 或 [B, T]
        """
        key_padding_mask = self._convert_encoder_mask(encoder_mask)

        h = self.hidden_norm(hidden)
        e = self.encoder_norm(encoder_out)

        attn_out, _ = self.cross_attn(
            query=h,
            key=e,
            value=e,
            key_padding_mask=key_padding_mask
        )  # [B, L, D]

        if self.use_gate:
            gate = self.gate_proj(torch.cat([h, attn_out], dim=-1))
            fused = hidden + gate * attn_out
        else:
            fused = hidden + self.alpha * attn_out

        fused = fused + self.ffn(self.out_norm(fused))
        return fused


class NonCausalRefiner(nn.Module):
    """
    非因果纠错 Refiner（去掉 MLM 版本）
    输入:
        hidden: [B, L, 512]    (来自 Wenet decoder hidden，已去掉<sos>)
        target: [B, L]
        lengths:[B]
        predict:[B, L]         (原始 decoder / rescoring 的预测结果)
        encoder_out: [B, T, 512] 可选
        encoder_mask:[B,1,T] 或 [B,T] 可选

    输出:
        logits_ref
        loss_ref
        loss_error
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        error_weight: float = 2.0,
        use_acoustic_preprocess: bool = True,
        acoustic_heads: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.error_weight = error_weight
        self.use_acoustic_preprocess = use_acoustic_preprocess

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 用于错误位置替换的可学习 mask 向量
        self.mask_embed = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_embed, mean=0.0, std=0.02)

        # 声学补偿预处理
        if use_acoustic_preprocess:
            self.acoustic_preprocessor = AcousticPreprocessor(
                hidden_dim=hidden_dim,
                num_heads=acoustic_heads,
                dropout=dropout,
                use_gate=True,
            )

        # 非因果 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refiner = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def _build_padding_mask(self, lengths: torch.Tensor, max_len: int):
        """
        返回 [B, L], True 表示 padding
        """
        device = lengths.device
        seq_range = torch.arange(max_len, device=device).unsqueeze(0)
        return seq_range >= lengths.unsqueeze(1)

    def _prepare_hidden(self, hidden, encoder_out=None, encoder_mask=None):
        """
        预处理 hidden，并可选融合 encoder_out 的声学信息
        """
        hidden = hidden.float()
        x = self.input_proj(hidden)

        if self.use_acoustic_preprocess and encoder_out is not None:
            x = self.acoustic_preprocessor(x, encoder_out.float(), encoder_mask)

        return x

    def _compute_ref_loss(self, logits, target):
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target.reshape(-1),
            ignore_index=-100
        )

    def _compute_error_weighted_loss(self, logits, target, predict, padding_mask):
        """
        在全位置 CE 的基础上，对错误位置加权
        """
        error_mask = (predict != target) & (~padding_mask) & (target != -100)

        loss_per_token = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target.reshape(-1),
            ignore_index=-100,
            reduction="none"
        ).view_as(target)

        weights = torch.ones_like(loss_per_token)
        weights[error_mask] = self.error_weight
        weights[padding_mask] = 0.0
        weights[target == -100] = 0.0

        loss_error = (loss_per_token * weights).sum() / (weights.sum() + 1e-8)
        return loss_error, error_mask

    def forward(
        self,
        hidden: torch.Tensor,           # [B, L, 512]
        target: torch.Tensor,           # [B, L]
        lengths: torch.Tensor,          # [B]
        predict: torch.Tensor,          # [B, L]
        encoder_out: torch.Tensor = None,
        encoder_mask: torch.Tensor = None,
        compute_ref_loss: bool = True,
        compute_error_loss: bool = True,
    ):
        target = target.long()
        predict = predict.long()

        result = {
            "logits_ref": None,
            "loss_ref": torch.tensor(0.0, device=hidden.device),
            "loss_error": torch.tensor(0.0, device=hidden.device),
            "error_mask": None,
        }

        B, L, _ = hidden.shape
        padding_mask = self._build_padding_mask(lengths, L).to(hidden.device)

        # ===== 预处理 hidden =====
        x = self._prepare_hidden(hidden, encoder_out=encoder_out, encoder_mask=encoder_mask)

        # ===== 非因果 Refiner =====
        x = self.refiner(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        logits_ref = self.output_layer(x)    # [B, L, V]
        result["logits_ref"] = logits_ref

        # ===== 主任务：hidden -> Y =====
        if compute_ref_loss:
            loss_ref = self._compute_ref_loss(logits_ref, target)
            result["loss_ref"] = loss_ref

        # ===== 辅助任务：错误位置增强 =====
        if compute_error_loss:
            loss_error, error_mask = self._compute_error_weighted_loss(
                logits_ref, target, predict, padding_mask
            )
            result["loss_error"] = loss_error
            result["error_mask"] = error_mask
        # print(loss_error,loss_ref)
        return result
        #掩码连续词语句子上下文非因果预测

    @torch.no_grad()
    def decode(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
        encoder_out: torch.Tensor = None,
        encoder_mask: torch.Tensor = None,
    ):
        """
        推理时直接输出修正后的 token ids
        """
        B, L, _ = hidden.shape
        padding_mask = self._build_padding_mask(lengths, L).to(hidden.device)

        x = self._prepare_hidden(hidden, encoder_out=encoder_out, encoder_mask=encoder_mask)
        x = self.refiner(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        pred = torch.argmax(logits, dim=-1)
        return pred, logits