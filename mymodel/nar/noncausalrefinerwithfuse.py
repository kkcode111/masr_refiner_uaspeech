import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


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
        enable_span_mlm: bool = False,
        span_max_len: int = 3,
        span_max_ratio: float = 0.15,
        span_min_error_tokens: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.error_weight = error_weight
        self.use_acoustic_preprocess = use_acoustic_preprocess
        self.enable_span_mlm = enable_span_mlm
        self.span_max_len = span_max_len
        self.span_max_ratio = span_max_ratio
        self.span_min_error_tokens = span_min_error_tokens

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

        if self.enable_span_mlm:
            self.semantic_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            aux_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.aux_refiner = nn.TransformerEncoder(aux_layer, num_layers=num_layers)
            self.aux_norm = nn.LayerNorm(hidden_dim)
            self.aux_output_layer = nn.Linear(hidden_dim, vocab_size)

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

    def _extract_contiguous_spans(self, mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
        idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return []
        if idx.numel() == 1:
            pos = int(idx.item())
            return [(pos, pos)]
        diff = idx[1:] - idx[:-1]
        breaks = torch.nonzero(diff > 1, as_tuple=False).squeeze(1)
        starts = torch.cat([idx[:1], idx[breaks + 1]])
        ends = torch.cat([idx[breaks], idx[-1:]])
        return [(int(s.item()), int(e.item())) for s, e in zip(starts, ends)]

    def _build_span_mask(
        self,
        error_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, L = error_mask.shape
        device = error_mask.device
        span_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        valid_for_mask = (~padding_mask) & (target != -100)

        for b in range(B):
            valid_errors = error_mask[b] & valid_for_mask[b]
            if valid_errors.sum().item() < int(self.span_min_error_tokens):
                continue
            spans = self._extract_contiguous_spans(valid_errors)
            if len(spans) == 0:
                continue
            positions: List[torch.Tensor] = []
            for start, end in spans:
                if self.span_max_len is not None and int(self.span_max_len) > 0:
                    end = min(end, start + int(self.span_max_len) - 1)
                if end >= start:
                    positions.append(torch.arange(start, end + 1, device=device, dtype=torch.long))
            if len(positions) == 0:
                continue
            pos = torch.cat(positions, dim=0)

            if self.span_max_ratio is not None and float(self.span_max_ratio) > 0.0:
                budget = int(max(1, int(lengths[b].item() * float(self.span_max_ratio))))
                if pos.numel() > budget:
                    pos = pos[:budget]
            span_mask[b, pos] = True

        span_mask = span_mask & valid_for_mask
        return span_mask

    def _build_masked_hidden(self, x: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        if span_mask.sum().item() == 0:
            return x
        x_masked = x.clone()
        mask_vec = self.mask_embed.to(device=x.device, dtype=x.dtype)
        x_masked[span_mask] = mask_vec
        return x_masked

    def _compute_span_mlm_loss(self, logits: torch.Tensor, target: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        effective_mask = span_mask & (target != -100)
        if effective_mask.sum().item() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        labels = target.clone()
        labels[~effective_mask] = -100
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )

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
        compute_mlm_loss: bool = True,
    ):
        target = target.long()
        predict = predict.long()

        result = {
            "logits_ref": None,
            "loss_ref": torch.tensor(0.0, device=hidden.device),
            "loss_error": torch.tensor(0.0, device=hidden.device),
            "loss_mlm": torch.tensor(0.0, device=hidden.device),
            "error_mask": None,
            "span_mask": None,
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

        if self.enable_span_mlm and compute_mlm_loss:
            error_mask_for_span = (predict != target) & (~padding_mask) & (target != -100)
            span_mask = self._build_span_mask(error_mask_for_span, padding_mask, target, lengths)
            result["span_mask"] = span_mask

            x0 = self._prepare_hidden(hidden, encoder_out=encoder_out, encoder_mask=encoder_mask).detach()
            x_sem = x0 + self.semantic_proj(x0)
            x_mlm = self._build_masked_hidden(x_sem, span_mask)
            x_mlm = self.aux_refiner(x_mlm, src_key_padding_mask=padding_mask)
            x_mlm = self.aux_norm(x_mlm)
            logits_mlm = self.aux_output_layer(x_mlm)
            result["loss_mlm"] = self._compute_span_mlm_loss(logits_mlm, target, span_mask)

        return result

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
        
        x_main = self.refiner(x, src_key_padding_mask=padding_mask)
        x_main = self.final_norm(x_main)
        logits_main = self.output_layer(x_main)
        
        if self.enable_span_mlm:
            x_sem = x + self.semantic_proj(x)
            x_aux = self.aux_refiner(x_sem, src_key_padding_mask=padding_mask)
            x_aux = self.aux_norm(x_aux)
            logits_aux = self.aux_output_layer(x_aux)
            logits = logits_main + 0.5 * logits_aux
        else:
            logits = logits_main
            
        pred = torch.argmax(logits, dim=-1)
        return pred, logits
