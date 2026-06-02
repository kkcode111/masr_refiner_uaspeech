import importlib
from typing import Tuple

import torch
import torch.nn.functional as F
from masr.data_utils.normalizer import FeatureNormalizer
# noinspection PyUnresolvedReferences
from masr.model_utils.transformer.decoder import *
# noinspection PyUnresolvedReferences
from masr.model_utils.conformer.encoder import *
from masr.model_utils.loss.ctc import CTCLoss
from masr.model_utils.loss.label_smoothing_loss import LabelSmoothingLoss
from masr.model_utils.utils.cmvn import GlobalCMVN
from masr.model_utils.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy, reverse_pad_list)
from masr.utils.utils import DictObject
from mymodule import PhonemeAuxiliaryLoss
# from mymodule.bertmodel import BERTLanguageModel
# from mymodule.nar.noncausalrefiner import NonCausalRefiner
from mymodule.nar.noncausalrefinerwithfuse import NonCausalRefiner
from mymodule.phonmeattribute.multiattrubuteheadmodule import MultiAttributeHeadModule

__all__ = ["ConformerModel"]


class ConformerModel(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            vocab_size: int,
            mean_istd_path: str,
            eos_id: int,
            streaming: bool = True,
            encoder_conf: DictObject = None,
            decoder_conf: DictObject = None,
            ctc_weight: float = 0.5,
            phoneme_ctc_weight: float = 0.1,
            ignore_id: int = IGNORE_ID,
            reverse_weight: float = 0.0,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            use_enhance_module:bool=False,
            use_phoneme_ctc: bool = False,
            # NonCausalRefiner 相关参数 (替换原来的BERT)
            use_refiner: bool = False,
            refiner_weight: float = 0.1,
            refiner_mlm_weight: float = 0.1,
            refiner_input_dim: int = 512,      # 输入维度，应与decoder hidden size匹配
            refiner_hidden_dim: int = 512,
            refiner_num_layers: int = 2,
            refiner_num_heads: int = 4,
            refiner_ffn_dim: int = 2048,
            refiner_dropout: float = 0.1,
            refiner_mlm_prob: float = 0.10,
     ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        super().__init__()
        self.input_size = input_size
        # 设置是否为流式模型
        self.streaming = streaming
        use_dynamic_chunk = False
        causal = False
        if self.streaming:
            use_dynamic_chunk = True
            causal = True
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
        global_cmvn = GlobalCMVN(torch.from_numpy(feature_normalizer.mean).float(),
                                 torch.from_numpy(feature_normalizer.istd).float())
        # 创建编码器和解码器
        mod = importlib.import_module(__name__)
        self.encoder = getattr(mod, encoder_conf.encoder_name)
        self.encoder = self.encoder(input_size=input_size,
                                    global_cmvn=global_cmvn,
                                    use_dynamic_chunk=use_dynamic_chunk,
                                    causal=causal,
                                    use_enhance_module=use_enhance_module,
                                    **encoder_conf.encoder_args if encoder_conf.encoder_args is not None else {})
        self.decoder = getattr(mod, decoder_conf.decoder_name)
        self.decoder = self.decoder(vocab_size=vocab_size,
                                    encoder_output_size=self.encoder.output_size(),
                                    **decoder_conf.decoder_args if decoder_conf.decoder_args is not None else {})

        self.ctc = CTCLoss(vocab_size, self.encoder.output_size(),zero_infinity= True)
        self.use_phoneme_ctc=use_phoneme_ctc
        self.phoneme_vocab_size = 62
        if use_phoneme_ctc:
            print("使用音素CTC: ", use_phoneme_ctc, " 音素CTC权重: ",phoneme_ctc_weight)
            self.phoneme_norm=nn.Sequential(
                nn.LayerNorm(self.encoder.output_size()),
                nn.Dropout(0.1)
            )
            self.phoneme_ctc =CTCLoss(self.phoneme_vocab_size, self.encoder.output_size(),zero_infinity= True)
            self.phoneme_attribute_head = MultiAttributeHeadModule(self.encoder.output_size(),"mymodule/phonmeattribute/phonmeID2attributeID.json",device="cuda")
            # self.phoneme_loss = PhonemeAuxiliaryLoss(self.encoder.output_size(),self.phoneme_vocab_size)
        # sos 和 eos 使用相同的ID
        self.sos = eos_id
        self.eos = eos_id
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.phoneme_ctc_weight=phoneme_ctc_weight
        self.reverse_weight = reverse_weight
        print("ctc_weight: ", ctc_weight, " phoneme_ctc_weight: ", phoneme_ctc_weight)

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = NonCausalRefiner(
                vocab_size=vocab_size,
                input_dim=refiner_input_dim,
                hidden_dim=refiner_hidden_dim,
                num_layers=refiner_num_layers,
                num_heads=refiner_num_heads,
                ffn_dim=refiner_ffn_dim,
                dropout=refiner_dropout,
            )

            # 保存权重系数供后续损失计算使用
            self.refiner_weight = refiner_weight
            self.refiner_mlm_weight = refiner_mlm_weight
        else:
            self.refiner = None
            self.refiner_weight = 0.0
            self.refiner_mlm_weight = 0.0
        self.decoder_out_cache = None
    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            phoneme_text: torch.Tensor = None,
            phoneme_text_lengths: torch.Tensor = None,
    ):
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        Returns:
            total_loss, attention_loss, ctc_loss
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out,encoder_mid_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)  # [B, 1, T] -> [B]

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths,
                                                    self.reverse_weight)
        else:
            loss_att = 0.0
            acc_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            loss_ctc = 0.0

        if self.use_phoneme_ctc:
        # 3b. phoneme CTC branch
            if self.phoneme_ctc_weight != 0.0:
                encoder_mid_out = self.phoneme_norm(encoder_mid_out)
                loss_phoneme_ctc = self.phoneme_ctc(encoder_mid_out, encoder_out_lens, phoneme_text, phoneme_text_lengths)
                phoneme_probs=self.phoneme_ctc.softmax(encoder_mid_out)
                phoneme_attribute_res = self.phoneme_attribute_head(encoder_mid_out, encoder_mask.squeeze(),phoneme_probs,hard=False,topk=15)
                loss_phoneme_attribute = phoneme_attribute_res['losses']['total_loss']
                loss_phoneme_ctc=loss_phoneme_ctc+loss_phoneme_attribute*0.1
                # loss_phoneme_ctc=self.phoneme_loss(encoder_mid_out, encoder_out_lens,encoder_mask.squeeze(), phoneme_text, phoneme_text_lengths)
            else:
                loss_phoneme_ctc = 0.0
        else:
            loss_phoneme_ctc = 0.0

        # NonCausalRefiner 损失
        loss_ref = torch.tensor(0.0, device=speech.device)
        loss_error = torch.tensor(0.0, device=speech.device)

        if self.use_refiner:
            # 获取decoder hidden states
            # decoder_hidden: [B, U+1, D], 第一个位置是sos，需要去掉
            # 取 [:, 1:U+1, :] 对齐到真实文本长度
            U = text.size(1)
            hidden = self.decoder_hidden_cache[:, 0:U, :]  # [B, U, D]
            predict=self.decoder_out_cache[:,0:U,:]
            predict = torch.nn.functional.log_softmax(predict, dim=-1)
            predict=torch.argmax(predict,dim=-1)
            # 准备target: 原始文本，将ignore_id转换为-100
            target = text.clone()
            target[target == self.ignore_id] = -100
            predict[predict==self.ignore_id] = -100
            mask = (target == 0) | (target == 1) | (target == 2) | (target == 3)
            target[mask] = -100
            predict[mask] = -100

            # refiner前向传播
            refiner_result = self.refiner(
                hidden=hidden,
                target=target,
                lengths=text_lengths,
                predict=predict,
                encoder_out=encoder_out,
                encoder_mask=encoder_mask
            )
            loss_ref = refiner_result["loss_ref"]
            loss_error = refiner_result["loss_error"]

        # 3. Total loss - 统一计算公式
        total_weight = self.ctc_weight + self.phoneme_ctc_weight
        if total_weight >= 1.0:
            # 如果权重和>=1，调整attention权重避免过强惩罚
            att_weight = max(0.0, 1.0 - total_weight)
            loss = (self.ctc_weight * loss_ctc +
                    self.phoneme_ctc_weight * loss_phoneme_ctc +
                    att_weight * loss_att)
        else:
            # 正常情况
            loss = (self.ctc_weight * loss_ctc +
                    self.phoneme_ctc_weight * loss_phoneme_ctc +
                    (1.0 - total_weight) * loss_att)
        if self.use_refiner:
            # print(loss_ref,loss_ref_mlm)
            loss = loss + self.refiner_weight * loss_ref + self.refiner_mlm_weight * loss_error

        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "loss_phoneme_ctc": loss_phoneme_ctc,
            "loss_refiner": loss_ref  # 新增返回Refiner损失
        }
    def _calc_att_loss(self,
                       encoder_out: torch.Tensor,
                       encoder_mask: torch.Tensor,
                       ys_pad: torch.Tensor,
                       ys_pad_lens: torch.Tensor,
                       reverse_weight: float) -> Tuple[torch.Tensor, float]:
        """Calc attention loss.

        Args:
            encoder_out (torch.Tensor): [B, Tmax, D]
            encoder_mask (torch.Tensor): [B, 1, Tmax]
            ys_pad (torch.Tensor): [B, Umax]
            ys_pad_lens (torch.Tensor): [B]
            reverse_weight (float): reverse decoder weight.

        Returns:
            Tuple[torch.Tensor, float]: attention_loss, accuracy rate
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _,hidden,r_hidden= self.decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad, self.reverse_weight)
        #获取decoder 输出用于教师模型知识蒸馏
        self.decoder_out_cache = decoder_out
        # 保存decoder hidden states供refiner使用
        self.decoder_hidden_cache = hidden
        self.decoder_hidden_cache_r = r_hidden
        # print(ys_pad.shape,decoder_out.shape,encoder_out.shape,hidden.shape,r_hidden.shape )
        # exit()

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size),
                              ys_out_pad,
                              ignore_label=self.ignore_id)
        return loss_att, acc_att

    @torch.no_grad()
    def get_bert_score(
            self,
            hyps_pad: torch.Tensor,  # (N, Tmax) 已经 pad + 含 <sos> (建议不含 <eos> 也行)
            hyps_lens: torch.Tensor,  # (N,) 含 <sos> 的长度
            length_norm: bool = True,
            alpha: float = 1.0,
    ) -> torch.Tensor:
        """给 N-best hypotheses 打 Refiner 分数（占位符，原BERT已替换为NonCausalRefiner）。
        返回: (N,) 每条hyp一个分数，越大越好。
        TODO: 实现基于NonCausalRefiner的评分逻辑。
        """
        device = hyps_pad.device
        # 暂时返回零分数
        return torch.zeros(hyps_pad.size(0), device=device)

    @torch.jit.export
    def subsampling_rate(self) -> int:
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        return self.encoder.embed.right_context

    @torch.jit.export
    def ignore_symbol(self) -> int:
        return self.ignore_id

    @torch.jit.export
    def sos_symbol(self) -> int:
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        return self.eos

    @torch.jit.export
    def get_encoder_out(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get encoder output

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths (torch.Tensor): (batch, )
        Returns:
            Tensor: ctc softmax output
        """
        encoder_outs,encoder_mid_out, encoder_mask = self.encoder(speech,
                                                  speech_lengths,
                                                  decoding_chunk_size=-1,
                                                  num_decoding_left_chunks=-1)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(encoder_outs)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        return encoder_outs, ctc_probs, encoder_lens

    @torch.jit.export
    def get_phoneme_encoder_out(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get encoder output

               Args:
                   speech (torch.Tensor): (batch, max_len, feat_dim)
                   speech_lengths (torch.Tensor): (batch, )
               Returns:
                   Tensor: ctc softmax output
               """
        encoder_outs, encoder_mid_outs, encoder_mask = self.encoder(speech,
                                                                    speech_lengths,
                                                                    decoding_chunk_size=-1,
                                                                    num_decoding_left_chunks=-1)  # (B, maxlen, encoder_dim)

        if self.phoneme_ctc_weight != 0.0 and self.use_phoneme_ctc:  # 添加条件判断
            encoder_mid_outs = self.phoneme_norm(encoder_mid_outs)
            ctc_probs = self.phoneme_ctc.log_softmax(encoder_mid_outs)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            # res=self.phoneme_attribute_head.decode(encoder_mid_outs,encoder_mask.squeeze(1))
            # print(res)
            return encoder_outs, ctc_probs, encoder_lens
        else:
            # 当phoneme CTC权重为0时，返回相应形状的零张量
            batch_size = speech.size(0)
            max_len = encoder_outs.size(1)
            device = speech.device
            dtype = speech.dtype

            zero_encoder_outs = torch.zeros_like(encoder_outs)
            zero_ctc_probs = torch.zeros(batch_size, max_len, self.phoneme_vocab_size,
                                         device=device, dtype=dtype)
            zero_encoder_lens = torch.zeros(batch_size, device=device, dtype=torch.long)

            return zero_encoder_outs, zero_ctc_probs, zero_encoder_lens

    @torch.jit.export
    def get_phoneme_encoder_out_ce(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        """ Get encoder output

               Args:
                   speech (torch.Tensor): (batch, max_len, feat_dim)
                   speech_lengths (torch.Tensor): (batch, )
               Returns:
                   Tensor: ctc softmax output
               """
        encoder_outs, encoder_mid_out, encoder_mask = self.encoder(speech,
                                                                   speech_lengths,
                                                                   decoding_chunk_size=-1,
                                                                   num_decoding_left_chunks=-1)  # (B, maxlen, encoder_dim)
        # ctc_probs = self.phoneme_ctc.log_softmax(encoder_outs)
        probs,_=self.phoneme_loss.decode(encoder_mid_out,encoder_mask.squeeze(1))
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        return encoder_outs, probs, encoder_lens
    @torch.jit.export
    def get_encoder_out_chunk(self,
                              speech: torch.Tensor,
                              offset: int,
                              required_cache_size: int,
                              att_cache: torch.Tensor = torch.zeros([0, 0, 0, 0]),
                              cnn_cache: torch.Tensor = torch.zeros([0, 0, 0, 0])) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get encoder output

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
        Returns:
            Tensor: ctc softmax output
        """
        xs, att_cache, cnn_cache = self.encoder.forward_chunk(xs=speech,
                                                              offset=offset,
                                                              required_cache_size=required_cache_size,
                                                              att_cache=att_cache,
                                                              cnn_cache=cnn_cache)
        ctc_probs = self.ctc.log_softmax(xs)
        return ctc_probs, att_cache, cnn_cache

    @torch.jit.export
    def get_decoder_out(
            self,
            hyps: torch.Tensor,
            hyps_lens: torch.Tensor,
            encoder_out: torch.Tensor,
            reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        r_hyps_lens = hyps_lens - 1
        r_hyps = hyps[:, 1:]
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        index = index * seq_mask
        r_hyps = torch.gather(r_hyps, 1, index)
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        decoder_out, r_decoder_out, _,hidden,r_hidden= self.decoder(encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
                                                     reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out

    # model.py 中修改

    @torch.jit.export
    def get_decoder_out_with_refiner(
            self,
            hyps: torch.Tensor,
            hyps_lens: torch.Tensor,
            encoder_out: torch.Tensor,
            reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带 Refiner 精炼的解码输出
        Args:
            hyps: [num_hyps, max_len] 已添加 sos 的候选序列
            hyps_lens: [num_hyps] 每条候选的长度（含 sos）
            encoder_out: [1, T, D] 编码器输出
            reverse_weight: 反向解码器权重
        Returns:
            decoder_out: [num_hyps, max_len, vocab_size] refiner 精炼后的输出
            r_decoder_out: [num_hyps, max_len, vocab_size] 反向解码器输出
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps

        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(
            num_hyps, 1, encoder_out.size(1),
            dtype=torch.bool, device=encoder_out.device
        )

        # 构建反向解码输入
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

        # Decoder 前向传播
        decoder_out, r_decoder_out, _, hidden, r_hidden = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps, reverse_weight
        )

        # ===== Refiner 精炼 =====
        # 需要逐条处理，因为不同 hyp 长度不同
        max_hyp_len = hyps.size(1)
        vocab_size = decoder_out.size(-1)
        refined_decoder_out = torch.zeros(
            num_hyps, max_hyp_len, vocab_size,
            device=decoder_out.device, dtype=decoder_out.dtype
        )

        for i in range(num_hyps):
            hyp_len = hyps_lens[i].item()  # 含 sos 的长度
            # hidden[i] 形状: [max_hyp_len, D]
            # 取第 1 到 hyp_len-1 位置（去掉 sos，不含 eos）
            # 因为 decoder 输出对应预测下一个 token，hidden[i, 0] 对应 sos 输入
            hidden_text = hidden[i:i + 1, 1:hyp_len, :]  # [1, text_len, D]
            text_len = hyp_len - 1  # 实际文本 token 数量

            # Refiner decode
            refined_logits = self.refiner.decode(
                hidden_text,
                torch.tensor([text_len], device=hidden.device),
                encoder_out[i:i + 1],
                encoder_mask[i:i + 1]
            )  # 返回 tuple: (pred, logits)，取 logits

            # refined_logits 是 [1, text_len, vocab_size]
            if isinstance(refined_logits, tuple):
                refined_logits = refined_logits[1]

            # 拼接: decoder_out[i, 0] 是 sos 位置的输出（预测第一个文本token）
            # refined_logits 是对文本 token hidden 的精炼输出
            # 我们用 decoder_out[0] 作为第一个位置的输出（因为 refiner 没有处理 sos）
            refined_decoder_out[i, 0, :] = decoder_out[i, 0, :]
            refined_decoder_out[i, 1:text_len + 1, :] = refined_logits[0, :, :]
            # padding 位置保持为 0

        decoder_out = torch.nn.functional.log_softmax(refined_decoder_out, dim=-1)
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)

        return decoder_out, r_decoder_out

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

    @torch.no_grad()
    def refine_hyp(
            self,
            hyp: torch.Tensor,  # [1, L]
            hyp_len: torch.Tensor,  # [1]
            encoder_out: torch.Tensor,  # [1, T, D]
            encoder_mask: torch.Tensor = None,  # 新增参数
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
        )  # [1, L+1, D]

        # 去掉 sos，对齐到真实文本长度
        hidden_text = decoder_hidden[:, 1:hyp_len.item() + 1, :]  # [1, L, D]

        # 构建 encoder_mask 如果未提供
        if encoder_mask is None:
            encoder_mask = torch.ones(
                1, 1, encoder_out.size(1),
                dtype=torch.bool,
                device=encoder_out.device
            )

        refined_pred, refined_logits = self.refiner.decode(
            hidden_text,
            hyp_len,
            encoder_out,
            encoder_mask  # 传入 encoder_mask
        )

        return refined_pred[0]  # [L]



