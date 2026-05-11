import argparse
import functools
import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import torch

from masr.trainer import MASRTrainer
from masr.utils.checkpoint import load_checkpoint, load_pretrained, save_checkpoint
from masr.utils.utils import add_arguments, print_arguments


def _append_overwrite(overwrites: str, kv: str) -> str:
    if overwrites is None or len(overwrites.strip()) == 0:
        return kv
    key = kv.split("=", 1)[0].strip()
    for part in overwrites.split(","):
        if part.strip().startswith(f"{key}="):
            return overwrites
    return f"{overwrites},{kv}"


def _set_module_requires_grad(module: torch.nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def _freeze_encoder_and_enable_refiner(trainer: MASRTrainer):
    model = trainer.model.module if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else trainer.model
    if hasattr(model, "encoder"):
        _set_module_requires_grad(model.encoder, False)
    if hasattr(model, "nar") and model.nar is not None:
        _set_module_requires_grad(model.nar, True)


def pretrain(
        trainer: MASRTrainer,
        save_model_path: str,
        log_dir: str,
        max_epoch: int,
        resume_model: str = None,
        pretrained_model: str = None,
):
    nranks = torch.cuda.device_count()
    if nranks > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        trainer.local_rank = int(os.environ["LOCAL_RANK"])
    writer = None
    if trainer.local_rank == 0:
        from visualdl import LogWriter
        writer = LogWriter(logdir=log_dir)

    trainer._MASRTrainer__setup_dataloader(is_train=True)
    trainer._MASRTrainer__setup_model(
        input_dim=trainer.audio_featurizer.feature_dim,
        tokenizer=trainer.tokenizer,
        is_train=True,
    )
    trainer.model = load_pretrained(model=trainer.model, pretrained_model=pretrained_model)
    trainer.model, trainer.optimizer, trainer.amp_scaler, trainer.scheduler, last_epoch, trainer.eval_best_error_rate = \
        load_checkpoint(
            configs=trainer.configs,
            model=trainer.model,
            optimizer=trainer.optimizer,
            amp_scaler=trainer.amp_scaler,
            scheduler=trainer.scheduler,
            step_epoch=len(trainer.train_loader),
            save_model_path=save_model_path,
            resume_model=resume_model,
        )

    if nranks > 1:
        trainer.model.to(trainer.local_rank)
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model, device_ids=[trainer.local_rank])

    trainer.configs.train_conf.max_epoch = max_epoch
    _freeze_encoder_and_enable_refiner(trainer)

    if trainer.local_rank == 0:
        print(f"vocab_size: {trainer.tokenizer.vocab_size}")
        print(f"train_dataset: {len(trainer.train_dataset)}")
        print(f"test_dataset: {len(trainer.test_dataset)}")

    trainer.train_loss, trainer.eval_loss = None, None
    trainer.test_log_step, trainer.train_log_step = 0, 0
    trainer.train_batch_sampler.epoch = last_epoch
    trainer._maybe_update_decoder_freeze(last_epoch + 1)

    trainer.max_step = len(trainer.train_loader) * trainer.configs.train_conf.max_epoch
    trainer.train_step = max(last_epoch, 0) * len(trainer.train_loader)

    for epoch_id in range(last_epoch, trainer.configs.train_conf.max_epoch):
        if trainer.stop_train:
            break
        epoch_id += 1
        trainer._maybe_update_decoder_freeze(epoch_id)
        start_epoch = torch.cuda.Event(enable_timing=True) if trainer.use_gpu else None
        end_epoch = torch.cuda.Event(enable_timing=True) if trainer.use_gpu else None
        if trainer.use_gpu:
            start_epoch.record()
        trainer._MASRTrainer__train_epoch(epoch_id=epoch_id, save_model_path=save_model_path, writer=writer)

        if trainer.local_rank == 0 and not trainer.stop_eval:
            trainer.eval_loss, trainer.eval_error_result = trainer.evaluate()
            if writer is not None:
                writer.add_scalar(f"Test/{trainer.metrics_type}", trainer.eval_error_result, trainer.test_log_step)
                writer.add_scalar("Test/Loss", trainer.eval_loss, trainer.test_log_step)
            trainer.test_log_step += 1
            trainer.model.train()
            if trainer.eval_error_result <= trainer.eval_best_error_rate:
                trainer.eval_best_error_rate = trainer.eval_error_result
                save_checkpoint(
                    configs=trainer.configs,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    amp_scaler=trainer.amp_scaler,
                    save_model_path=save_model_path,
                    epoch_id=epoch_id,
                    error_rate=trainer.eval_error_result,
                    metrics_type=trainer.metrics_type,
                    best_model=True,
                )
            save_checkpoint(
                configs=trainer.configs,
                model=trainer.model,
                optimizer=trainer.optimizer,
                amp_scaler=trainer.amp_scaler,
                save_model_path=save_model_path,
                epoch_id=epoch_id,
                error_rate=trainer.eval_error_result,
                metrics_type=trainer.metrics_type,
            )
        if trainer.use_gpu:
            end_epoch.record()
            torch.cuda.synchronize()


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("configs", str, "configs/librispeech_conformer.yml", "配置文件")
add_arg("data_augment_configs", str, "configs/augmentation.yml", "数据增强配置文件")
add_arg("use_gpu", bool, True, "是否使用GPU训练")
add_arg("metrics_type", str, "wer", "评估指标类型，中文用cer，英文用wer，中英混合用mer")
add_arg("save_model_path", str, "pretrain_models/refiner_pretrain_v1/", "模型保存的路径")
add_arg("log_dir", str, "pretrain_models/logs/refiner_pretrain_v1/", "保存VisualDL日志文件的路径")
add_arg("resume_model", str, None, "恢复训练，当为None则不使用恢复模型")
add_arg("pretrained_model", str, "/root/autodl-tmp/MASR/Comformer_Librispeech/models/ConformerModel_fbank/best_model/", "初始化权重的路径（可选）")
add_arg("overwrites", str, None, '覆盖配置参数，例："train_conf.max_epoch=20"')
add_arg("max_epoch", int, 20, "预训练epoch数")
add_arg("freeze_decoder_epochs", int, 10, "前N个epoch冻结decoder，之后解冻")
args = parser.parse_args()

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print_arguments(args=args)

args.overwrites = _append_overwrite(args.overwrites, f"train_conf.max_epoch={args.max_epoch}")
args.overwrites = _append_overwrite(args.overwrites, f"train_conf.freeze_decoder_epochs={args.freeze_decoder_epochs}")
args.overwrites = _append_overwrite(args.overwrites, "model_conf.model_args.if_use_nar=True")

trainer = MASRTrainer(
    configs=args.configs,
    use_gpu=args.use_gpu,
    metrics_type=args.metrics_type,
    data_augment_configs=args.data_augment_configs,
    overwrites=args.overwrites,
)

pretrain(
    trainer=trainer,
    save_model_path=args.save_model_path,
    log_dir=args.log_dir,
    max_epoch=args.max_epoch,
    resume_model=args.resume_model,
    pretrained_model=args.pretrained_model,
)
