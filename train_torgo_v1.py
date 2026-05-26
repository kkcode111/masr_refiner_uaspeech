import argparse
import functools
import os

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("configs", str, "configs/torgo_conformer_v1.yml", "configs file")
add_arg("data_augment_configs", str, "configs/torgo_speed_augmentation.yml", "data augmentation configs file")
add_arg("local_rank", int, 0, "local GPU rank for distributed training")
add_arg("use_gpu", bool, True, "use GPU for training")
add_arg("metrics_type", str, "wer", "evaluation metric")
add_arg("save_model_path", str, "torgomodel/v1/", "model save path")
add_arg("log_dir", str, "torgomodel/logs/v1/", "VisualDL log path")
add_arg("resume_model", str, None, "checkpoint path for resume training")
add_arg("pretrained_model", str, "Comformer_Librispeech/models/ConformerModel_fbank/best_model",
        "external ASR pretrained model path")
add_arg("overwrites", str, None, "config overwrites, separated by comma")
add_arg("freeze_decoder_epochs", int, 0, "decoder freeze epochs, 0 means no freeze")
args = parser.parse_args()

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print_arguments(args=args)

if args.freeze_decoder_epochs is not None:
    kv = f"train_conf.freeze_decoder_epochs={args.freeze_decoder_epochs}"
    args.overwrites = kv if args.overwrites is None else f"{args.overwrites},{kv}"

trainer = MASRTrainer(configs=args.configs,
                      use_gpu=args.use_gpu,
                      metrics_type=args.metrics_type,
                      data_augment_configs=args.data_augment_configs,
                      overwrites=args.overwrites)

trainer.train(save_model_path=args.save_model_path,
              log_dir=args.log_dir,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
