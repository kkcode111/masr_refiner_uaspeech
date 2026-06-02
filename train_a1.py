import argparse
import functools
import os

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,    'configs/conformer_a1.yml',     'configs file')
add_arg('data_augment_configs', str,    'configs/augmentation.yml',    'data augmentation configs file')
add_arg('data_profile',         str,    'uaspeech_only',               'dataset profile: uaspeech_only, base, speed_augmented, or config',
        choices=['uaspeech_only', 'base', 'speed_augmented', 'config'])
add_arg('use_online_augmentation', bool, False,                        'apply augmentation.yml augmentors while loading audio')
add_arg("local_rank",           int,    0,                             'local GPU rank for distributed training')
add_arg("use_gpu",              bool,   True,                          'use GPU for training')
add_arg('metrics_type',         str,    'wer',                         'evaluation metric')
add_arg('save_model_path',      str,    'uaspeechmodel/refiner_a1/',   'model save path')
add_arg('log_dir',              str,    'uaspeechmodel/logs/refiner_a1/', 'VisualDL log path')
add_arg('resume_model',         str,    None,                          'checkpoint path for resume training')
add_arg('pretrained_model',     str,    'pretrain_models/refiner_pretrain_v1/ConformerModel_fbank/best_model', 'pretrained model path')
add_arg('overwrites',           str,    'train_conf.max_epoch=60',     'config overwrites, separated by comma')
add_arg('freeze_decoder_epochs', int,   40,                            'decoder freeze epochs, 0 means no freeze')
args = parser.parse_args()

if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    print_arguments(args=args)


def append_overwrite(overwrites, key, value):
    item = f"{key}={value}"
    return item if overwrites is None else f"{overwrites},{item}"


if args.freeze_decoder_epochs is not None:
    args.overwrites = append_overwrite(args.overwrites, 'train_conf.freeze_decoder_epochs',
                                       args.freeze_decoder_epochs)

audio_path_contains = None
if args.data_profile == 'uaspeech_only':
    args.overwrites = append_overwrite(args.overwrites, 'dataset_conf.train_manifest',
                                       'dataset/train.jsonl')
    args.overwrites = append_overwrite(args.overwrites, 'dataset_conf.test_manifest',
                                       'dataset/test_original.jsonl')
    audio_path_contains = 'audio/UASpeech'
elif args.data_profile == 'base':
    args.overwrites = append_overwrite(args.overwrites, 'dataset_conf.train_manifest',
                                       'dataset/train.jsonl')
    args.overwrites = append_overwrite(args.overwrites, 'dataset_conf.test_manifest',
                                       'dataset/test_original.jsonl')
elif args.data_profile == 'speed_augmented':
    args.overwrites = append_overwrite(args.overwrites, 'dataset_conf.train_manifest',
                                       'dataset/train_aug.jsonl')
    args.overwrites = append_overwrite(args.overwrites, 'dataset_conf.test_manifest',
                                       'dataset/test_original.jsonl')

if not args.use_online_augmentation:
    args.data_augment_configs = None

trainer = MASRTrainer(configs=args.configs,
                      use_gpu=args.use_gpu,
                      metrics_type=args.metrics_type,
                      data_augment_configs=args.data_augment_configs,
                      overwrites=args.overwrites,
                      audio_path_contains=audio_path_contains)

trainer.train(save_model_path=args.save_model_path,
              log_dir=args.log_dir,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
