import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from masr.utils.utils import dict_to_object


DEFAULT_SPEED_RATES = (0.9, 1.0, 1.1)


def normalize_text(text, keep_instruction_prompts=False):
    text = text.strip()
    if not text:
        return ""
    if text.lower().startswith("input/images/"):
        return ""
    if text.startswith("[") and text.endswith("]") and not keep_instruction_prompts:
        return ""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"x+", text):
        return ""
    return text


def iter_torgo_rows(torgo_root, mic_name, keep_instruction_prompts):
    torgo_root = Path(torgo_root)
    for wav_path in sorted(torgo_root.glob(f"*/Session*/{mic_name}/*.wav")):
        speaker = wav_path.parts[-4]
        session = wav_path.parts[-3]
        prompt_path = wav_path.parents[1] / "prompts" / f"{wav_path.stem}.txt"
        if not prompt_path.exists():
            continue
        text = normalize_text(prompt_path.read_text(encoding="utf-8", errors="ignore"),
                              keep_instruction_prompts=keep_instruction_prompts)
        if not text:
            continue
        info = sf.info(str(wav_path))
        duration = float(info.frames) / float(info.samplerate)
        yield {
            "audio_filepath": wav_path.as_posix(),
            "text": text,
            "duration": round(duration, 5),
            "speaker": speaker,
            "session": session,
            "mic": mic_name,
        }


def stratify_value(row, stratify_by):
    if stratify_by == "speaker":
        return row["speaker"]
    if stratify_by == "group":
        return "control" if row["speaker"].startswith(("FC", "MC")) else "dysarthric"
    return "all"


def subset_by_duration(rows, target_hours, stratify_by, seed):
    if target_hours is None or target_hours <= 0:
        return rows
    target_duration = target_hours * 3600.0
    available_duration = sum(row["duration"] for row in rows)
    if target_duration >= available_duration:
        return rows

    rng = random.Random(seed)
    grouped = {}
    for row in rows:
        grouped.setdefault(stratify_value(row, stratify_by), []).append(row)

    selected = []
    for _, group_rows in sorted(grouped.items()):
        group_rows = list(group_rows)
        rng.shuffle(group_rows)
        group_duration = sum(row["duration"] for row in group_rows)
        group_target = target_duration * group_duration / available_duration
        running_duration = 0.0
        for row in group_rows:
            if running_duration >= group_target:
                break
            selected.append(row)
            running_duration += row["duration"]
    selected.sort(key=lambda row: row["duration"])
    return selected


def split_rows(rows, split_strategy, test_speakers, test_ratio, seed, stratify_by):
    if split_strategy == "speaker":
        test_speakers = set(test_speakers)
        train_rows = [row for row in rows if row["speaker"] not in test_speakers]
        test_rows = [row for row in rows if row["speaker"] in test_speakers]
    elif split_strategy == "random":
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        test_size = max(1, int(round(len(rows) * test_ratio)))
        test_rows = rows[:test_size]
        train_rows = rows[test_size:]
    else:
        rng = random.Random(seed)
        train_rows = []
        test_rows = []
        grouped = {}
        for row in rows:
            grouped.setdefault(stratify_value(row, stratify_by), []).append(row)
        for _, group_rows in sorted(grouped.items()):
            group_rows = list(group_rows)
            rng.shuffle(group_rows)
            test_target = sum(row["duration"] for row in group_rows) * test_ratio
            running_test_duration = 0.0
            for row in group_rows:
                if running_test_duration < test_target:
                    test_rows.append(row)
                    running_test_duration += row["duration"]
                else:
                    train_rows.append(row)
    train_rows.sort(key=lambda row: row["duration"])
    test_rows.sort(key=lambda row: row["duration"])
    return train_rows, test_rows


def filter_duration(rows, min_duration, max_duration):
    filtered = []
    for row in rows:
        if row["duration"] < min_duration:
            continue
        if max_duration != -1 and row["duration"] > max_duration:
            continue
        filtered.append(row)
    return filtered


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_speed_manifest(path, rows, speed_rates):
    augmented_rows = []
    for row in rows:
        for speed_rate in speed_rates:
            speed_row = dict(row)
            speed_row["speed_rate"] = speed_rate
            speed_row["duration"] = round(row["duration"] / speed_rate, 5)
            augmented_rows.append(speed_row)
    augmented_rows.sort(key=lambda row: row["duration"])
    write_jsonl(path, augmented_rows)
    return augmented_rows


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return dict_to_object(yaml.safe_load(f))


def build_vocab(config, manifest_paths):
    from masr.data_utils.tokenizer import MASRTokenizer

    tokenizer_conf = dict(config["tokenizer_conf"])
    tokenizer = MASRTokenizer(is_build_vocab=True, **tokenizer_conf)
    tokenizer.build_vocab(manifest_paths=manifest_paths)


def compute_cmvn(config, train_manifest, mean_istd_path, num_samples):
    from masr.data_utils.normalizer import FeatureNormalizer

    normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
    normalizer.compute_mean_istd(
        manifest_path=train_manifest,
        preprocess_conf=config["preprocess_conf"],
        data_loader_conf=config["dataset_conf"]["dataLoader"],
        num_samples=num_samples,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TORGO manifests for MASR training.")
    parser.add_argument("--torgo_root", default="dataset/audio/TORGO",
                        help="TORGO raw dataset root. Falls back to dataset/torgo if this path is absent.")
    parser.add_argument("--mic", default="wav_arrayMic", choices=["wav_arrayMic", "wav_headMic"])
    parser.add_argument("--config", default="configs/torgo_conformer_v18.yml")
    parser.add_argument("--train_manifest", default="dataset/torgo/train.jsonl")
    parser.add_argument("--train_aug_manifest", default="dataset/torgo/train_speed.jsonl")
    parser.add_argument("--test_manifest", default="dataset/torgo/test.jsonl")
    parser.add_argument("--mean_istd_path", default="dataset/torgo/mean_istd.json")
    parser.add_argument("--split_strategy", default="stratified", choices=["speaker", "random", "stratified"])
    parser.add_argument("--stratify_by", default="speaker", choices=["speaker", "group", "none"],
                        help="Stratification key for split_strategy=stratified.")
    parser.add_argument("--target_total_hours", type=float, default=5.5,
                        help="Randomly downsample to this many hours before splitting. Use <=0 to keep all rows.")
    parser.add_argument("--test_speakers", default="F01,M01,FC01,MC01",
                        help="Comma-separated speakers held out when split_strategy=speaker.")
    parser.add_argument("--test_ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--speed_rates", default=",".join(str(rate) for rate in DEFAULT_SPEED_RATES),
                        help="Comma-separated speed rates written only to train_aug_manifest.")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Samples used for CMVN. Use -1 for all training rows.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override dataset_conf.dataLoader.num_workers during CMVN.")
    parser.add_argument("--skip_cmvn", action="store_true")
    parser.add_argument("--build_vocab", action="store_true",
                        help="Build tokenizer from TORGO. Default keeps the config vocab for pretrained loading.")
    parser.add_argument("--keep_instruction_prompts", action="store_true",
                        help="Keep bracketed TORGO articulation prompts instead of dropping them.")
    parser.add_argument("--no_duration_filter", action="store_true",
                        help="Do not pre-filter rows with dataset min_duration/max_duration from the config.")
    return parser.parse_args()


def main():
    args = parse_args()
    torgo_root = Path(args.torgo_root)
    if not torgo_root.exists() and args.torgo_root == "dataset/audio/TORGO":
        torgo_root = Path("dataset/torgo")
    if not torgo_root.exists():
        raise FileNotFoundError(f"TORGO root not found: {args.torgo_root}")

    config = load_config(args.config)
    rows = list(iter_torgo_rows(torgo_root, args.mic, args.keep_instruction_prompts))
    if not args.no_duration_filter:
        rows = filter_duration(
            rows,
            min_duration=config["dataset_conf"]["dataset"]["min_duration"],
            max_duration=config["dataset_conf"]["dataset"]["max_duration"],
        )
    rows = subset_by_duration(rows, args.target_total_hours, args.stratify_by, args.seed)
    if not rows:
        raise RuntimeError(f"No TORGO utterances found under {torgo_root} with mic={args.mic}")

    test_speakers = [speaker.strip() for speaker in args.test_speakers.split(",") if speaker.strip()]
    train_rows, test_rows = split_rows(rows, args.split_strategy, test_speakers, args.test_ratio, args.seed,
                                       args.stratify_by)
    if not train_rows or not test_rows:
        raise RuntimeError("Split produced an empty train or test set. Adjust split settings.")

    speed_rates = [float(rate.strip()) for rate in args.speed_rates.split(",") if rate.strip()]
    write_jsonl(args.train_manifest, train_rows)
    train_aug_rows = write_speed_manifest(args.train_aug_manifest, train_rows, speed_rates)
    write_jsonl(args.test_manifest, test_rows)

    total_hours = sum(row["duration"] for row in rows) / 3600.0
    train_hours = sum(row["duration"] for row in train_rows) / 3600.0
    test_hours = sum(row["duration"] for row in test_rows) / 3600.0
    train_aug_hours = sum(row["duration"] for row in train_aug_rows) / 3600.0
    print(f"TORGO root: {torgo_root}")
    print(f"rows: total={len(rows)}, train={len(train_rows)}, test={len(test_rows)}")
    print(f"hours: total={total_hours:.2f}, train={train_hours:.2f}, "
          f"train_speed={train_aug_hours:.2f}, test={test_hours:.2f}")
    print(f"wrote: {args.train_manifest}")
    print(f"wrote: {args.train_aug_manifest}")
    print(f"wrote: {args.test_manifest}")

    config["dataset_conf"]["train_manifest"] = args.train_aug_manifest
    config["dataset_conf"]["test_manifest"] = args.test_manifest
    config["dataset_conf"]["mean_istd_path"] = args.mean_istd_path
    if args.num_workers is not None:
        config["dataset_conf"]["dataLoader"]["num_workers"] = args.num_workers

    if not args.skip_cmvn:
        compute_cmvn(config, args.train_manifest, args.mean_istd_path, args.num_samples)
        print(f"wrote: {args.mean_istd_path}")

    if args.build_vocab:
        build_vocab(config, [args.train_manifest, args.test_manifest])
        print(f"wrote vocab under: {config['tokenizer_conf']['vocab_model_dir']}")


if __name__ == "__main__":
    main()
