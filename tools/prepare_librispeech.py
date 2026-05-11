import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import wave
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from masr.data_utils.normalizer import FeatureNormalizer
from masr.utils.utils import dict_to_object


@dataclass
class ManifestItem:
    audio_filepath: str
    duration: float
    text: str


def _iter_transcription_files(split_root: str) -> Iterable[str]:
    for root, _, files in os.walk(split_root):
        for fn in files:
            if fn.endswith(".trans.txt"):
                yield os.path.join(root, fn)


def _parse_trans_file(trans_path: str) -> Iterable[Tuple[str, str]]:
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1].strip()


def _find_audio_path(dir_path: str, utt_id: str) -> str:
    for ext in (".flac", ".wav"):
        p = os.path.join(dir_path, f"{utt_id}{ext}")
        if os.path.exists(p):
            return p
    return ""


def _get_duration_sec(audio_path: str) -> float:
    if audio_path.endswith(".flac"):
        with open(audio_path, "rb") as f:
            magic = f.read(4)
            if magic != b"fLaC":
                raise ValueError(audio_path)
            while True:
                header = f.read(4)
                if len(header) < 4:
                    raise ValueError(audio_path)
                block_type = header[0] & 0x7F
                block_len = int.from_bytes(header[1:4], "big")
                block = f.read(block_len)
                if len(block) < block_len:
                    raise ValueError(audio_path)
                if block_type == 0:
                    if len(block) != 34:
                        raise ValueError(audio_path)
                    x = int.from_bytes(block[10:18], "big")
                    sample_rate = (x >> 44) & ((1 << 20) - 1)
                    total_samples = x & ((1 << 36) - 1)
                    if sample_rate <= 0:
                        raise ValueError(audio_path)
                    return float(total_samples) / float(sample_rate)
                if header[0] & 0x80:
                    break
        raise ValueError(audio_path)
    if audio_path.endswith(".wav"):
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            if sr <= 0:
                raise ValueError(audio_path)
            return float(frames) / float(sr)
    raise ValueError(audio_path)


def _make_relative(path: str, base_dir: str) -> str:
    rel = os.path.relpath(path, base_dir)
    return rel.replace("\\", "/")


def build_manifest(
        librispeech_root: str,
        splits: List[str],
        repo_root: str,
        min_duration: float,
        max_duration: float,
        max_items: int = 0,
) -> Tuple[List[Dict], Dict[str, int]]:
    items: List[ManifestItem] = []
    skipped_audio = 0
    missing_audio = 0
    for split in splits:
        split_root = os.path.join(librispeech_root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(split_root)
        for trans_path in _iter_transcription_files(split_root):
            dir_path = os.path.dirname(trans_path)
            for utt_id, text in _parse_trans_file(trans_path):
                audio_path = _find_audio_path(dir_path, utt_id)
                if not audio_path:
                    missing_audio += 1
                    continue
                try:
                    duration = _get_duration_sec(audio_path)
                except Exception:
                    skipped_audio += 1
                    continue
                if duration < min_duration:
                    continue
                if max_duration != -1 and duration > max_duration:
                    continue
                items.append(ManifestItem(
                    audio_filepath=_make_relative(audio_path, repo_root),
                    duration=duration,
                    text=text.lower().strip(),
                ))
                if isinstance(max_items, int) and max_items > 0 and len(items) >= max_items:
                    items.sort(key=lambda x: x.duration)
                    return [item.__dict__ for item in items], {
                        "skipped_audio": skipped_audio,
                        "missing_audio": missing_audio,
                    }
    items.sort(key=lambda x: x.duration)
    return [item.__dict__ for item in items], {
        "skipped_audio": skipped_audio,
        "missing_audio": missing_audio,
    }


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_mean_istd(config_path: str, train_manifest: str, mean_istd_path: str, num_samples: int):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    preprocess_conf = dict_to_object(cfg["preprocess_conf"])
    data_loader_conf = cfg["dataset_conf"]["dataLoader"]
    data_loader_conf = dict(data_loader_conf)
    data_loader_conf["num_workers"] = 0
    normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
    normalizer.compute_mean_istd(
        manifest_path=train_manifest,
        preprocess_conf=preprocess_conf,
        data_loader_conf=data_loader_conf,
        num_samples=num_samples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--librispeech_root", type=str, default=None)
    parser.add_argument("--train_splits", type=str, default="train-clean-100,train-clean-360,train-other-500")
    parser.add_argument("--dev_splits", type=str, default="dev-clean,dev-other")
    parser.add_argument("--train_manifest", type=str, default="librispeech/manifest.train.jsonl")
    parser.add_argument("--dev_manifest", type=str, default="librispeech/manifest.dev.jsonl")
    parser.add_argument("--config_for_cmvn", type=str, default="configs/librispeech_conformer.yml")
    parser.add_argument("--mean_istd_path", type=str, default="librispeech/mean_istd.json")
    parser.add_argument("--min_duration", type=float, default=0.5)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=200000)
    parser.add_argument("--skip_mean_istd", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    if args.librispeech_root is None:
        librispeech_root = os.path.join(repo_root, "librispeech")
    else:
        librispeech_root = os.path.abspath(args.librispeech_root)

    train_splits = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    dev_splits = [s.strip() for s in args.dev_splits.split(",") if s.strip()]

    train_manifest_path = os.path.join(repo_root, args.train_manifest)
    dev_manifest_path = os.path.join(repo_root, args.dev_manifest)
    mean_istd_path = os.path.join(repo_root, args.mean_istd_path)

    print(f"repo_root={repo_root}")
    print(f"librispeech_root={librispeech_root}")
    print(f"train_splits={train_splits}")
    print(f"dev_splits={dev_splits}")

    train_rows, train_stats = build_manifest(
        librispeech_root=librispeech_root,
        splits=train_splits,
        repo_root=repo_root,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_items=args.max_items,
    )
    dev_rows, dev_stats = build_manifest(
        librispeech_root=librispeech_root,
        splits=dev_splits,
        repo_root=repo_root,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_items=args.max_items,
    )

    print(f"train_rows={len(train_rows)}")
    print(f"dev_rows={len(dev_rows)}")
    print(f"train_stats={train_stats}")
    print(f"dev_stats={dev_stats}")

    write_jsonl(train_manifest_path, train_rows)
    write_jsonl(dev_manifest_path, dev_rows)
    print(f"wrote {train_manifest_path}")
    print(f"wrote {dev_manifest_path}")

    if not args.skip_mean_istd:
        config_path = os.path.join(repo_root, args.config_for_cmvn)
        compute_mean_istd(
            config_path=config_path,
            train_manifest=train_manifest_path,
            mean_istd_path=mean_istd_path,
            num_samples=args.num_samples,
        )
        print(f"wrote {mean_istd_path}")


if __name__ == "__main__":
    main()
