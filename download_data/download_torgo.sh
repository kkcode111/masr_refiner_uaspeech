#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash download_data/download_torgo.sh [--output-dir dataset/torgo] [--skip-extract]

Downloads the official TORGO corpus archives into dataset/torgo by default.
The download is resumable when curl is available.
EOF
}

output_dir="dataset/torgo"
skip_extract=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="${2:?Missing value for --output-dir}"
      shift 2
      ;;
    --skip-extract)
      skip_extract=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
target_dir="$repo_root/$output_dir"

base_url="https://www.cs.toronto.edu/~complingweb/data/TORGO"
files=(
  "F.tar.bz2"
  "FC.tar.bz2"
  "M.tar.bz2"
  "MC.tar.bz2"
  "doc/ERRORS.xls"
  "doc/CoilLocations.pdf"
)
archives=(
  "F.tar.bz2"
  "FC.tar.bz2"
  "M.tar.bz2"
  "MC.tar.bz2"
)

mkdir -p "$target_dir"

download_file() {
  local url="$1"
  local destination="$2"

  mkdir -p "$(dirname -- "$destination")"
  echo "Downloading $url"

  if command -v curl >/dev/null 2>&1; then
    curl \
      --location \
      --fail \
      --continue-at - \
      --retry 8 \
      --retry-all-errors \
      --retry-delay 15 \
      --output "$destination" \
      "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget \
      --continue \
      --tries=8 \
      --waitretry=15 \
      --output-document="$destination" \
      "$url"
  else
    echo "Neither curl nor wget is installed. Please install one of them and retry." >&2
    exit 1
  fi
}

for file in "${files[@]}"; do
  download_file "$base_url/$file" "$target_dir/$file"
done

if [[ "$skip_extract" -eq 0 ]]; then
  if ! command -v tar >/dev/null 2>&1; then
    echo "tar is not installed. Install tar or rerun with --skip-extract." >&2
    exit 1
  fi

  for archive in "${archives[@]}"; do
    archive_path="$target_dir/$archive"
    echo "Extracting $archive_path"
    tar -xjf "$archive_path" -C "$target_dir"
  done
fi

echo "TORGO download finished at $target_dir"
