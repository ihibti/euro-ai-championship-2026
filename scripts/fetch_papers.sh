#!/usr/bin/env bash
# Re-download the three organizer reference papers from arXiv and verify checksums.
set -euo pipefail

cd "$(dirname "$0")/../papers"

fetch() {
  local id="$1" name="$2" sha="$3"
  local file="${id}_${name}.pdf"
  if [[ ! -f "$file" ]]; then
    curl -fsSL --max-time 180 -o "$file.part" "https://arxiv.org/pdf/${id}"
    mv "$file.part" "$file"
  fi
  echo "${sha}  ${file}" | shasum -a 256 -c -
}

fetch 2512.16874v1 pixel-seal 76f12ac4e8039150fa931d54b06d52ddbe85675b1d71d724e5cac29cdea78962
fetch 2410.20247v2 model-equality-testing 07aa991f780630045e3f44104f3f63fbc496c4abb431c0c2ca9c7bca3746e98d
fetch 2302.09880v3 scrub-unbounded-unlearning 4c9486691a7a83f7fdb97b1d860f4ba12f948ed8bcb5db4406628f5589da6db4
