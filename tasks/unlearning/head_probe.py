"""Which loss the original model was trained with, read from its last layer alone.

Fetches only the pickle and four tensors of the public model.pt (about 0.4 MB) with HTTP range
requests, and prints the head bias, the class-mean weight vector and a bound on the mean logit
over the 100 classes that holds for every input. Under softmax cross-entropy the mean over
classes receives no gradient and stays near its starting point of 0; under a binary
cross-entropy it is pushed far below 0. No GPU, no token:

    uv run --extra api python tasks/unlearning/head_probe.py
"""

from __future__ import annotations

import collections
import io
import pickle
import zipfile

import numpy as np

URL = "https://huggingface.co/datasets/SprintML/unlearning/resolve/main/model.pt"
BLOCK = 1 << 16
WANTED = ("head.weight", "head.bias", "norm.weight", "norm.bias")
DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}
NINE = (0, 6, 7, 8, 29, 30, 31, 33, 38)


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, get):
        self.get = get
        probe = get(url, {"Range": "bytes=0-0"})
        self.url = probe.url
        self.size = int(probe.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.blocks: dict[int, bytes] = {}
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def block(self, index: int) -> bytes:
        if index not in self.blocks:
            start = index * BLOCK
            end = min(self.size, start + BLOCK) - 1
            reply = self.get(self.url, {"Range": f"bytes={start}-{end}"})
            if reply.status_code != 206:
                raise RuntimeError(f"range request answered {reply.status_code}")
            self.blocks[index] = reply.content
            self.fetched += len(reply.content)
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        n = max(0, min(len(buffer), self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            index, offset = divmod(self.pos, BLOCK)
            chunk = self.block(index)[offset : offset + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        buffer[:n] = out
        return n


class TensorIndex(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *_: {
                "storage": storage,
                "offset": offset,
                "size": tuple(size),
                "stride": tuple(stride),
            }
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")

    def persistent_load(self, pid):
        _, kind, key, _, numel = pid
        return {"kind": kind, "key": key, "numel": numel}


def read_tensor(archive: zipfile.ZipFile, prefix: str, entry: dict) -> np.ndarray:
    storage = entry["storage"]
    flat = np.frombuffer(archive.read(f"{prefix}/data/{storage['key']}"), DTYPES[storage["kind"]])
    view = np.lib.stride_tricks.as_strided(
        flat[entry["offset"] :],
        shape=entry["size"],
        strides=tuple(s * flat.itemsize for s in entry["stride"]),
    )
    return view.astype(np.float64)


def head_statistics(
    weight: np.ndarray, bias: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> dict[str, float]:
    norms = np.linalg.norm(weight, axis=1)
    mean_row = weight.mean(axis=0)
    scaled = mean_row * gamma
    nine = np.zeros(len(bias), dtype=bool)
    nine[list(NINE)] = True
    return {
        "nine_bias": float(bias[nine].mean()),
        "other_bias": float(bias[~nine].mean()),
        "nine_norm": float(norms[nine].mean()),
        "other_norm": float(norms[~nine].mean()),
        "bias_mean": float(bias.mean()),
        "bias_sd": float(bias.std()),
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "mean_row_ratio": float(np.linalg.norm(mean_row) / norms.mean()),
        "chance_ratio": float(1 / np.sqrt(len(bias))),
        "mean_logit_centre": float(bias.mean() + mean_row @ beta),
        "mean_logit_radius": float(np.sqrt(len(gamma)) * np.linalg.norm(scaled - scaled.mean())),
    }


def main() -> None:
    import requests

    session = requests.Session()
    reader = RangeReader(URL, lambda url, headers: session.get(url, headers=headers, timeout=60))
    with zipfile.ZipFile(io.BufferedReader(reader, buffer_size=BLOCK)) as archive:
        pickled = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pickled[: -len("/data.pkl")]
        index = TensorIndex(io.BytesIO(archive.read(pickled))).load()
        tensors = {name: read_tensor(archive, prefix, index[name]) for name in WANTED}
    stats = head_statistics(
        tensors["head.weight"], tensors["head.bias"], tensors["norm.weight"], tensors["norm.bias"]
    )
    print(f"model.pt {reader.size:,} bytes, {len(index)} tensors, fetched {reader.fetched:,} bytes")
    print(f"last keys: {list(index)[-4:]}")
    print(
        f"head bias: mean {stats['bias_mean']:+.4f}, sd {stats['bias_sd']:.4f}, "
        f"range {stats['bias_min']:+.4f} .. {stats['bias_max']:+.4f}"
    )
    print(
        f"norm of the class-mean weight row / mean row norm: {stats['mean_row_ratio']:.4f} "
        f"(rows with no common direction: about {stats['chance_ratio']:.3f})"
    )
    print(
        f"mean logit over the 100 classes, any input: {stats['mean_logit_centre']:+.4f} "
        f"+/- {stats['mean_logit_radius']:.4f}"
    )
    print(
        f"nine 130-image classes: bias {stats['nine_bias']:+.4f}, "
        f"row norm {stats['nine_norm']:.4f}; other classes: bias {stats['other_bias']:+.4f}, "
        f"row norm {stats['other_norm']:.4f}"
    )


if __name__ == "__main__":
    main()
