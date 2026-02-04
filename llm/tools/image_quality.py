import base64
import io
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import url2pathname

import cv2
import imagehash
import numpy as np
from PIL import Image

def video_to_image(path, floder):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{floder}/frame{count:05d}.jpg", image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+)?;base64,(?P<b64>.*)$", re.IGNORECASE | re.DOTALL)

def _looks_like_base64(s: str) -> bool:
    v = (s or "").strip()
    if not v:
        return False

    if len(v) < 64:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=\s]+", v) is not None


def load_image_any(image_input: Any) -> Image.Image:

    if image_input is None:
        raise ValueError("image_input is None")

    if isinstance(image_input, Image.Image):
        return image_input

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Assume BGR (opencv) and convert to RGB.
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image_input)))

    if isinstance(image_input, Path):
        return Image.open(str(image_input))

    if isinstance(image_input, str):
        s = image_input.strip()
        if not s:
            raise ValueError("empty image string")

        if s.startswith("http://") or s.startswith("https://"):
            import requests

            resp = requests.get(s, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))

        if s.startswith("file:"):
            parsed = urlparse(s)

            local_path = url2pathname(parsed.path)
            if re.match(r"^/[A-Za-z]:", parsed.path):
                local_path = local_path.lstrip("\\/")
            return Image.open(local_path)

        m = _DATA_URL_RE.match(s)
        if m:
            b64 = (m.group("b64") or "").strip()
            raw = base64.b64decode(b64, validate=False)
            return Image.open(io.BytesIO(raw))

        if _looks_like_base64(s):
            raw = base64.b64decode(s, validate=False)
            return Image.open(io.BytesIO(raw))

        return Image.open(s)

    raise TypeError(f"Unsupported image_input type: {type(image_input)}")


def phash_image(image_input: Any):
    img = load_image_any(image_input)
    return imagehash.phash(img)


def find_duplicates(folder):
    hashes = {}
    duplicates = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(folder, filename)
            h = phash_image(full_path)

            if h in hashes:
                duplicates.append((filename, hashes[h]))
            else:
                hashes[h] = filename

    return duplicates

def remove_duplicates(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hashes = {}
    duplicates = []

    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(input_folder, filename)

            try:
                h = str(phash_image(full_path))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            if h in hashes:
                duplicates.append((filename, hashes[h]))
            else:
                hashes[h] = filename
                shutil.copy(full_path, os.path.join(output_folder, filename))

    return hashes, duplicates


def find_duplicates_in_items(items: Iterable[Any]):

    first_by_hash: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []

    for idx, item in enumerate(items):
        try:
            h = str(phash_image(item))
        except Exception as e:
            raise ValueError(f"Failed to hash item[{idx}]: {e}") from e

        if h in first_by_hash:
            duplicates.append((idx, first_by_hash[h]))
        else:
            first_by_hash[h] = idx

    return first_by_hash, duplicates


def duplicate_pairs_to_groups(duplicates: list[tuple[int, int]]):

    groups: dict[int, list[int]] = {}
    for dup_idx, first_idx in duplicates:
        g = groups.setdefault(int(first_idx), [int(first_idx)])
        g.append(int(dup_idx))

    out = []
    for k in sorted(groups.keys()):
        out.append(sorted(set(groups[k])))
    return out


def merge_overlapping_groups(groups: list[list[int]]) -> list[list[int]]:

    normalized = []
    for g in groups or []:
        if not g:
            continue
        normalized.append(sorted(set(int(x) for x in g)))
    if not normalized:
        return []

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g in normalized:
        root = g[0]
        for x in g[1:]:
            union(root, x)

    buckets: dict[int, set[int]] = {}
    for g in normalized:
        for x in g:
            r = find(x)
            buckets.setdefault(r, set()).add(x)

    merged = [sorted(v) for v in buckets.values()]
    merged.sort(key=lambda xs: (len(xs), xs[0] if xs else 10**9))
    return merged


def find_duplicate_groups_in_items(items: Iterable[Any]):
    _, duplicates = find_duplicates_in_items(items)
    return duplicate_pairs_to_groups(duplicates)


def find_duplicate_indices_in_items(items: Iterable[Any], *, one_based: bool = False) -> list[int]:

    _, duplicates = find_duplicates_in_items(items)
    dup_indices = [dup_idx for dup_idx, _ in duplicates]
    if one_based:
        return [i + 1 for i in dup_indices]
    return dup_indices


def find_duplicate_indices_in_base64_list(images_base64: list[str], *, one_based: bool = False) -> list[int]:

    return find_duplicate_indices_in_items(images_base64, one_based=one_based)


def find_duplicate_indices_in_url_list(image_urls: list[str], *, one_based: bool = False) -> list[int]:

    return find_duplicate_indices_in_items(image_urls, one_based=one_based)


def dedupe_items(items: Iterable[Any]):

    items_list = list(items)
    first_by_hash, duplicates = find_duplicates_in_items(items_list)
    dup_set = {dup_idx for dup_idx, _ in duplicates}
    unique_items = [v for i, v in enumerate(items_list) if i not in dup_set]
    return unique_items, duplicates


if __name__ == "__main__":
    video_path = "/home/shawn/CIP/video-test/data/case.webm"
    image_folder = "/home/shawn/CIP/video-test/data/video_image"
    # video_to_image(video_path, image_folder)
    # dups = find_duplicates(image_folder)
    # print("重复图片:")
    # for dup in dups:
    #     print(dup)
    remove_duplicates(image_folder, "/home/shawn/CIP/video-test/data/video_image_no_dup")