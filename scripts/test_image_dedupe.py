import argparse
import base64
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.tools.image_quality import (
    duplicate_pairs_to_groups,
    find_duplicates_in_items,
    find_duplicate_indices_in_base64_list,
    find_duplicate_indices_in_url_list,
    merge_overlapping_groups,
)


def _to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base64", "url"], default="base64")
    args = parser.parse_args()

    # Create 3 images: 0 and 2 are identical; 1 is different
    img0 = Image.new("RGB", (64, 64), (255, 0, 0))
    img1 = Image.new("RGB", (64, 64), (0, 255, 0))
    img2 = Image.new("RGB", (64, 64), (255, 0, 0))

    if args.mode == "base64":
        b64_0 = _to_b64_png(img0)
        b64_1 = _to_b64_png(img1)
        b64_2 = _to_b64_png(img2)

        # Mixed: data URL + raw base64
        items = [
            f"data:image/png;base64,{b64_0}",
            b64_1,
            b64_2,
        ]
        dup_0 = find_duplicate_indices_in_base64_list(items)
        dup_1 = find_duplicate_indices_in_base64_list(items, one_based=True)
    else:
        # file:// URLs (simulate real URL list)
        p0 = ROOT / "llm" / "tools" / "video-test" / "data" / "trains" / "train1.png"
        p1 = ROOT / "llm" / "tools" / "video-test" / "data" / "trains" / "train2.png"
        items = [p0.as_uri(), p1.as_uri(), p0.as_uri()]
        dup_0 = find_duplicate_indices_in_url_list(items)
        dup_1 = find_duplicate_indices_in_url_list(items, one_based=True)

    hashes, duplicates = find_duplicates_in_items(items)
    groups = duplicate_pairs_to_groups(duplicates)
    merged = merge_overlapping_groups(groups)

    print(f"mode: {args.mode}")
    print("duplicates (dup_index, first_index):", duplicates)
    print("groups:", groups)
    print("merged groups:", merged)
    print("duplicate indices (0-based):", dup_0)
    print("duplicate indices (1-based):", dup_1)


if __name__ == "__main__":
    main()
