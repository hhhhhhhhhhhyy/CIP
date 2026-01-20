import argparse
import re
import csv
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from zipfile import BadZipFile

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def norm(name: object) -> str:
        s = str(name)
        # Normalize unicode spaces and repeated whitespace
        s = s.replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    df.columns = [norm(c) for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )


def _try_read_excel(path: Path, password: str = "") -> pd.DataFrame:
    """Read an Excel file that may be a real .xlsx, legacy .xls, or an encrypted Office file (OLE + EncryptedPackage)."""
    # 1) Try as real xlsx first
    try:
        return pd.read_excel(path, engine="openpyxl")
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied while reading {path}. The file is likely open/locked by Excel or another process (or blocked by OneDrive sync). "
            "Please close the file and retry."
        ) from e
    except BadZipFile:
        # Not a zip-based xlsx
        pass
    except Exception:
        # Could still be xls or package; fall through
        pass

    # 2) Try legacy xls via xlrd
    try:
        return pd.read_excel(path, engine="xlrd")
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied while reading {path}. The file is likely open/locked by Excel or another process (or blocked by OneDrive sync). "
            "Please close the file and retry."
        ) from e
    except Exception:
        pass

    # 3) Try as encrypted Office file / OLE container
    try:
        import olefile  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to read Excel via openpyxl/xlrd, and olefile is not installed. "
            "Install it with: pip install olefile"
        ) from e

    try:
        is_ole = olefile.isOleFile(str(path))
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied while reading {path}. The file is likely open/locked by Excel or another process (or blocked by OneDrive sync). "
            "Please close the file and retry."
        ) from e

    if not is_ole:
        raise ValueError(f"Unsupported input file format (not xlsx/xls/OLE): {path}")

    with olefile.OleFileIO(str(path)) as ole:
        streams = {"/".join(s) for s in ole.listdir(streams=True, storages=False)}

    # Encrypted Office (OOXML) typically shows these streams.
    if "EncryptedPackage" in streams:
        try:
            import msoffcrypto  # type: ignore
        except Exception as e:
            raise ImportError(
                "Input looks like an encrypted Office file (EncryptedPackage). Install dependency with: pip install msoffcrypto-tool"
            ) from e

        try:
            with path.open("rb") as f:
                office = msoffcrypto.OfficeFile(f)
                office.load_key(password=password or "")
                decrypted = BytesIO()
                office.decrypt(decrypted)
            decrypted.seek(0)
            return pd.read_excel(decrypted, engine="openpyxl")
        except Exception as e:
            # Some org-protected files are IRM/DRM (DataSpaces/DRMEncrypted*) and are not decryptable by msoffcrypto.
            err_name = e.__class__.__name__
            if err_name == "FileFormatError":
                raise ValueError(
                    f"{path} is an OLE container with EncryptedPackage but is not a password-encrypted Office file that msoffcrypto can decrypt. "
                    "It may be IRM/DRM/sensitivity-label protected. Please open it in Excel and export/save as an unencrypted .xlsx, or export as .tsv, then rerun this script."
                ) from e

            raise ValueError(
                f"{path} appears to be an encrypted .xlsx. Provide the password via --password, or re-export it as an unencrypted .xlsx/.tsv. Error: {e}"
            ) from e

    # Fallback: OLE Package embedded file (rare for .xlsx naming)
    raise ValueError(
        f"{path} is an OLE compound file but not a readable Excel workbook with current tooling. Streams: {sorted(list(streams))}. "
        "If it's protected/encrypted, export as plaintext .xlsx/.tsv or provide password if applicable."
    )


def _guess_delimiter(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(4096)
        if sample.count("\t") > sample.count(","):
            return "\t"
        # If csv.Sniffer can decide, trust it
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            return dialect.delimiter
        except Exception:
            return ","
    except Exception:
        return ","


def _try_read_csv_like(path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "utf-8", "gb18030"]
    sep = _guess_delimiter(path)

    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                dtype=object,
                keep_default_na=True,
            )
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied while reading {path}. The file is likely open/locked by another process (or blocked by OneDrive sync). "
                "Please close the file and retry."
            ) from e
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Failed to read CSV/TSV file: {path}. Last error: {last_err}")


def _read_input(path: Path, password: str = "") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return _try_read_excel(path, password=password)
    if suffix in {".csv", ".tsv", ".txt"}:
        return _try_read_csv_like(path)
    # Best-effort: try csv-like first, then excel
    try:
        return _try_read_csv_like(path)
    except Exception:
        return _try_read_excel(path, password=password)


def main() -> int:
    parser = argparse.ArgumentParser(description="Process CIP sample file (csv/tsv/xlsx) into data/cip_sample.tsv.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "cip_sample.csv"),
        help="Input file path (default: data/cip_sample.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "cip_sample.tsv"),
        help="Output TSV path (default: data/cip_sample.tsv)",
    )
    parser.add_argument(
        "--password",
        default="",
        help="Optional password for encrypted Excel files (default: empty)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _read_input(input_path, password=str(args.password))

    df = _normalize_columns(df)

    # Required source columns
    src_permalink = "permalink"
    src_type = "类型"
    src_vendor = "vendor judgement"
    src_reason = "结果分析"

    _require_columns(df, [src_permalink, src_type, src_vendor, src_reason])

    out = df[[src_permalink, src_type, src_vendor, src_reason]].copy()

    # Trim permalink and normalize to string
    out[src_permalink] = out[src_permalink].fillna("").astype(str).str.strip()

    # Drop empty rows (based on permalink)
    out = out[out[src_permalink].astype(str).str.strip() != ""].copy()

    # Rename columns
    out = out.rename(
        columns={
            src_permalink: "link",
            src_type: "type",
            src_vendor: "human_result",
            src_reason: "human_reason",
        }
    )

    # Final trims
    out["link"] = out["link"].fillna("").astype(str).str.strip()
    out["type"] = out["type"].fillna("").astype(str).str.strip()
    out["human_result"] = out["human_result"].fillna("").astype(str).str.strip()
    # Normalize Spam/spam to a single canonical value
    out.loc[out["human_result"].str.lower().eq("spam"), "human_result"] = "Spam"
    out["human_reason"] = out["human_reason"].fillna("").astype(str).str.strip()

    # Re-drop any rows that became empty after normalization
    out = out[out["link"].ne("")].copy()

    # 按 link 去重：保留 first 行
    out_dedup = out.drop_duplicates(subset=["link"], keep="first").copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_dedup.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    print(f"Input rows: {len(df)}")
    print(f"Output rows (non-empty link): {len(out)}")
    print(f"Output rows (dedup by link, keep first): {len(out_dedup)}")
    print(f"Saved cleaned TSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())