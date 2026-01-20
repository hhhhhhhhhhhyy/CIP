import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


ND_RE = re.compile(r"\(ND_LIST:\s*([^\)]+)\)")
STEP_RE = re.compile(r"step\s*number\s*[:：\-]?\s*(\d+)", re.IGNORECASE)


def _norm_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def _extract_nd_tag(ai_reason: str) -> str:
    m = ND_RE.search(ai_reason or "")
    return (m.group(1).strip() if m else "")


def _extract_step(ai_reason: str) -> str:
    m = STEP_RE.search(ai_reason or "")
    return (m.group(1) if m else "")


def _short(s: str, n: int = 160) -> str:
    s = _norm_str(s)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _categorize_mismatch(human_reason: str, ai_reason: str, human: str, ai: str, nd_tag: str) -> str:
    hr = _norm_str(human_reason).lower()
    ar = _norm_str(ai_reason).lower()
    h = _norm_str(human)
    a = _norm_str(ai)
    nd = _norm_str(nd_tag).lower()

    if h == "Spam" or a == "Spam":
        return "spam"

    # ND_LIST-driven categories first
    if nd.startswith("nd07") or "temporal" in nd or "continuity" in nd:
        return "temporal_or_continuity"
    if nd.startswith("nd03") or "hover" in nd or "clickability" in nd:
        return "hover_or_clickability"
    if nd.startswith("nd08") or "precondition" in nd or "conditional" in nd:
        return "precondition_unknown"
    if nd.startswith("nd10") or "cross_step" in nd or "dependency" in nd:
        return "cross_step_dependency"
    if nd.startswith("nd13") or "insufficient" in nd or "visibility" in nd:
        return "insufficient_visibility"

    # Keyword-driven categories
    kw = hr + "\n" + ar
    if any(k in kw for k in ["断网", "离线", "offline", "network", "wifi", "internet"]):
        return "network_offline"
    if any(k in kw for k in ["下滑", "滚动", "scroll", "swipe"]):
        return "scroll"
    if any(k in kw for k in ["点击", "单击", "click", "tap", "press"]):
        return "click_tap"
    if any(k in kw for k in ["悬停", "hover", "mouse over"]):
        return "hover"
    if any(k in kw for k in ["等待", "wait", "loading", "load", "刷新", "refresh"]):
        return "wait_load_refresh"
    if any(k in kw for k in ["无关", "错误页面", "wrong page", "redirect", "跳转", "walmart.com"]):
        return "wrong_page_or_navigation"
    if any(k in kw for k in ["开关", "toggle", "switch", "on/off", "打开", "关闭"]):
        return "toggle_state"
    if any(k in kw for k in ["数值", "value", "文本", "label", "标题", "title", "名称", "name"]):
        return "value_or_label"

    return "other"


def _summary_tags_from_human_reason(human_reason: str) -> List[str]:
    """Heuristically derive what should be mentioned in ai_summary.

    This is intentionally simple and keyword-based. We use it to measure whether
    ai_summary addresses the same core evidence the human_reason cites.
    """
    hr = _norm_str(human_reason)
    low = hr.lower()
    tags: List[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if any(k in hr for k in ["截图", "图片", "例图", "圈", "标记"]):
        add("visual_evidence")

    if any(k in hr for k in ["相同", "一样", "完全一样", "没有任何操作", "没有进行操作", "无意义", "无操作"]):
        add("spam_like_no_progress")

    if any(k in hr for k in ["加载", "loading", "刷新", "转圈", "spinner"]):
        add("loading_only")

    if any(k in hr for k in ["无关", "错误页面", "跳转", "转到", "wrong page", "redirect"]):
        add("wrong_page")

    if any(k in hr for k in ["下滑", "滚动", "scroll"]):
        add("scroll")

    if any(k in hr for k in ["点击", "单击", "tap", "press", "click"]):
        add("click")

    if any(k in hr for k in ["悬停", "hover"]):
        add("hover")

    if any(k in hr for k in ["断网", "离线", "offline", "网络", "wifi", "internet"]):
        add("network_offline")

    if any(k in hr for k in ["等待", "30秒", "分钟", "wait"]):
        add("timing")

    if any(k in hr for k in ["宽度", "像素", "px", "分辨率", "1920", "1100"]):
        add("measurement")

    if any(k in hr for k in ["多步骤", "多个步骤", "分开验证", "分别验证", "逐个验证"]):
        add("multi_action_split_verify")

    return tags


def _ai_summary_mentions_tags(ai_summary: str, tags: List[str]) -> Tuple[bool, List[str]]:
    s = _norm_str(ai_summary)
    low = s.lower()
    missing: List[str] = []

    # For each tag, we accept either Chinese or English mentions.
    patterns: Dict[str, List[str]] = {
        "spam_like_no_progress": [
            "all screenshots", "same", "identical", "no evidence", "no meaningful", "no action", "static", "unchanged",
            "new tab", "msn", "feed", "desktop only", "没有操作", "无操作", "截图相同", "图片相同",
        ],
        "loading_only": ["loading", "spinner", "refresh", "still loading", "加载", "转圈", "刷新"],
        "wrong_page": ["wrong page", "unrelated", "redirect", "different page", "无关", "错误页面", "跳转"],
        "scroll": ["scroll", "swipe", "down", "下滑", "滚动"],
        "click": ["click", "tap", "press", "select", "点击", "单击"],
        "hover": ["hover", "mouse over", "悬停"],
        "network_offline": ["offline", "network", "wifi", "internet", "disconnect", "断网", "离线"],
        "timing": ["wait", "seconds", "minutes", "等待", "秒", "分钟"],
        "measurement": ["px", "pixel", "resolution", "width", "1920", "1100", "像素", "分辨率", "宽度"],
        "multi_action_split_verify": [
            "multiple", "each", "for each", "separately", "all of the following", "repeat",
            "分开验证", "分别验证", "逐个", "多个步骤", "多步骤",
        ],
        "visual_evidence": ["screenshot", "visible", "shown", "not shown", "截图", "可见", "未显示"],
    }

    for t in tags:
        pats = patterns.get(t)
        if not pats:
            continue
        if not any(p in low or p in s for p in pats):
            missing.append(t)

    ok = len(missing) == 0
    return ok, missing


def _compute_confusion(df: pd.DataFrame, human_col: str, ai_col: str) -> pd.DataFrame:
    labels = ["Correct", "Incorrect", "NeedDiscussion", "Spam", "Error", "Skipped"]
    human = df[human_col].fillna("").astype(str).str.strip()
    ai = df[ai_col].fillna("").astype(str).str.strip()

    human = human.replace({"spam": "Spam"})
    ai = ai.replace({"spam": "Spam"})

    human = human.where(human.isin(labels), other="Error")
    ai = ai.where(ai.isin(labels), other="Error")
    return pd.crosstab(human, ai, rownames=["human"], colnames=["ai"], dropna=False)


@dataclass
class ReportPaths:
    md: str
    clusters_tsv: str
    mismatches_tsv: str


def _derive_outputs(input_path: str, out_md: str, out_clusters_tsv: str, out_mismatches_tsv: str) -> ReportPaths:
    base = os.path.splitext(input_path)[0]

    if not out_md:
        out_md = base + "_report.md"
    if not out_clusters_tsv:
        out_clusters_tsv = base + "_mismatch_clusters.tsv"
    if not out_mismatches_tsv:
        out_mismatches_tsv = base + "_mismatches_enriched.tsv"

    return ReportPaths(md=out_md, clusters_tsv=out_clusters_tsv, mismatches_tsv=out_mismatches_tsv)


def build_report(df: pd.DataFrame, title: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    for c in [
        "link",
        "type",
        "human_result",
        "human_reason",
        "page_issue_type",
        "ai_result",
        "ai_reason",
        "ai_summary",
        "error",
    ]:
        if c in df.columns:
            df[c] = df[c].map(_norm_str)

    df["nd_tag"] = df["ai_reason"].map(_extract_nd_tag)
    df["step_num"] = df["ai_reason"].map(_extract_step)

    total = len(df)
    unique_links = df["link"].nunique(dropna=True) if "link" in df.columns else 0

    ai_counts = df["ai_result"].value_counts(dropna=False)
    issue_counts = df["page_issue_type"].value_counts(dropna=False)

    # Health stats
    num_error = int((df["ai_result"] == "Error").sum())
    num_skipped = int((df["ai_result"] == "Skipped").sum())
    num_done = total - num_error - num_skipped

    cm_all = _compute_confusion(df, "human_result", "ai_result")

    # “evaluated” subset: exclude Skipped
    eval_df = df[df["ai_result"] != "Skipped"].copy()
    cm_eval = _compute_confusion(eval_df, "human_result", "ai_result")
    eval_total = len(eval_df)
    eval_acc = float((eval_df["human_result"] == eval_df["ai_result"]).mean()) if eval_total else 0.0

    # Mismatches (excluding Skipped)
    mism = eval_df[eval_df["human_result"] != eval_df["ai_result"]].copy()
    mism["mismatch_category"] = mism.apply(
        lambda r: _categorize_mismatch(
            r.get("human_reason", ""),
            r.get("ai_reason", ""),
            r.get("human_result", ""),
            r.get("ai_result", ""),
            r.get("nd_tag", ""),
        ),
        axis=1,
    )

    # ai_summary coverage vs human_reason (heuristic)
    if "ai_summary" in mism.columns:
        mism["summary_expected_tags"] = mism["human_reason"].map(_summary_tags_from_human_reason)
        cov = mism.apply(
            lambda r: _ai_summary_mentions_tags(r.get("ai_summary", ""), r.get("summary_expected_tags", [])),
            axis=1,
        )
        mism["summary_covered"] = cov.map(lambda x: bool(x[0]))
        mism["summary_missing_tags"] = cov.map(lambda x: ",".join(x[1]) if x[1] else "")
    else:
        mism["summary_expected_tags"] = [[] for _ in range(len(mism))]
        mism["summary_covered"] = True
        mism["summary_missing_tags"] = ""

    # Aggregations
    top_pairs = (
        mism.groupby(["human_result", "ai_result"]).size().sort_values(ascending=False).head(15)
    )
    top_cats = mism["mismatch_category"].value_counts().head(15)
    top_nd = mism["nd_tag"].replace({"": "(none)"}).value_counts().head(15)

    # Cluster table
    cluster = (
        mism.groupby(["page_issue_type", "human_result", "ai_result", "mismatch_category", "nd_tag"]).agg(
            count=("link", "size"),
            example_link=("link", "first"),
            example_human_reason=("human_reason", lambda s: _short(s.iloc[0]) if len(s) else ""),
            example_ai_reason=("ai_reason", lambda s: _short(s.iloc[0]) if len(s) else ""),
        )
    ).reset_index().sort_values(["count"], ascending=False)

    md_lines: List[str] = []
    md_lines.append(f"# {title}")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append(f"- rows: {total}")
    md_lines.append(f"- unique links: {unique_links}")
    md_lines.append(f"- evaluated rows (ai_result != Skipped): {eval_total}")
    md_lines.append(f"- exact-match accuracy (evaluated): {eval_acc:.3f}")
    md_lines.append(f"- ai_result=Skipped: {num_skipped}")
    md_lines.append(f"- ai_result=Error: {num_error}")
    md_lines.append("")

    md_lines.append("## page_issue_type distribution")
    md_lines.append("```text")
    md_lines.append(issue_counts.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## ai_result distribution")
    md_lines.append("```text")
    md_lines.append(ai_counts.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Confusion matrix (all rows; includes Skipped/Error)")
    md_lines.append("```text")
    md_lines.append(cm_all.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Confusion matrix (evaluated only; ai_result != Skipped)")
    md_lines.append("```text")
    md_lines.append(cm_eval.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Top mismatch pairs (evaluated only)")
    md_lines.append("```text")
    md_lines.append(top_pairs.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Top mismatch categories (heuristic)")
    md_lines.append("```text")
    md_lines.append(top_cats.to_string())
    md_lines.append("```")
    md_lines.append("")

    if "ai_summary" in df.columns:
        md_lines.append("## AI summary coverage (mismatches; heuristic)")
        if mism.empty:
            md_lines.append("(none)")
        else:
            covered_rate = float(mism["summary_covered"].mean()) if len(mism) else 1.0
            md_lines.append(f"- summary_covered rate: {covered_rate:.3f} (1.0 means ai_summary mentions all key tags inferred from human_reason)")

            miss_counts = (
                mism[mism["summary_missing_tags"].ne("")]["summary_missing_tags"]
                .str.split(",")
                .explode()
                .replace({"": None})
                .dropna()
                .value_counts()
                .head(15)
            )
            if not miss_counts.empty:
                md_lines.append("\nMost common missing tags:")
                md_lines.append("```text")
                md_lines.append(miss_counts.to_string())
                md_lines.append("```")

            # Show a small sample of low-coverage mismatches
            low_cov = mism[mism["summary_covered"] == False].head(12).copy()  # noqa: E712
            if not low_cov.empty:
                low_cov_view = low_cov[
                    [
                        "page_issue_type",
                        "human_result",
                        "ai_result",
                        "mismatch_category",
                        "summary_missing_tags",
                        "link",
                        "human_reason",
                        "ai_summary",
                    ]
                ].copy()
                low_cov_view["human_reason"] = low_cov_view["human_reason"].map(_short)
                low_cov_view["ai_summary"] = low_cov_view["ai_summary"].map(lambda x: _short(x, 220))
                md_lines.append("\nLow-coverage samples (first 12):")
                md_lines.append("```text")
                md_lines.append(low_cov_view.to_string(index=False))
                md_lines.append("```")

        md_lines.append("")

    md_lines.append("## Top ND_LIST tags inside mismatches")
    md_lines.append("```text")
    md_lines.append(top_nd.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Sample mismatches (first 25)")
    if mism.empty:
        md_lines.append("(none)")
    else:
        sample = mism.head(25)[
            [
                "page_issue_type",
                "human_result",
                "ai_result",
                "mismatch_category",
                "step_num",
                "nd_tag",
                "link",
                "human_reason",
                "ai_reason",
            ]
        ].copy()
        sample["human_reason"] = sample["human_reason"].map(_short)
        sample["ai_reason"] = sample["ai_reason"].map(_short)
        md_lines.append("```text")
        md_lines.append(sample.to_string(index=False))
        md_lines.append("```")

    md_lines.append("")
    md_lines.append("## Prompt tuning direction (actionable)")

    # A few opinionated next steps keyed off the most common mismatch shapes
    pair_counts = mism.groupby(["human_result", "ai_result"]).size().sort_values(ascending=False)
    inc_to_nd = int(pair_counts.get(("Incorrect", "NeedDiscussion"), 0))
    spam_to_other = int(mism[(mism["human_result"] == "Spam") & (mism["ai_result"] != "Spam")].shape[0])
    nd_to_inc = int(pair_counts.get(("NeedDiscussion", "Incorrect"), 0))

    summary_low = int((mism.get("summary_covered") == False).sum()) if len(mism) else 0  # noqa: E712

    bullets: List[str] = []
    if inc_to_nd:
        bullets.append(
            f"- Human=Incorrect -> AI=NeedDiscussion ({inc_to_nd}): 对明显可见的失败收紧 ND 门槛（例如：明确缺少必需 UI、明显在错误页面/错误区域、明显未执行关键点击/滑动/切换）。"
        )
    if nd_to_inc:
        bullets.append(
            f"- Human=NeedDiscussion -> AI=Incorrect ({nd_to_inc}): 对‘天然无法证明’场景加硬规则（键盘/鼠标悬停/时间连续性/多步骤依赖/截图不可见），避免误判成 Incorrect。"
        )
    if spam_to_other:
        bullets.append(
            f"- Human=Spam -> AI!=Spam ({spam_to_other}): 把‘截图完全重复/无进展/未进入任务入口’写成更决定性的 Spam 规则，并放到决策最前。"
        )
    if summary_low:
        bullets.append(
            f"- ai_summary 覆盖不足的 mismatch ({summary_low}): 在 prompt 强调：当 human_reason 常见点是‘截图相同/加载页/无操作/多步骤需分开验证’时，ai_summary 必须明确点出这些证据。"
        )
    if not bullets:
        bullets.append("- 当前样本上 mismatch 不集中；建议优先看 cluster 表里 count 最高的 3–5 类，逐条把规则写成一句话并保持决策顺序不变。")

    md_lines.extend(bullets)
    md_lines.append("")

    return "\n".join(md_lines), cluster, mism


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze CIP AI results TSV and produce a report + mismatch clusters.")
    parser.add_argument("--input", default="data/result/cip_sample_result.tsv", help="Input TSV (AI-augmented)")
    parser.add_argument("--out-md", default="", help="Output markdown path (default: <input>_report.md)")
    parser.add_argument("--out-clusters-tsv", default="", help="Output clusters TSV path (default: <input>_mismatch_clusters.tsv)")
    parser.add_argument("--out-mismatches-tsv", default="", help="Output enriched mismatches TSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    outputs = _derive_outputs(args.input, args.out_md, args.out_clusters_tsv, args.out_mismatches_tsv)

    report_all, clusters_all, mism_all = build_report(df, title="CIP human vs AI report (overall)")

    noi = df[df.get("page_issue_type", "").astype(str).str.strip().eq("No Issue found")].copy()
    report_noi, clusters_noi, mism_noi = build_report(noi, title="CIP human vs AI report (No Issue found only)")

    os.makedirs(os.path.dirname(outputs.md) or ".", exist_ok=True)
    with open(outputs.md, "w", encoding="utf-8") as f:
        f.write(report_all)
        f.write("\n\n---\n\n")
        f.write(report_noi)

    clusters_all.to_csv(outputs.clusters_tsv, sep="\t", index=False, encoding="utf-8-sig")

    # Save enriched mismatches (overall evaluated mismatches)
    mism_all.to_csv(outputs.mismatches_tsv, sep="\t", index=False, encoding="utf-8-sig")

    print(f"Saved report: {outputs.md}")
    print(f"Saved mismatch clusters: {outputs.clusters_tsv}")
    print(f"Saved enriched mismatches: {outputs.mismatches_tsv}")

    # Also save NOI-only cluster table next to the main clusters file
    base, ext = os.path.splitext(outputs.clusters_tsv)
    noi_clusters_path = base + "_noi" + ext
    clusters_noi.to_csv(noi_clusters_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"Saved NOI mismatch clusters: {noi_clusters_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
