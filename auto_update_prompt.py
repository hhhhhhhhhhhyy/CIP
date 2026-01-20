import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from enums.issue_enum import IssueEnum
from parsers.document_parser import extract_steps_from_left_pane, extract_steps_from_right_pane
from llm.image_to_steps_check import compare_operations


@dataclass
class Job:
    row_id: int
    link: str
    human_result: str
    human_reason: str


@dataclass
class JobResult:
    row_id: int
    link: str
    page_issue_type: str
    ai_result: str
    ai_reason: str
    ai_summary: str
    error: str


def normalize_issue_type(raw: str) -> str:
    s = (raw or "").strip()
    s = " ".join(s.split())
    low = s.lower()

    # Be robust to extra prefix/suffix text from UI.
    if "no issue" in low and "found" in low:
        return IssueEnum.NO_ISSUE_FOUND.value
    if "feature" in low and "not" in low and "found" in low:
        return IssueEnum.FEATURE_NOT_FOUND.value
    if "issue" in low and "found" in low:
        return IssueEnum.ISSUE_FOUND.value

    return s


def try_click_sign_in(driver: webdriver.Edge, timeout_s: int = 3) -> None:
    """Best-effort: sign-in button may or may not exist depending on session/cookies."""
    try:
        sign_in_button = WebDriverWait(driver, timeout_s).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
        )
        sign_in_button.click()
    except TimeoutException:
        return
    except Exception:
        return


def _wait_for_case_loaded(driver: webdriver.Edge) -> None:
    wait = WebDriverWait(driver, 30)
    try:
        wait.until(EC.presence_of_element_located((By.ID, "leftPane")))
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col")))
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "textColorRed")))
    except TimeoutException:
        # One more attempt: sometimes sign-in renders late.
        try_click_sign_in(driver, timeout_s=20)
        wait.until(EC.presence_of_element_located((By.ID, "leftPane")))
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col")))
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "textColorRed")))


def open_case_and_get_issue_type(driver: webdriver.Edge, page_url: str) -> str:
    driver.get(page_url)

    # Sign-in might be required for content; try but do not fail hard.
    try_click_sign_in(driver, timeout_s=5)
    _wait_for_case_loaded(driver)

    issue_type_raw = driver.find_element(By.CLASS_NAME, "textColorRed").text
    return normalize_issue_type(issue_type_raw)


def extract_case_steps(driver: webdriver.Edge) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    standard_steps = extract_steps_from_left_pane(driver)
    judge_comment, actual_steps = extract_steps_from_right_pane(driver)
    return standard_steps, actual_steps, judge_comment


def fetch_case_data(driver: webdriver.Edge, page_url: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Fetch issue_type, standard_steps, actual_steps, judge_comment from the permalink page."""
    issue_type = open_case_and_get_issue_type(driver, page_url)
    standard_steps, actual_steps, judge_comment = extract_case_steps(driver)
    return issue_type, standard_steps, actual_steps, judge_comment


class Worker(threading.Thread):
    def __init__(
        self,
        name: str,
        job_queue: Queue,
        results: List[JobResult],
        results_lock: threading.Lock,
        progress: Optional[Dict[str, Any]],
        progress_lock: Optional[threading.Lock],
        page_timeout_s: int,
        retry: int,
        slow_mo_s: float,
        only_no_issue_found: bool,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.job_queue = job_queue
        self.results = results
        self.results_lock = results_lock
        self.progress = progress
        self.progress_lock = progress_lock
        self.page_timeout_s = page_timeout_s
        self.retry = retry
        self.slow_mo_s = slow_mo_s
        self.only_no_issue_found = only_no_issue_found

        self.driver: Optional[webdriver.Edge] = None
        self.driver_init_error: str = ""

    def _init_driver(self) -> webdriver.Edge:
        driver = webdriver.Edge()
        driver.maximize_window()
        driver.set_page_load_timeout(self.page_timeout_s)
        return driver

    def _quit_driver(self) -> None:
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def run(self) -> None:
        try:
            self.driver = self._init_driver()
        except Exception as e:
            # Do not crash the worker thread; it can still drain jobs and emit Error results.
            self.driver = None
            self.driver_init_error = f"driver_init_failed={repr(e)}"
        try:
            while True:
                job = self.job_queue.get()
                if job is None:
                    self.job_queue.task_done()
                    break

                # Ensure queue bookkeeping is always correct: any uncaught exception must not
                # prevent task_done(), otherwise main thread can hang forever on job_queue.join().
                try:
                    try:
                        result = self._process_job(job)
                    except Exception as e:
                        result = JobResult(
                            row_id=job.row_id,
                            link=job.link,
                            page_issue_type="",
                            ai_result="Error",
                            ai_reason="",
                            ai_summary="",
                            error=f"worker_unhandled_exception={repr(e)}",
                        )

                    with self.results_lock:
                        self.results.append(result)

                    # Progress reporting (best-effort)
                    if self.progress is not None and self.progress_lock is not None:
                        with self.progress_lock:
                            self.progress["done"] = int(self.progress.get("done", 0)) + 1
                            done = int(self.progress["done"])
                            total = int(self.progress.get("total", 0) or 0)
                            every = int(self.progress.get("every", 0) or 0)
                            if every > 0 and (done % every == 0 or (total and done == total)):
                                pct = (done / total * 100.0) if total else 0.0
                                elapsed = time.time() - float(self.progress.get("start", time.time()))
                                print(
                                    f"[progress] {done}/{total} ({pct:.1f}%) elapsed={elapsed:.0f}s last={result.ai_result} worker={self.name}",
                                    flush=True,
                                )
                finally:
                    self.job_queue.task_done()
        finally:
            self._quit_driver()

    def _process_job(self, job: Job) -> JobResult:
        last_error = ""
        for attempt in range(self.retry + 1):
            try:
                if self.driver is None:
                    try:
                        self.driver = self._init_driver()
                        self.driver_init_error = ""
                    except Exception as e:
                        init_err = self.driver_init_error or f"driver_init_failed={repr(e)}"
                        return JobResult(
                            row_id=job.row_id,
                            link=job.link,
                            page_issue_type="",
                            ai_result="Error",
                            ai_reason="",
                            ai_summary="",
                            error=init_err,
                        )

                if self.slow_mo_s:
                    time.sleep(self.slow_mo_s)

                issue_type = open_case_and_get_issue_type(self.driver, job.link)

                if self.only_no_issue_found and issue_type != IssueEnum.NO_ISSUE_FOUND.value:
                    return JobResult(
                        row_id=job.row_id,
                        link=job.link,
                        page_issue_type=issue_type,
                        ai_result="Skipped",
                        ai_reason="",
                        ai_summary="",
                        error="",
                    )

                standard_steps, actual_steps, judge_comment = extract_case_steps(self.driver)

                if not standard_steps:
                    return JobResult(
                        row_id=job.row_id,
                        link=job.link,
                        page_issue_type=issue_type,
                        ai_result="Error",
                        ai_reason="",
                        ai_summary="",
                        error="No standard steps found",
                    )

                if not actual_steps:
                    return JobResult(
                        row_id=job.row_id,
                        link=job.link,
                        page_issue_type=issue_type,
                        ai_result="Error",
                        ai_reason="",
                        ai_summary="",
                        error="No actual steps found",
                    )

                if len(standard_steps) != len(actual_steps):
                    # Keep consistent with existing main.py behavior
                    return JobResult(
                        row_id=job.row_id,
                        link=job.link,
                        page_issue_type=issue_type,
                        ai_result="Error",
                        ai_reason="",
                        ai_summary="",
                        error=f"Mismatched number of steps: standard={len(standard_steps)} actual={len(actual_steps)}",
                    )

                report = compare_operations(standard_steps, actual_steps, issue_type, judge_comment)
                if not report or "final_summary" not in report:
                    return JobResult(
                        row_id=job.row_id,
                        link=job.link,
                        page_issue_type=issue_type,
                        ai_result="Error",
                        ai_reason="",
                        ai_summary="",
                        error="Model returned invalid JSON or missing final_summary",
                    )

                final = report.get("final_summary", {})
                ai_result = str(final.get("final_result", "NeedDiscussion"))
                ai_reason = str(final.get("reason", ""))
                ai_summary = str(final.get("ai_summary", final.get("summary", "")))

                return JobResult(
                    row_id=job.row_id,
                    link=job.link,
                    page_issue_type=issue_type,
                    ai_result=ai_result,
                    ai_reason=ai_reason,
                    ai_summary=ai_summary,
                    error="",
                )
            except Exception as e:
                last_error = repr(e)
                if attempt < self.retry:
                    # Driver/session can become invalid (e.g., connection refused). Recreate driver on retry.
                    try:
                        self._quit_driver()
                    except Exception:
                        pass
                    try:
                        self.driver = self._init_driver()
                    except Exception as init_e:
                        last_error = f"{last_error}; driver_reinit_failed={repr(init_e)}"
                    continue
                return JobResult(
                    row_id=job.row_id,
                    link=job.link,
                    page_issue_type="",
                    ai_result="Error",
                    ai_reason="",
                    ai_summary="",
                    error=last_error,
                )


def _safe_write_tsv(df: pd.DataFrame, path: str) -> str:
    """Write TSV safely.

    - Writes to a temp file first.
    - Atomically replaces the target.
    - If the target is locked (common on Windows when opened in Excel), falls back to a timestamped filename.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".tsv"
        path = base + ext

    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    df.to_csv(tmp_path, sep="\t", index=False, encoding="utf-8-sig")
    try:
        os.replace(tmp_path, path)
        return path
    except PermissionError:
        # Target likely opened/locked; write to a new name instead.
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt_path = f"{base}_{ts}{ext}"
        os.replace(tmp_path, alt_path)
        print(f"[warn] Output file locked, saved to: {alt_path}", flush=True)
        return alt_path
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def compute_confusion(df: pd.DataFrame, human_col: str, ai_col: str) -> pd.DataFrame:
    labels = ["Correct", "Incorrect", "NeedDiscussion", "Spam", "Error"]
    human = df[human_col].fillna("").replace({"": ""})
    ai = df[ai_col].fillna("").replace({"": ""})

    # Normalize common variants
    human = human.replace({"spam": "Spam"})
    ai = ai.replace({"spam": "Spam"})

    # Keep only known labels; everything else -> Error
    human = human.where(human.isin(labels), other="Error")
    ai = ai.where(ai.isin(labels), other="Error")

    return pd.crosstab(human, ai, rownames=["human"], colnames=["ai"], dropna=False)


def print_suggestions(noi: pd.DataFrame) -> None:
    """Heuristic suggestions focused on No Issue found prompt behavior."""
    if noi.empty:
        print("No No-Issue-Found rows available for analysis.")
        return

    total = len(noi)
    correct = int((noi["human_result"] == noi["ai_result"]).sum())
    acc = correct / total if total else 0.0

    print("\n==================== No Issue found analysis ====================")
    print(f"Rows: {total}")
    print(f"Exact-match accuracy (human_result == ai_result): {acc:.3f}")

    cm = compute_confusion(noi, "human_result", "ai_result")
    print("\nConfusion matrix (No Issue found):")
    print(cm)

    # Key mismatch patterns
    mism = noi[noi["human_result"] != noi["ai_result"]].copy()
    if mism.empty:
        print("\nNo mismatches. Prompt likely aligned for this sample.")
        return

    patt = (
        mism.groupby(["human_result", "ai_result"]).size().sort_values(ascending=False).head(10)
    )
    print("\nTop mismatch patterns:")
    print(patt)

    # Suggestion heuristics
    print("\nSuggestions (prompt-level, No Issue found):")

    nd_as_inc = mism[(mism["human_result"] == "NeedDiscussion") & (mism["ai_result"] == "Incorrect")]
    if len(nd_as_inc) > 0:
        print(
            f"- Many Human=NeedDiscussion but AI=Incorrect ({len(nd_as_inc)}): strengthen 'cannot prove' triggers (keyboard/hover/time/lock-wake/multi-action/outdated) and avoid penalizing as Incorrect when proof is inherently missing."
        )

    inc_as_nd = mism[(mism["human_result"] == "Incorrect") & (mism["ai_result"] == "NeedDiscussion")]
    if len(inc_as_nd) > 0:
        print(
            f"- Many Human=Incorrect but AI=NeedDiscussion ({len(inc_as_nd)}): tighten Incorrect rules for clearly visible misses (required UI state absent, wrong page, wrong toggle state, wrong value/label)."
        )

    spam_as_other = mism[(mism["human_result"] == "Spam") & (mism["ai_result"] != "Spam")]
    if len(spam_as_other) > 0:
        print(
            f"- Human=Spam but AI!=Spam ({len(spam_as_other)}): Spam gate may be too strict; consider adding clearer Spam examples like 'all screenshots identical / no progression' as decisive."
        )

    other_as_spam = mism[(mism["human_result"] != "Spam") & (mism["ai_result"] == "Spam")]
    if len(other_as_spam) > 0:
        print(
            f"- AI=Spam but Human!=Spam ({len(other_as_spam)}): Spam may be over-triggering; add a guard: if screenshots show task-related UI but wrong/incomplete actions, prefer Incorrect/NeedDiscussion over Spam."
        )

    err = noi[noi["ai_result"] == "Error"]
    if len(err) > 0:
        print(
            f"- AI errors ({len(err)}): add Selenium robustness (longer waits, retry sign-in, handle intermittent iframe load)."
        )

    print("- Review a small set of high-impact mismatches and convert them into explicit, short rules (1 sentence each). Avoid duplicates or exceptions that change decision order.")

    # Show a few examples for manual inspection
    sample = mism.head(8)[
        ["link", "page_issue_type", "human_result", "ai_result", "human_reason", "ai_reason"]
    ]
    print("\nSample mismatches (top 8):")
    for _, r in sample.iterrows():
        print(
            f"- {r['human_result']} -> {r['ai_result']} | {r['page_issue_type']} | {r['link']}\n"
            f"  human_reason: {str(r['human_reason'])[:160]}\n"
            f"  ai_reason: {str(r['ai_reason'])[:160]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-run CIP reviewer on links and compare AI vs human labels.")
    parser.add_argument(
        "--input-tsv",
        default="data/test_processed.tsv",
        help="Input TSV containing at least: link, type, human_result, human_reason (default: data/test_processed.tsv)",
    )
    parser.add_argument(
        "--output-tsv",
        default="data/test_processed_ai.tsv",
        help="Output TSV with AI results appended (default: data/test_processed_ai.tsv)",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="Number of worker threads/drivers")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of rows to process (0 = all)")
    parser.add_argument(
        "--only-human-result",
        type=str,
        default="",
        help="Optional filter: run only rows whose human_result matches (comma-separated). Example: Spam",
    )
    parser.add_argument("--page-timeout", type=int, default=60, help="Selenium page load timeout seconds")
    parser.add_argument("--retry", type=int, default=1, help="Retries per link (recreates driver on retry)")
    parser.add_argument("--slow-mo", type=float, default=0.0, help="Optional sleep seconds before each job")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed jobs (default: 10; 0 = disable)",
    )

    parser.add_argument(
        "--only-no-issue-found",
        action="store_true",
        help="Only call the LLM when the on-page issue_type is 'No Issue found'. Other types will be marked as Skipped.",
    )
    parser.add_argument(
        "--output-noi-tsv",
        default="",
        help="Optional output TSV containing only No Issue found rows (default: derived from --output-tsv when --only-no-issue-found is set)",
    )
    parser.add_argument(
        "--output-noi-mismatches-tsv",
        default="",
        help="Optional output TSV containing only No Issue found mismatches (human_result != ai_result)",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")

    # Prefer on-page issue_type as the canonical type for downstream usage.
    # Preserve the original TSV-provided 'type' (if present) as 'tsv_issue_type'.
    if "type" in df.columns and "tsv_issue_type" not in df.columns:
        df = df.rename(columns={"type": "tsv_issue_type"})
    for col in ["link", "human_result", "human_reason"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.input_tsv}. Columns: {list(df.columns)}")

    df = df.copy()
    df["link"] = df["link"].fillna("").astype(str).str.strip()
    df["human_result"] = df["human_result"].fillna("").astype(str).str.strip()
    df["human_reason"] = df["human_reason"].fillna("").astype(str).str.strip()

    df = df[df["link"].ne("")].copy()

    only_human = (args.only_human_result or "").strip()
    if only_human:
        allowed = {x.strip() for x in only_human.split(",") if x.strip()}
        if allowed:
            df = df[df["human_result"].isin(allowed)].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    df = df.reset_index(drop=True).copy()
    df["row_id"] = df.index.astype(int)

    jobs: List[Job] = []
    for _, row in df.iterrows():
        jobs.append(
            Job(
                row_id=int(row["row_id"]),
                link=str(row["link"]),
                human_result=str(row["human_result"]),
                human_reason=str(row["human_reason"]),
            )
        )

    job_queue: Queue = Queue()
    results: List[JobResult] = []
    results_lock = threading.Lock()

    progress: Dict[str, Any] = {
        "done": 0,
        "total": len(jobs),
        "every": int(args.progress_every),
        "start": time.time(),
    }
    progress_lock = threading.Lock()

    workers: List[Worker] = []
    for i in range(args.max_workers):
        w = Worker(
            name=f"worker-{i+1}",
            job_queue=job_queue,
            results=results,
            results_lock=results_lock,
            progress=progress,
            progress_lock=progress_lock,
            page_timeout_s=args.page_timeout,
            retry=args.retry,
            slow_mo_s=args.slow_mo,
            only_no_issue_found=bool(args.only_no_issue_found),
        )
        workers.append(w)
        w.start()

    for job in jobs:
        job_queue.put(job)

    # Sentinel to stop workers
    for _ in workers:
        job_queue.put(None)

    job_queue.join()

    # Basic end-of-run signal (useful when running in terminals that buffer output)
    print(f"\nAll jobs completed. Total input jobs: {len(jobs)}. Results collected: {len(results)}")

    # Merge results
    res_df = pd.DataFrame([r.__dict__ for r in results])
    if res_df.empty:
        raise RuntimeError("No results produced.")

    out = df.merge(res_df, on=["row_id"], how="left")
    out = out.drop(columns=[c for c in ["link_y"] if c in out.columns]).rename(
        columns={"link_x": "link"}
    )

    # Make the TSV's primary issue type reflect the on-page value.
    out["page_issue_type"] = out["page_issue_type"].fillna("").astype(str)
    out["type"] = out["page_issue_type"]
    out["issue_type"] = out["page_issue_type"]

    # Ensure stable ordering as input
    saved_out = _safe_write_tsv(out, args.output_tsv)
    print(f"Saved AI-augmented TSV: {saved_out}")

    # Overall analysis
    out["ai_result"] = out["ai_result"].fillna("")

    # No Issue found analysis (by on-page issue_type)
    noi = out[out["page_issue_type"].eq(IssueEnum.NO_ISSUE_FOUND.value)].copy()

    if args.only_no_issue_found:
        print("\n==================== Mode: Only No Issue found ====================")
    else:
        overall_cm = compute_confusion(out, "human_result", "ai_result")
        print("\n==================== Overall confusion (all types) ====================")
        print(overall_cm)

    print_suggestions(noi)

    if args.only_no_issue_found:
        noi_out = args.output_noi_tsv
        if not noi_out:
            if args.output_tsv.lower().endswith(".tsv"):
                noi_out = args.output_tsv[:-4] + "_noi.tsv"
            else:
                noi_out = args.output_tsv + "_noi.tsv"
        saved_noi_out = _safe_write_tsv(noi, noi_out)
        print(f"Saved No Issue found TSV: {saved_noi_out}")

        mism = noi[noi["human_result"] != noi["ai_result"]].copy()
        mism_out = args.output_noi_mismatches_tsv
        if not mism_out:
            if noi_out.lower().endswith(".tsv"):
                mism_out = noi_out[:-4] + "_mismatches.tsv"
            else:
                mism_out = noi_out + "_mismatches.tsv"
        saved_mism_out = _safe_write_tsv(mism, mism_out)
        print(f"Saved No Issue found mismatches TSV: {saved_mism_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
