import sys
import json
import os
import re
import asyncio
import math
import functools
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from parsers.document_parser import (
    extract_steps_from_left_pane,
    extract_steps_from_right_pane,
)
from llm.worker import compare_operations_async, optimize_prompttions_async
from concurrent.futures import ThreadPoolExecutor
from utils.parameters import parse_parameters


def process_page(
    page_url: str,
    human_judge: str = None,
    expected_result: str = None,
    work_type: str = "C",
):

    driver = webdriver.Edge()

    driver.get(page_url)
    driver.maximize_window()

    sign_in_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
    )
    sign_in_button.click()

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "leftPane")))

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col"))
    )

    issue_type = driver.find_element(By.CLASS_NAME, "textColorRed").text
    print(f"Issue Type: {issue_type}")

    standard_steps = extract_steps_from_left_pane(driver)

    print(f"Standard steps extracted: {len(standard_steps)}")
    judge_comment, actual_steps = extract_steps_from_right_pane(driver)

    print(f"Actual steps extracted: {len(actual_steps)}")

    if not standard_steps:
        print("Error: No standard steps found in the left pane.")
        driver.quit()
        return "Error", "No standard steps found."

    if not actual_steps:
        print("Error: No actual steps found in the right pane.")
        driver.quit()
        return "Error", "No actual steps found."

    if len(standard_steps) != len(actual_steps):
        print("Error: Mismatched number of steps.")
        driver.quit()
        return "Error", "Mismatched number of steps."

    print("All checks passed.")
    driver.quit()

    print("Comparing steps...")
    selected_work_type = (work_type or "C").upper()
    if selected_work_type == "O":
        work = optimize_prompttions_async
    else:
        # Default to compare when unknown
        work = compare_operations_async

    report = asyncio.run(work(standard_steps, actual_steps, issue_type, judge_comment, human_judge, expected_result))

    # Normalize report shape (some paths may return JSON strings or final_summary as a string)
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except Exception:
            report = {
                "final_summary": {
                    "final_result": "NeedDiscussion",
                    "reason": f"Model returned non-JSON string report: {report}",
                }
            }

    if isinstance(report, dict) and isinstance(report.get("final_summary"), str):
        report = {
            "final_summary": {
                "final_result": report.get("final_summary") or "NeedDiscussion",
                "reason": "final_summary was a string; normalized.",
            }
        }

    if not report or "final_summary" not in report:
        report = {
            "final_summary": {
                "final_result": "NeedDiscussion",
                "reason": "No valid model report returned; manual review required.",
            }
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    final_result = report["final_summary"].get("final_result", "NeedDiscussion")
    step_number = report["final_summary"].get("step_number", -1)
    reason = report["final_summary"].get("reason", "No reason provided")

    return final_result, step_number, reason


def process_excel(file_path: str, concurrency: int, work_type: str = "C"):
    return asyncio.run(process_excel_async(file_path, concurrency=concurrency, work_type=work_type))


async def _process_one_page(
    idx: int,
    page_url: str,
    human_judge: str,
    expected_result: str,
    work_type: str,
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
):

    if page_url is None:
        url = ""
    elif isinstance(page_url, float) and math.isnan(page_url):
        url = ""
    else:
        url = str(page_url).strip()
    if not url:
        return idx, page_url, "Error", -1, "Empty URL"

    async with sem:
        try:
            loop = asyncio.get_running_loop()
            runner = functools.partial(process_page, url, human_judge, expected_result, work_type)
            final_result, step_number, reason = await loop.run_in_executor(executor, runner)
            return idx, page_url, final_result, step_number, reason
        except Exception as e:
            print(f"Error processing {page_url}: {e}")
            if str(os.getenv("CIP_DEBUG_TRACEBACK", "")).lower() in {"1", "true", "yes"}:
                import traceback
                print(traceback.format_exc())
            return idx, page_url, "Error", -1, str(e)


async def process_excel_async(file_path: str, concurrency: int = 10, work_type: str = "C"):

    df = pd.read_excel(file_path, engine='openpyxl')
    print(df.head())

    links = df["permalink"].tolist()
    vender_judges = df["vendor judgement"].tolist()
    reasons = df["结果分析"].tolist()

    sem = asyncio.Semaphore(concurrency)
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        tasks = [
            asyncio.create_task(_process_one_page(i, url, vender_judges[i], reasons[i], work_type, sem, executor))
            for i, url in enumerate(links)
        ]
        for task in asyncio.as_completed(tasks):
            results.append(await task)

    for idx, url, final_result, step_number, reason in results:
        df.at[idx, 'final_result'] = final_result
        df.at[idx, 'step_number'] = step_number
        df.at[idx, 'reason'] = reason

    if file_path.lower().endswith(".xlsx"):
        output_file = file_path[:-5] + "_updated.xlsx"
    elif file_path.lower().endswith(".xls"):
        output_file = file_path[:-4] + "_updated.xlsx"
    else:
        output_file = file_path + "_updated.xlsx"

    df.to_excel(output_file, index=False)
    print(f"Updated Excel file saved as {output_file}")

    return output_file


def is_url(string):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.match(string)

def is_excel_file(file_path):
    return file_path.lower().endswith(('.xlsx', '.xls'))

if __name__ == "__main__":

    args = parse_parameters()
    test_file_or_url = args.test_file_or_url

    if is_url(test_file_or_url):
        print(f"Detected page URL: {test_file_or_url}")
        process_page(test_file_or_url, work_type=args.work_type)

    elif is_excel_file(test_file_or_url):
        print(f"Detected Excel file: {test_file_or_url}")
        process_excel(test_file_or_url, args.concurrency, work_type=args.work_type)

    else:
        print("The provided argument is neither a valid URL nor an Excel file path. Use --test_file to specify a URL or Excel path.")
        sys.exit(1)
