import sys
import json
import os
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from parsers.document_parser import (
    extract_steps_from_left_pane,
    extract_steps_from_right_pane,
)
from llm.image_to_steps_check import compare_operations
from concurrent.futures import ThreadPoolExecutor


def normalize_issue_type(raw: str) -> str:
    s = (raw or "").strip()
    s = " ".join(s.split())
    low = s.lower()

    if ("no issue" in low and "found" in low) or ("无" in s and "问题" in s):
        return "No Issue found"
    if ("feature" in low and "not" in low and "found" in low) or ("功能" in s and ("未" in s or "没" in s) and ("找到" in s or "发现" in s)):
        return "Feature not found"
    if ("issue" in low and "found" in low) or ("问题" in s and ("发现" in s or "存在" in s or "有" in s)):
        return "Issue found"

    return s


# Modify the main function to return final_result and reason
def process_page(page_url: str):
    # Initialize the Edge WebDriver
    driver = webdriver.Edge()

    # Open the given page URL
    driver.get(page_url)
    driver.maximize_window()

    # Wait for the sign-in button to be clickable
    sign_in_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
    )
    sign_in_button.click()

    # Wait until the left and right panes are loaded
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "leftPane")))

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col"))
    )

    issue_type_raw = driver.find_element(By.CLASS_NAME, "textColorRed").text
    issue_type = normalize_issue_type(issue_type_raw)
    print(f"Issue Type: {issue_type}")

    # Extract standard steps and actual steps from the page
    standard_steps = extract_steps_from_left_pane(driver)
    print(f"Standard steps extracted: {len(standard_steps)}")
    judge_comment, actual_steps = extract_steps_from_right_pane(driver)
    print(f"Actual steps extracted: {len(actual_steps)}")

    # Validate extracted steps
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
    # Close the browser
    driver.quit()

    print("Comparing steps...")

    # Compare the steps and return the result in JSON format
    report = compare_operations(standard_steps, actual_steps, issue_type, judge_comment)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    final_result = report["final_summary"]["final_result"]
    reason = report["final_summary"]["reason"]

    return final_result, reason


def process_excel(file_path: str):
    # Read the Excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    # Extract the "permalink" column
    links = df["permalink"].tolist()

    # Create a thread pool to process each page_url in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_page, url.strip()): url for url in links}

        for future in futures:
            page_url = futures[future]
            try:
                final_result, reason = future.result()
                results.append((page_url, final_result, reason))
            except Exception as e:
                print(f"Error processing {page_url}: {e}")
                results.append((page_url, "Error", str(e)))

    # Write the results back to the last two columns in the Excel file
    for index, (url, final_result, reason) in enumerate(results):
        df.at[index, 'final_result'] = final_result
        df.at[index, 'reason'] = reason

    # Save the updated Excel file
    output_file = file_path.replace(".xlsx", "_updated.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Updated Excel file saved as {output_file}")


def is_url(string):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.match(string)

def is_excel_file(file_path):
    return file_path.lower().endswith(('.xlsx', '.xls'))

if __name__ == "__main__":
    # Input can be a page URL or an Excel file path
    if len(sys.argv) < 2:
        print("Please provide a page URL or Excel file path.")
        sys.exit(1)

    file_path_or_url = sys.argv[1]

    # If the argument is a URL, call process_page
    if is_url(file_path_or_url):
        print(f"Detected page URL: {file_path_or_url}")
        process_page(file_path_or_url)

    # If the argument is an Excel file path, call process_excel
    elif is_excel_file(file_path_or_url):
        print(f"Detected Excel file: {file_path_or_url}")
        process_excel(file_path_or_url)

    # If neither a URL nor an Excel file path, display an error message
    else:
        print("The provided argument is neither a valid URL nor an Excel file path.")
        sys.exit(1)
