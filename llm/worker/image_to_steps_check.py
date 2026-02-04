import json
import re
import asyncio
from pathlib import Path
from llm.client_manager import ClientManager
from enums.issue_enum import IssueEnum, SceneEnum
from utils.file_utils import load_prompt, get_prompt_file, resource_path
from utils.parameters import parse_parameters
from llm.agents.planer_agent import Planner
# from llm.tools import SemanticMemory

# semantic_memory = SemanticMemory(name="SemanticMemory")

IDENTIFY_JUDGE_SYSTEM_PROMPT = """
## ROLE
You are a top-tier prompt engineer expert.
Populate the prompt based on the results.

## Input Variables
human_judge: {human_judge},
result_reason: {result_reason}.
Given the current prompt and a labeled example case,
rewrite the system prompt to make future judgments more accurate.


## Output JSON Format
Output actions for EACH Section in the following JSON format:
{{
    "result_number": "If result_reason is not empty, return the smallest step number in the result where the problem exists"
}}
"""

COMPARISON_SYSTEM_PROMPT="""
## ROLE
You are a top-notch functional testing expert: extremely proficient in functional testing.

## INPUT VARIABLES

step_type_rule:{step_type_rule}
history_step_results: {history_steps}

## GOAL
The plot and step descriptions, as well as historical steps, are used to determine the current step's test result.

## Output JSON Format
Output actions for EACH Section in the following JSON format:
{{
    "final_summary": {{
        "final_result": "Correct" | "Incorrect" | "Spam" | "NeedDiscussion",
        "reason": "Explanation for the final result."
    }}
}}
"""


def _get_step_success_reason_by_raw_desc(test_cases: object, raw_text: str) -> str | None:

    raw_key = (raw_text or "").strip()
    if not raw_key:
        return None
    if not isinstance(test_cases, list):
        return None

    for case in test_cases:
        if not isinstance(case, dict):
            continue
        if str(case.get("step_raw_desc", "")).strip() == raw_key:
            value = case.get("step_success_reason")
            return None if value is None else str(value)
    return None



def _strip_code_fences(text: str) -> str:

    if not text:
        return text
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].lstrip().startswith("```") and lines[-1].lstrip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:

    if not text:
        return None

    s = text
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def _try_parse_json_object(raw_text: str) -> dict | None:
    if raw_text is None:
        return None
    cleaned = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    candidate = _extract_first_json_object(cleaned)
    if not candidate:
        return None

    candidate = re.sub(r",\s*(\}|\])", r"\1", candidate)

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_final_result(value: str | None) -> str:
    raw = (value or "").strip()
    key = re.sub(r"\s+", "", raw).lower()
    if key == "correct":
        return "Correct"
    if key == "incorrect":
        return "Incorrect"
    if key == "spam":
        return "Spam"
    if key in {"needdiscussion", "need_discussion", "need-discussion", "needdicussion"}:
        return "NeedDiscussion"
    return "NeedDiscussion"


async def check_steps_with_image_matching_async(steps_json, issue_type, judge_comment):

    planner = Planner()
    plans, group_duplicates = planner.plan(steps_json)
    total_step = len(plans)
    print(f'plans length: {len(plans)} ')
    args = parse_parameters()
    args.async_client = True
    client = ClientManager(args=args)

    print("Duplicate image step numbers:", group_duplicates)

    try:

        await asyncio.sleep(3)

        parsed = None
        content = None
        history_steps: list[dict] = []

        for step in plans:

            step_type = step.get("step_type", "")
            print(f"Processing step type: {step_type}")
            step_type_rule_path = get_prompt_file(step_type)
            step_type_rule = load_prompt(step_type_rule_path) if step_type_rule_path else ""

            system_prompt_step = COMPARISON_SYSTEM_PROMPT.format(
                step_type_rule=step_type_rule,
                history_steps=json.dumps(history_steps, ensure_ascii=False),
            )
            step_number = step.get("step_number", 999)
            user_content_structured = []

            ai_optimize_supple_text = step.get("text", "")
            raw_text = step.get("actual_text", "")
            image_url = step.get("actual_image_url") or step.get("standard_image_url")

            test_cases_path = Path(resource_path("llm/semanticmemory/cases.json"))
            test_cases: list[dict] = []
            try:
                if test_cases_path.exists():
                    raw_cases = (test_cases_path.read_text(encoding="utf-8") or "").strip()
                    test_cases = json.loads(raw_cases) if raw_cases else []
            except Exception:
                test_cases = []

            matched_step_success_reason = _get_step_success_reason_by_raw_desc(test_cases, raw_text)
            # aa = semantic_memory.query_steps(standard_text)

            # print(f"RAG return: {aa}")

            old_duplicates: list[int] = []
            try:
                if isinstance(group_duplicates, list) and group_duplicates:
                    if isinstance(group_duplicates[0], list):
                        for g in group_duplicates:
                            if step_number in g:

                                old_duplicates = [i for i in g if i <= step_number]
                                break
                    else:
                        if step_number in group_duplicates:
                            old_duplicates = [i for i in group_duplicates if i <= step_number]
            except Exception:
                old_duplicates = []

            user_content_structured = [
                {"type": "text", "text": f"Step standard description: {ai_optimize_supple_text}"},
                {"type": "text", "text": f"Duplicate image step numbers:{old_duplicates}, please consider this information when making judgments."},
                {"type": "text", "text": f"Step actual description: {raw_text}"},
            ]

            if matched_step_success_reason:
                print(f"matched_step_success_reason: {matched_step_success_reason}")
                user_content_structured.append(
                    {
                        "type": "text",
                        "text": (
                            "Matched example-case step_success_reason (same step_raw_desc): "
                            f"{matched_step_success_reason}"
                        ),
                    }
                )

            if isinstance(image_url, str):
                image_url = image_url.strip()
            else:
                image_url = None

            if image_url and (
                image_url.startswith("http://")
                or image_url.startswith("https://")
                or image_url.startswith("data:")
            ):
                user_content_structured.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )

            content = await client.chat_completion_async(
                messages=[
                    {"role": "system", "content": system_prompt_step},
                    {
                        "role": "user",
                        "content":user_content_structured
                    }
                ]
            )

            if content:
                parsed = _try_parse_json_object(content)
                if parsed and isinstance(parsed.get("final_summary"), dict):
                    final = parsed.get("final_summary", {})
                    final_result = _normalize_final_result(final.get("final_result"))
                    if final_result == "Correct":
                        print(f"Step {step.get('step_number', '?')}/{total_step} judged as Correct.")
                        history_steps.append({
                            "step_number": step_number,
                            "final_result": "Correct",
                            "reason": "",
                        })
                        continue
                    else:
                        final_reason = str(final.get("reason", "")).strip() or "No reason provided"

                        return {
                            "final_summary": {
                                "step_number": step_number,
                                "final_result": final_result,
                                "reason": final_reason
                            }
                        }

            print("Warning: model returned empty content, retrying in 3 seconds...")
            await asyncio.sleep(3)

        return {
            "final_summary": {
                "final_result": "Correct",
                "reason": "",
            }
        }

    finally:

        try:
            await client.aclose()
        except Exception:
            pass


async def compare_operations_async(standard_steps, actual_steps, issue_type, judge_comment, human_judge_result, expected_result):

    steps_json = build_steps_json(standard_steps, actual_steps)

    result = await check_steps_with_image_matching_async(steps_json, issue_type, judge_comment)
    return result


def build_steps_json(standard_steps, actual_steps):

    steps_json = []
    total_steps = max(len(standard_steps), len(actual_steps))
    for i in range(total_steps):
        step_number = i + 1
        standard_text = standard_steps[i]["text"] if i < len(standard_steps) else ""
        actual_text = actual_steps[i]["text"] if i < len(actual_steps) else ""
        standard_image_url = (
            standard_steps[i].get("img") if i < len(standard_steps) else None
        )
        actual_image_url = actual_steps[i].get("img") if i < len(actual_steps) else None

        step = {
            "step_number": step_number,
            "standard_text": standard_text,
            "actual_text": actual_text,
        }
        if standard_image_url:
            step["standard_image_url"] = standard_image_url

        if actual_image_url:
            step["actual_image_url"] = actual_image_url

        steps_json.append(step)

    return steps_json