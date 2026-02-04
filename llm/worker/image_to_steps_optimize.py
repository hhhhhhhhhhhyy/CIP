import json
import re
import asyncio
from pathlib import Path
from llm.client_manager import ClientManager
from enums.issue_enum import IssueEnum, SceneEnum, ScenarioEnum
from utils.file_utils import load_prompt, resource_path, get_prompt_file
from utils.parameters import parse_parameters
from llm.agents.planer_agent import Planner
# from llm.tools import SemanticMemory

# semantic_memory = SemanticMemory(name="SemanticMemory")

example_case = "llm/semanticmemory/cases.json"


def _load_example_cases(path: str) -> list[dict]:
    try:
        p = Path(resource_path(path))
        if not p.exists():
            return []
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _example_case_key(step_type: str | None, step_raw_desc: str | None) -> tuple[str, str]:
    return (str(step_type or "").strip(), str(step_raw_desc or "").strip())


def _append_example_case_if_new(
    path: str,
    *,
    step_type: str,
    step_raw_desc: str,
    step_ai_desc: str,
    step_success_reason: str,
) -> bool:
    cases = _load_example_cases(path)
    new_key = _example_case_key(step_type, step_raw_desc)
    if any(_example_case_key(c.get("step_type"), c.get("step_raw_desc")) == new_key for c in cases if isinstance(c, dict)):
        return False

    cases.append(
        {
            "step_type": step_type,
            "step_raw_desc": step_raw_desc,
            "step_ai_desc": step_ai_desc,
            "step_success_reason": step_success_reason,
        }
    )

    p = Path(resource_path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

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

OPTIMIZA_SYSTEM_PROMPT="""
## ROLE
You are a top-notch functional testing expert: extremely proficient in functional testing.

## INPUT VARIABLES
step_type_rule: {step_type_rule}

## Output JSON Format
Output actions for EACH Section in the following JSON format:
{{
    "final_summary": {{
        "final_result": "Correct" | "Incorrect" | "Spam" | "NeedDiscussion",
        "reason": "Explanation for the final result."
    }}
}}
"""

OPTIMIZATION_SYSTEM_PROMPT = """
## ROLE
You are a top-tier functional testing expert, proficient in functional testing and prompt development.

## GOAL
Optimize the prompt based on results to ensure consistency between AI and human results.

## INPUT VARIABLES
Correct_Result: {human_judge_result}
Correct_Reason: {human_judge_reason}
AI_Judgment_Result: {ai_judge_result}
AI_Judgment_Reason: {ai_judge_reason}
History_Rule: {history_rule}

## Output JSON Format
Output actions for EACH Section in the following JSON format:
{{
    "step_type_rule":"<the full updated sype rule>"
}}
"""


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


async def optimization_steps_with_image_matching_async(step_type_rule, user_content_structured):


    args = parse_parameters()
    args.async_client = True
    client = ClientManager(args=args)
    optimized_prompt = OPTIMIZA_SYSTEM_PROMPT.format(
        step_type_rule=step_type_rule
    )

    try:

        await asyncio.sleep(3)

        content = await client.chat_completion_async(
            messages=[
                {"role": "system", "content": optimized_prompt},
                {"role": "user", "content":user_content_structured}
            ]
        )

        if content:
            parsed = _try_parse_json_object(content)
            if parsed and isinstance(parsed.get("final_summary"), dict):
                final = parsed.get("final_summary", {})
                final_result = _normalize_final_result(final.get("final_result"))
                final_reason = str(final.get("reason", "")).strip()
                return {"final_result": final_result, "reason": final_reason}

        return {
            "final_result": "NeedDiscussion",
            "reason": "Model returned empty/invalid content.",
        }


    finally:

        try:
            await client.aclose()
        except Exception:
            pass

async def check_steps_with_image_matching_async(steps_json, issue_type, judge_comment):

    planner = Planner()
    plans, group_duplicates = planner.plan(steps_json)
    total_step = len(plans)
    print(f'plans length: {len(plans)} ')
    args = parse_parameters()
    args.async_client = True
    client = ClientManager(args=args)

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

            standard_text = step.get("text", "")
            actual_text = step.get("actual_text", "")

            image_url = step.get("actual_image_url") or step.get("standard_image_url")


            user_content_structured = [
                {"type": "text", "text": f"Step standard description: {standard_text}"},
                {"type": "text", "text": f"Step actual description: {actual_text}"},
            ]

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




async def optimize_prompt_async(steps_json, issue_type, judge_comment, human_judge, expected_result: str):

    planner = Planner()
    plans, group_duplicates = planner.plan(steps_json)
    total_step = len(plans)
    print(f"plans length: {total_step} ")

    if not plans:
        return {
            "final_summary": {
                "final_result": "NeedDiscussion",
                "reason": "Planner returned empty plan; cannot optimize prompt.",
            }
        }

    args = parse_parameters()
    args.async_client = True
    client = ClientManager(args=args)

    def _parse_result_number_from_reason(reason_text: str | None) -> int | None:
        try:
            m = re.search(r"step\s*number\s*:\s*(\d+)", str(reason_text or ""), re.IGNORECASE)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    def _is_valid_image_url(url: str | None) -> bool:
        if not isinstance(url, str):
            return False
        u = url.strip()
        return bool(u) and (u.startswith("http://") or u.startswith("https://") or u.startswith("data:"))

    async def _judge_step(step_type_rule: str, history_steps: list[dict], user_content_structured: list[dict]) -> tuple[str, str]:
        system_prompt_step = COMPARISON_SYSTEM_PROMPT.format(
            step_type_rule=step_type_rule or "",
            history_steps=json.dumps(history_steps, ensure_ascii=False),
        )
        content_compare = await client.chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt_step},
                {"role": "user", "content": user_content_structured},
            ]
        )
        parsed_compare = _try_parse_json_object(content_compare)
        final_compare = (parsed_compare or {}).get("final_summary") if isinstance(parsed_compare, dict) else None
        if isinstance(final_compare, dict):
            ai_result = _normalize_final_result(final_compare.get("final_result"))
            ai_reason = str(final_compare.get("reason", "")).strip()
            return ai_result, ai_reason
        return "NeedDiscussion", "Model compare output invalid."

    try:
        await asyncio.sleep(1)


        # 获取错误的步数
        if expected_result is not None:
            result_number = _parse_result_number_from_reason(expected_result)

            if result_number is None:
                identify_judge_system_prompt = IDENTIFY_JUDGE_SYSTEM_PROMPT.format(
                    human_judge=human_judge,
                    result_reason=expected_result,
                )
                content = await client.chat_completion_async(
                    messages=[{"role": "system", "content": identify_judge_system_prompt}]
                )
                optimization_of_prompts = _try_parse_json_object(content) or {}
                try:
                    result_number = int(optimization_of_prompts.get("result_number") or total_step)
                except Exception:
                    result_number = total_step
        else:
            result_number = total_step

        print(f"Result number: {result_number}")

        human_problem_result = _normalize_final_result(str(human_judge or ""))
        human_problem_reason = str(expected_result or "").strip()

        prompt_cache: dict[str, str] = {}
        touched_paths: set[str] = set()

        history_steps: list[dict] = []

        for step in plans:
            print(f"Optimizing step type: {step.get('step_type', '')}")
            try:
                step_number = int(step.get("step_number", 999))
            except Exception:
                step_number = 999

            if step_number > result_number:
                break

            step_type = str(step.get("step_type", "")).strip()
            prompt_path = get_prompt_file(step_type)
            if not prompt_path:
                return {
                    "final_summary": {
                        "final_result": "NeedDiscussion",
                        "reason": f"Unknown step_type for optimization: {step_type}",
                    }
                }

            step_type_rule = prompt_cache.get(prompt_path)
            if step_type_rule is None:
                step_type_rule = load_prompt(prompt_path) or ""
                prompt_cache[prompt_path] = step_type_rule


            ai_optimize_supple_text = step.get("text", "")
            raw_text = step.get("actual_text", "")
            image_url = step.get("actual_image_url") or step.get("standard_image_url")

            user_content_structured = [
                {"type": "text", "text": f"Step AI optimization and supplementation description: {ai_optimize_supple_text}"},
                {"type": "text", "text": f"Step actual description: {raw_text}"},
            ]
            if _is_valid_image_url(image_url):
                user_content_structured.append({"type": "image_url", "image_url": {"url": image_url.strip()}})

            if step_number < result_number:
                desired_result = "Correct"
                desired_reason = ""
            else:
                desired_result = human_problem_result
                desired_reason = human_problem_reason


            max_rounds = 6
            ai_judge_result = "NeedDiscussion"
            ai_judge_reason = ""
            for _ in range(max_rounds):
                ai_judge_result, ai_judge_reason = await _judge_step(
                    step_type_rule,
                    history_steps,
                    user_content_structured,
                )
                if ai_judge_result == desired_result and ai_judge_result == "Correct":

                    print("============================================================")
                    print(f"Step {step_number}/{total_step} optimized as Correct.")
                    print(f"ai judge reason: {ai_judge_reason}")
                    print("============================================================")

                    # semantic_memory.store_step(
                    #     step_type=step_type,
                    #     step_ai_desc=actual_text,
                    #     step_raw_desc=standard_text,
                    #     step_success_reason=ai_judge_reason,
                    # )
                    try:
                        saved = _append_example_case_if_new(
                            example_case,
                            step_type=step_type,
                            step_raw_desc=raw_text,
                            step_ai_desc=ai_optimize_supple_text,
                            step_success_reason=ai_judge_reason,
                        )
                        if saved:
                            print("Saved new example_case to cases.json")
                        else:
                            print("example_case already exists; skip saving")
                    except Exception:
                        pass


                    break
                elif ai_judge_result == desired_result and step_number == int(result_number) and ai_judge_result != "Correct":
                    break

                optimization_prompt = OPTIMIZATION_SYSTEM_PROMPT.format(
                    human_judge_result=desired_result,
                    human_judge_reason=desired_reason,
                    ai_judge_result=ai_judge_result,
                    ai_judge_reason=ai_judge_reason,
                    history_rule=step_type_rule,
                )

                content_opt = await client.chat_completion_async(
                    messages=[
                        {"role": "system", "content": optimization_prompt},
                        {"role": "user", "content": user_content_structured},
                    ]
                )

                parsed_opt = _try_parse_json_object(content_opt) or {}
                new_rule = parsed_opt.get("step_type_rule")
                if not isinstance(new_rule, str) or not new_rule.strip():
                    break

                step_type_rule = new_rule
                prompt_cache[prompt_path] = step_type_rule
                touched_paths.add(prompt_path)

            if desired_result == "Correct" and ai_judge_result == "Correct":
                history_steps.append(
                    {"step_number": step_number, "final_result": "Correct", "reason": ""}
                )
            else:
                if step_number == int(result_number):
                    break

        for path in touched_paths:
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompt_cache.get(path, ""))

        return await check_steps_with_image_matching_async(steps_json, issue_type, judge_comment)

    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def compare_operations_async(standard_steps, actual_steps, issue_type, judge_comment, human_judge_result, expected_result):

    steps_json = build_steps_json(standard_steps, actual_steps)

    result = await check_steps_with_image_matching_async(steps_json, issue_type, judge_comment)
    return result


async def optimize_prompttions_async(standard_steps, actual_steps, issue_type, judge_comment, human_judge_result, expected_result):

    steps_json = build_steps_json(standard_steps, actual_steps)

    result = await optimize_prompt_async(steps_json, issue_type, judge_comment, human_judge_result, expected_result)
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