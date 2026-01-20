import json
from llm.client_manager import ClientManager
from enums.issue_enum import IssueEnum
from utils.file_utils import load_prompt


def check_steps_with_image_matching(steps_json, issue_type, judge_comment):

    PROMPT_FILES = {
        IssueEnum.FEATURE_NOT_FOUND.value: "llm/prompts/image_feature_not_found_prompt.txt",
        IssueEnum.NO_ISSUE_FOUND.value: "llm/prompts/image_no_issue_found_prompt.txt",
        IssueEnum.ISSUE_FOUND.value: "llm/prompts/image_issue_found_prompt.txt",
    }

    # Load the system prompt based on issue type
    prompt_file = PROMPT_FILES.get(issue_type)
    if not prompt_file:
        # Defensive fallback: some cases may show unexpected labels (e.g., "NAS")
        # We still want a stable evaluation rather than crashing.
        prompt_file = PROMPT_FILES[IssueEnum.NO_ISSUE_FOUND.value]

    system_prompt = load_prompt(prompt_file)

    if issue_type in [IssueEnum.ISSUE_FOUND.value, IssueEnum.FEATURE_NOT_FOUND.value]:
        system_prompt = system_prompt.replace("{judge_comment}", judge_comment)

    # Prepare structured user content with 'type' fields
    user_content_structured = []
    for step in steps_json:

        # Step header (helps GPT maintain structure and stability)
        user_content_structured.append({
            "type": "text",
            "text": f"=== Step {step['step_number']} ==="
        })

        # Standard text (always provided)
        user_content_structured.append({
            "type": "text",
            "text": f"Standard Text: {step['standard_text']}"
        })

        # Standard image (may not exist)
        if step.get("standard_image_url"):
            user_content_structured.append({
                "type": "text",
                "text": "Standard Image:"
            })
            user_content_structured.append({
                "type": "image_url",
                "image_url": {"url": step["standard_image_url"]}
            })
        else:
            user_content_structured.append({
                "type": "text",
                "text": "No Standard Image Provided."
            })

        # Actual image (may not exist)
        if step.get("actual_image_url"):
            user_content_structured.append({
                "type": "text",
                "text": "Actual Image:"
            })
            user_content_structured.append({
                "type": "image_url",
                "image_url": {"url": step["actual_image_url"]}
            })
        else:
            user_content_structured.append({
                "type": "text",
                "text": "No Actual Image Provided."
            })


    # Call GPT model
    content = ClientManager().chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_structured},
        ],
        model="gpt-5.1",
        max_tokens=2000,
        temperature=0.0,
        top_p=1.0,
        timeout=120,
    )

    # Parse JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            parsed = json.loads(content[start:end])
        except Exception:
            print("Warning: model output is not valid JSON.")
            print(content)
            return None

    # Validate final_summary
    # Support both legacy formats:
    # 1) {"final_summary": {"final_result": ..., "reason": ..., "ai_summary"/"summary": ...}}
    # 2) {"final_result": ..., "reason": ..., "summary": ...} (older simplified prompt)
    final = parsed.get("final_summary")
    if not isinstance(final, dict):
        final = {
            "final_result": parsed.get("final_result"),
            "reason": parsed.get("reason", ""),
            # Normalize display-safe summary field name
            "ai_summary": parsed.get("ai_summary", parsed.get("summary", "")),
        }

    # Normalize fields to strings for downstream TSV storage
    if isinstance(final.get("ai_summary"), list):
        items = [str(x).strip() for x in final.get("ai_summary") if str(x).strip()]
        final["ai_summary"] = "\n".join(f"- {x}" for x in items)
    elif isinstance(final.get("ai_summary"), dict):
        final["ai_summary"] = json.dumps(final.get("ai_summary"), ensure_ascii=False)
    elif final.get("ai_summary") is None:
        final["ai_summary"] = ""

    if isinstance(final.get("reason"), list):
        items = [str(x).strip() for x in final.get("reason") if str(x).strip()]
        final["reason"] = " ".join(items)
    elif final.get("reason") is None:
        final["reason"] = ""

    final_result = final.get("final_result")
    if final_result not in {"Correct", "Incorrect", "Spam", "NeedDiscussion"}:
        final_result = "NeedDiscussion"
        final_reason = "Model returned unknown final_result, manual review required"
        final = {"final_result": final_result, "reason": final_reason, "ai_summary": ""}

    # Ensure ai_summary is present (display-safe, not chain-of-thought)
    if "ai_summary" not in final:
        final["ai_summary"] = final.get("summary", "")

    return {"final_summary": final}


def compare_operations(standard_steps, actual_steps, issue_type, judge_comment):

    steps_json = build_steps_json(standard_steps, actual_steps)
    result = check_steps_with_image_matching(steps_json, issue_type, judge_comment)
    return result


def build_steps_json(standard_steps, actual_steps):
    """
    Build a simplified steps JSON containing only standard text and actual step image.
    Output format:
    [
        {"step_number": 1, "standard_text": "<text>", "standard_image_url": "<url>", "actual_image_url": "<url>"},
        ...
    ]
    """
    steps_json = []
    total_steps = max(len(standard_steps), len(actual_steps))
    for i in range(total_steps):
        step_number = i + 1
        standard_text = standard_steps[i]["text"] if i < len(standard_steps) else ""
        standard_image_url = (
            standard_steps[i].get("img") if i < len(standard_steps) else None
        )
        actual_image_url = actual_steps[i].get("img") if i < len(actual_steps) else None

        step = {"step_number": step_number, "standard_text": standard_text}
        if standard_image_url:
            step["standard_image_url"] = standard_image_url

        if actual_image_url:
            step["actual_image_url"] = actual_image_url

        steps_json.append(step)

    return steps_json