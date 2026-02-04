import ast
import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from traitlets import Bool

from llm.tools.image_quality import find_duplicates_in_items, duplicate_pairs_to_groups
from llm.agents.hello_agent import HelloAgentsLLM

PLANNER_PROMPT_TEMPLATE = """
You are a top-tier AI planning functional test expert.
Your task is to analyze user-provided test case text and screenshots to develop an action plan for each step (Step Number).
Please ensure that each step in the plan is an independent, executable task, and arranged strictly in logical order.
Each step will have text and image information,
and we categorize each step as follows: UI_INTERACTION, STATE_VERIFICATION, CONDITIONAL, NAVIGATION, WAITING, DESCRIPTIVE, INPUT, SCROLL

Your output must be a Python list, where each element is a string describing a subtask.

Input: [text + image_url]

Please strictly adhere to the following format when outputting your plan, with ```python and ``` as prefixes and suffixes being necessary:
```python
[{
    "step_number": Step Number,
    "step_type": "Navigation & URL Redirection" | "Functional Layout Setting" | "UI Visibility, Layout & Rendering Verification" | "Appearance & Theme Settings" | "Browser Settings & Configuration" | "Environment & Precondition Setup" | "Advertising Verification & Reporting" | "Localization & Internationalization" | "Accessibility & Keyboard Navigation" | "Carousel & Slider Controls" | "Media Playback & Audio Control" | "Authentication & User Profile Management" | "Tab & Window Management" | "Modals, Popups & Notifications Handling" | "Search Functionality & SERP Module Validation" | "Widgets, Taskbar & OS-Level Integrations",
    "text": "English step description, based on the standard text and image.",
}]
```
""".strip()


class Planner:
    def __init__(self):
        self.llm_client = HelloAgentsLLM()

    def plan(self, question) -> list[dict]:

        content_structured, user_content_structured, user_act_images = self.assemble_json(question)

        group_duplicates = self.found_duplicates_images(user_act_images)

        group_duplicates = [[i + 1 for i in g] for g in group_duplicates]

        messages=[
            {"role": "system", "content": PLANNER_PROMPT_TEMPLATE},
            {"role": "user", "content": content_structured},
        ]

        print("--- Generating plan ---")

        response_text = self.llm_client.think(messages=messages) or ""

        print(f"✅ Plan generated:\n{response_text}")

        try:

            plan_str = response_text.split("```python")[1].split("```")[0].strip()

            plan = ast.literal_eval(plan_str)

            if isinstance(plan, list):
                plan = self.merge_plan_with_user_content(plan, user_content_structured)
            else:
                plan = user_content_structured

            return plan, group_duplicates
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ Error parsing plan: {e}")
            print(f"Raw response: {response_text}")
            return [], group_duplicates
        except Exception as e:
            print(f"❌ Unknown error while parsing plan: {e}")
            return [], group_duplicates

    def assemble_json(self, steps_json) -> list[dict]:

        content_structured = []
        user_content_structured = []
        user_act_images = []
        for step in steps_json:

            user_content_structured_tmp = []

            standard_image_url = step.get("standard_image_url")
            if standard_image_url:
                standard_image_url = self._normalize_image_url(standard_image_url)

            if standard_image_url:

                content_structured.append({
                    "type": "text",
                    "text": f"=== Step Number {step['step_number']} ===\nStandard Text: {step['standard_text']}\nStandard Image:"
                })

                content_structured.append({
                    "type": "image_url",
                    "image_url": {"url": standard_image_url}
                })
            else:
                content_structured.append({
                    "type": "text",
                    "text": f"=== Step Number {step['step_number']} ===\nStandard Text: {step['standard_text']}\nStandard Image: No Standard Image Provided."
                })



            actual_image_url = step.get("actual_image_url")
            if actual_image_url:
                actual_image_url = self._normalize_image_url(actual_image_url)

            actual_text = step.get("actual_text")
            if actual_text is None:
                actual_text = ""

            if actual_image_url:

                user_act_images.append(actual_image_url)

                user_content_structured_tmp.append({
                    "text": f"Standard Text: {step['standard_text']}\nActual Text: {actual_text}\nActual Image:"
                })

                user_content_structured_tmp.append({
                    "image_url": actual_image_url
                })
            else:

                user_content_structured_tmp.append({
                    "text": f"Standard Text: {step['standard_text']}\nActual Text: {actual_text}\nActual Image: No Actual Image Provided."

                })
            user_content_structured.append(user_content_structured_tmp)

        return content_structured, user_content_structured, user_act_images

    def _normalize_image_url(self, url: str) -> str | None:
        url = str(url).strip() if url is not None else ""
        if not url:
            return None
        if url.startswith("data:"):
            return url
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return url
        if parsed.scheme == "file":
            candidate = Path(parsed.path.lstrip("/"))
        else:
            candidate = Path(url)
        if not candidate.exists() or not candidate.is_file():
            return None
        mime, _ = mimetypes.guess_type(candidate.name)
        if not mime:
            mime = "image/png"
        data = candidate.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def merge_plan_with_user_content(self, plan, user_content):

        merged = []
        for step, content in zip(plan, user_content):

            actual_text = None
            actual_image_url = None

            if isinstance(content, list) and len(content) >= 1 and isinstance(content[0], dict):
                actual_text = content[0].get("text")
            if isinstance(content, list) and len(content) >= 2 and isinstance(content[1], dict):
                actual_image_url = content[1].get("image_url")

            merged.append({
                **step,
                "actual_text": actual_text,
                "actual_image_url": actual_image_url,
            })
        return merged


    def found_duplicates_images(self, items):

        _, duplicates = find_duplicates_in_items(items)
        groups = duplicate_pairs_to_groups(duplicates)

        return groups


