import sys
import os
from pathlib import Path
import re
from enums.issue_enum import IssueEnum, SceneEnum, ScenarioEnum

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    if relative_path is None:
        raise ValueError("resource_path() expected a non-None path")
    if not isinstance(relative_path, (str, os.PathLike)):
        raise TypeError(
            f"resource_path() expected str or os.PathLike, got {type(relative_path).__name__}"
        )

    p = Path(relative_path)
    if p.is_absolute():
        return str(p)

    # PyInstaller bundled runtime path
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return str(Path(bundle_root) / p)

    # dev environment: resolve relative to repo root (folder containing utils/)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / p)


def load_prompt(filename: str) -> str:

    if filename is None:
        raise ValueError("load_prompt() expected a non-None filename")

    include_pattern = re.compile(r"@@INCLUDE:\s*(.+?)\s*@@")

    def _load_inner(name: str, depth: int, seen: set[str]) -> str:
        if depth > 10:
            raise ValueError(f"Prompt include depth exceeded for: {name}")
        if name in seen:
            raise ValueError(f"Circular @@INCLUDE detected for: {name}")

        real_path = resource_path(name)
        prompt_path = Path(real_path)

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        text = prompt_path.read_text(encoding="utf-8")

        def _replace(match: re.Match) -> str:
            included = match.group(1).strip()
            return _load_inner(included, depth + 1, seen | {name})

        return include_pattern.sub(_replace, text)

    return _load_inner(filename, 0, set())


def get_prompt_file(issue_type: str) -> str | None:
    key = str(issue_type).strip() if issue_type is not None else ""

    prompt_files = {
        # Issue-type prompts (values)
        IssueEnum.FEATURE_NOT_FOUND.value: "llm/prompts/image_feature_not_found_prompt.txt",
        IssueEnum.NO_ISSUE_FOUND.value: "llm/prompts/image_no_issue_found_prompt.txt",
        IssueEnum.ISSUE_FOUND.value: "llm/prompts/image_issue_found_prompt.txt",

        # Step-type prompts: accept BOTH enum value ("UI Interaction") and enum name ("UI_INTERACTION")
        SceneEnum.UI_INTERACTION.value: "llm/prompt3/UI_INTERACTION.txt",
        SceneEnum.UI_INTERACTION.name: "llm/prompt3/UI_INTERACTION.txt",
        SceneEnum.STATE_VERIFICATION.value: "llm/prompt3/STATE_VERIFICATION.txt",
        SceneEnum.STATE_VERIFICATION.name: "llm/prompt3/STATE_VERIFICATION.txt",
        SceneEnum.CONDITIONAL.value: "llm/prompt3/CONDITIONAL.txt",
        SceneEnum.CONDITIONAL.name: "llm/prompt3/CONDITIONAL.txt",
        SceneEnum.NAVIGATION.value: "llm/prompt3/NAVIGATION.txt",
        SceneEnum.NAVIGATION.name: "llm/prompt3/NAVIGATION.txt",
        SceneEnum.WAITING.value: "llm/prompt3/WAITING.txt",
        SceneEnum.WAITING.name: "llm/prompt3/WAITING.txt",
        SceneEnum.DESCRIPTIVE.value: "llm/prompt3/DESCRIPTIVE.txt",
        SceneEnum.DESCRIPTIVE.name: "llm/prompt3/DESCRIPTIVE.txt",
        SceneEnum.SCROLL.value: "llm/prompt3/SCROLL.txt",
        SceneEnum.SCROLL.name: "llm/prompt3/SCROLL.txt",
        SceneEnum.INPUT.value: "llm/prompt3/INPUT.txt",
        SceneEnum.INPUT.name: "llm/prompt3/INPUT.txt",
        ScenarioEnum.ACCESSIBILITY_KEYBOARD_NAVIGATION.value: "llm/prompt4/Accessibility & Keyboard Navigation.txt",
        ScenarioEnum.ACCESSIBILITY_KEYBOARD_NAVIGATION.name: "llm/prompt4/Accessibility & Keyboard Navigation.txt",
        ScenarioEnum.ADVERTISING_VERIFICATION_REPORTING.value: "llm/prompt4/Advertising Verification & Reporting.txt",
        ScenarioEnum.ADVERTISING_VERIFICATION_REPORTING.name: "llm/prompt4/Advertising Verification & Reporting.txt",
        ScenarioEnum.APPEARANCE_THEME_SETTINGS.value: "llm/prompt4/Appearance & Theme Settings.txt",
        ScenarioEnum.APPEARANCE_THEME_SETTINGS.name: "llm/prompt4/Appearance & Theme Settings.txt",
        ScenarioEnum.AUTHENTICATION_USER_PROFILE_MANAGEMENT.value: "llm/prompt4/Authentication & User Profile Management.txt",
        ScenarioEnum.AUTHENTICATION_USER_PROFILE_MANAGEMENT.name: "llm/prompt4/Authentication & User Profile Management.txt",
        ScenarioEnum.BROWSER_SETTINGS_CONFIGURATION.value: "llm/prompt4/Browser Settings & Configuration.txt",
        ScenarioEnum.BROWSER_SETTINGS_CONFIGURATION.name: "llm/prompt4/Browser Settings & Configuration.txt",
        ScenarioEnum.CAROUSEL_SLIDER_CONTROLS.value: "llm/prompt4/Carousel & Slider Controls.txt",
        ScenarioEnum.CAROUSEL_SLIDER_CONTROLS.name: "llm/prompt4/Carousel & Slider Controls.txt",
        ScenarioEnum.ENVIRONMENT_PRECONDITION_SETUP.value: "llm/prompt4/Environment & Precondition Setup.txt",
        ScenarioEnum.ENVIRONMENT_PRECONDITION_SETUP.name: "llm/prompt4/Environment & Precondition Setup.txt",
        ScenarioEnum.LOCALIZATION_INTERNATIONALIZATION.value: "llm/prompt4/Localization & Internationalization.txt",
        ScenarioEnum.LOCALIZATION_INTERNATIONALIZATION.name: "llm/prompt4/Localization & Internationalization.txt",
        ScenarioEnum.MEDIA_PLAYBACK_AUDIO_CONTROL.value: "llm/prompt4/Media Playback & Audio Control.txt",
        ScenarioEnum.MEDIA_PLAYBACK_AUDIO_CONTROL.name: "llm/prompt4/Media Playback & Audio Control.txt",
        ScenarioEnum.MODALS_POPUPS_NOTIFICATIONS_HANDLING.value: "llm/prompt4/Modals, Popups & Notifications Handling.txt",
        ScenarioEnum.MODALS_POPUPS_NOTIFICATIONS_HANDLING.name: "llm/prompt4/Modals, Popups & Notifications Handling.txt",
        ScenarioEnum.NAVIGATION_URL_REDIRECTION.value: "llm/prompt4/Navigation & URL Redirection.txt",
        ScenarioEnum.NAVIGATION_URL_REDIRECTION.name: "llm/prompt4/Navigation & URL Redirection.txt",
        ScenarioEnum.SEARCH_FUNCTIONALITY_SERP_MODULE_VALIDATION.value: "llm/prompt4/Search Functionality & SERP Module Validation.txt",
        ScenarioEnum.SEARCH_FUNCTIONALITY_SERP_MODULE_VALIDATION.name: "llm/prompt4/Search Functionality & SERP Module Validation.txt",
        ScenarioEnum.TAB_WINDOW_MANAGEMENT.value: "llm/prompt4/Tab & Window Management.txt",
        ScenarioEnum.TAB_WINDOW_MANAGEMENT.name: "llm/prompt4/Tab & Window Management.txt",
        ScenarioEnum.UI_VISIBILITY_LAYOUT_RENDERING_VERIFICATION.value: "llm/prompt4/UI Visibility, Layout & Rendering Verification.txt",
        ScenarioEnum.UI_VISIBILITY_LAYOUT_RENDERING_VERIFICATION.name: "llm/prompt4/UI Visibility, Layout & Rendering Verification.txt",
        ScenarioEnum.WIDGETS_TASKBAR_OS_LEVEL_INTEGRATIONS.value: "llm/prompt4/Widgets, Taskbar & OS-Level Integrations.txt",
        ScenarioEnum.WIDGETS_TASKBAR_OS_LEVEL_INTEGRATIONS.name: "llm/prompt4/Widgets, Taskbar & OS-Level Integrations.txt",
        ScenarioEnum.FUNCTIONAL_LAYOUT_SETTING.value: "llm/prompt4/Functional Layout Setting.txt",
        ScenarioEnum.FUNCTIONAL_LAYOUT_SETTING.name: "llm/prompt4/Functional Layout Setting.txt",
    }

    prompt_path = prompt_files.get(key)
    if not prompt_path:
        return None

    preferred = prompt_path.replace("llm/prompts/", "llm/prompt3/")
    try:
        if Path(resource_path(preferred)).exists():
            return preferred
    except Exception:
        # If resource resolution fails for any reason, fall back.
        pass

    return prompt_path
