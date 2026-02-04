from enum import Enum

class IssueEnum(Enum):
    NO_ISSUE_FOUND = "No Issue found"
    ISSUE_FOUND = "Issue found"
    FEATURE_NOT_FOUND = "Feature not found"


class SceneEnum(Enum):
    UI_INTERACTION = "UI Interaction"
    STATE_VERIFICATION = "State Verification"
    CONDITIONAL = "Conditional"
    NAVIGATION = "Navigation"
    WAITING = "Waiting"
    DESCRIPTIVE = "Descriptive"
    INPUT = "Input"
    SCROLL = "Scroll"


class ScenarioEnum(Enum):
    ACCESSIBILITY_KEYBOARD_NAVIGATION = 'Accessibility & Keyboard Navigation'
    ADVERTISING_VERIFICATION_REPORTING = 'Advertising Verification & Reporting'
    APPEARANCE_THEME_SETTINGS = 'Appearance & Theme Settings'
    AUTHENTICATION_USER_PROFILE_MANAGEMENT = 'Authentication & User Profile Management'
    BROWSER_SETTINGS_CONFIGURATION = 'Browser Settings & Configuration'
    CAROUSEL_SLIDER_CONTROLS = 'Carousel & Slider Controls'
    ENVIRONMENT_PRECONDITION_SETUP = 'Environment & Precondition Setup'
    LOCALIZATION_INTERNATIONALIZATION = 'Localization & Internationalization'
    MEDIA_PLAYBACK_AUDIO_CONTROL = 'Media Playback & Audio Control'
    MODALS_POPUPS_NOTIFICATIONS_HANDLING = 'Modals, Popups & Notifications Handling'
    NAVIGATION_URL_REDIRECTION = 'Navigation & URL Redirection'
    SEARCH_FUNCTIONALITY_SERP_MODULE_VALIDATION = 'Search Functionality & SERP Module Validation'
    TAB_WINDOW_MANAGEMENT = 'Tab & Window Management'
    UI_VISIBILITY_LAYOUT_RENDERING_VERIFICATION = 'UI Visibility, Layout & Rendering Verification'
    WIDGETS_TASKBAR_OS_LEVEL_INTEGRATIONS = 'Widgets, Taskbar & OS-Level Integrations'
    FUNCTIONAL_LAYOUT_SETTING = 'Functional Layout Setting'

