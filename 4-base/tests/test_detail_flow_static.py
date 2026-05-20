import re
import unittest
from pathlib import Path


DETAIL_HTML = Path(__file__).resolve().parents[1] / "templates" / "detail.html"


class DetailFlowStaticTest(unittest.TestCase):
    def setUp(self):
        self.source = DETAIL_HTML.read_text(encoding="utf-8")

    def test_history_request_is_recorded_before_showing_record_scene(self):
        history_branch = re.search(
            r"if \(isChatRecordRequest\(userMessage\)\) \{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(history_branch)
        self.assertIn("recordQuestion(userMessage, LAST_CHAT_RECORD_TEXT);", history_branch.group("body"))

    def test_scripted_arrow_scene_input_advances_without_spending_question(self):
        send_click = re.search(
            r"async function onSendClick\(\) \{(?P<body>.*?)\n\s*\}\n\s*function dissolveToScene",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(send_click)
        body = send_click.group("body")
        self.assertLess(body.index("isScriptedAdvanceInput()"), body.index("decreaseBattery()"))
        self.assertIn("nextSceneLogic();", body)

    def test_chat_record_follow_up_waits_for_input(self):
        follow_up_scene = re.search(
            r"const followUpScene = \{(?P<body>.*?)\n\s*\};",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(follow_up_scene)
        body = follow_up_scene.group("body")
        self.assertIn('text: "다른 질문은 또 없으신가요?",', body)
        self.assertIn("showArrow: false", body)

    def test_input_overlay_does_not_block_visible_arrows(self):
        chat_container = re.search(
            r"\.chat-container\s*\{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )
        input_wrapper = re.search(
            r"\.custom-input-wrapper\s*\{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )
        input_bg = re.search(
            r"\.input-bg-img\s*\{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )
        input_content = re.search(
            r"\.input-content\s*\{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )
        user_input = re.search(
            r"#user-input\s*\{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )
        send_button = re.search(
            r"\.send-img-btn\s*\{(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(chat_container)
        self.assertIsNotNone(input_wrapper)
        self.assertIsNotNone(input_bg)
        self.assertIsNotNone(input_content)
        self.assertIsNotNone(user_input)
        self.assertIsNotNone(send_button)
        self.assertIn("pointer-events: none;", chat_container.group("body"))
        self.assertIn("pointer-events: none;", input_wrapper.group("body"))
        self.assertIn("pointer-events: none;", input_bg.group("body"))
        self.assertIn("pointer-events: none;", input_content.group("body"))
        self.assertIn("pointer-events: auto;", user_input.group("body"))
        self.assertIn("pointer-events: auto;", send_button.group("body"))
