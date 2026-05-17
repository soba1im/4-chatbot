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
