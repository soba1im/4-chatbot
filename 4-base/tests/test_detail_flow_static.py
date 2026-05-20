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

    def test_first_cockpit_input_display_waits_eight_seconds(self):
        cockpit_input_scene = re.search(
            r"id: 26,.*?roleAuthImage: \"images/hateslop/name_input_display.png\",(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(cockpit_input_scene)
        self.assertIn("autoAdvanceSec: 8", cockpit_input_scene.group("body"))

    def test_name_phase_resets_question_power_once(self):
        render_scene = re.search(
            r"function renderScene\(index\) \{(?P<body>.*?)\n\s*\}\n\s*function playIntroSequence",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(render_scene)
        body = render_scene.group("body")
        cockpit_check = body.index("scene.bgImage.includes('cockpit_bg')")
        reset_check = body.index("if (!namePhaseUnlocked)")
        self.assertGreater(reset_check, cockpit_check)
        self.assertIn("questionCount = 0;", body)
        self.assertIn("currentBattery = 5;", body)
        self.assertIn("resetBatteryIcons();", body)

    def test_final_job_question_waits_then_shows_pico_before_job_input(self):
        self.assertIn(
            'autoAdvanceSec: (currentPhase === "job" && isQuestion && currentBattery === 0) ? 10',
            self.source,
        )
        self.assertIn('disableInput: currentPhase === "job" && isQuestion && currentBattery === 0,', self.source)
        self.assertIn(
            "질문을 위해 할당되었던 전력이 모두 소모되었어요. 이제는 정말 기억을 되찾아야하는 시간이에요.",
            self.source,
        )
        self.assertIn("이제 시동을 걸러 가볼까요?", self.source)
        self.assertRegex(
            self.source,
            r"scenes\.push\(newScene, jobExhaustScene1, jobExhaustScene2, jobInputScene\);",
        )
        self.assertNotIn("}, 2000);", self.source)

    def test_first_auth_failures_delay_with_input_disabled(self):
        job_fail_scene = re.search(
            r"id: 17,.*?roleAuthImage: \"images/hateslop/role_authentication_1st_fail.png\",(?P<body>.*?)\n\s*\}",
            self.source,
            re.DOTALL,
        )
        name_fail_scene = re.search(
            r"const failScene = \{(?P<body>.*?)roleAuthImage: \"images/hateslop/name_1st_fail.png\"(?P<tail>.*?)\n\s*\};",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(job_fail_scene)
        job_body = job_fail_scene.group("body")
        self.assertIn("showArrow: false", self.source[job_fail_scene.start():job_fail_scene.end()])
        self.assertIn("autoAdvanceSec: 5", job_body)
        self.assertIn("disableInput: true", job_body)

        self.assertIsNotNone(name_fail_scene)
        name_body = name_fail_scene.group("body") + name_fail_scene.group("tail")
        self.assertIn("showArrow: false", name_body)
        self.assertIn("autoAdvanceSec: 5", name_body)
        self.assertIn("disableInput: true", name_body)

    def test_job_first_failure_returns_to_role_input_display_after_hint(self):
        self.assertRegex(
            self.source,
            r"(?s)id: 21,.*?showArrow: true,.*?showRoleAuth: false,.*?isJobInputScene: false,",
        )
        self.assertRegex(
            self.source,
            r"(?s)id: 21\.5,.*?showArrow: false,.*?showPico: false,.*?showRoleAuth: true,.*?roleAuthImage: \"images/hateslop/role_input_display.png\",.*?isJobInputScene: true,",
        )

    def test_third_name_question_hint_waits_for_next_question_input(self):
        hint_scene = re.search(
            r"const hintScene2 = \{(?P<body>.*?)\n\s*\};",
            self.source,
            re.DOTALL,
        )

        self.assertIsNotNone(hint_scene)
        body = hint_scene.group("body")
        self.assertIn("이 흐름을 타서 빨리 다음 질문을 던져보세요!", body)
        self.assertIn("showArrow: false", body)
