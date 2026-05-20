import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from services.chatbot_service import ChatbotService


class RecommendedReplyTest(unittest.TestCase):
    def setUp(self):
        self.service = ChatbotService.__new__(ChatbotService)
        self.service.config = {}
        self.service.memory = None
        self.service.collection = None
        self.service._conversation_buffer = []

    def test_job_recommended_questions_return_pico_fixed_replies(self):
        cases = {
            "너는 나를 어떤 사용자로 기억해?": (
                "제 기록 속 당신은 실험자이자 문제해결자에 가까웠어요. "
                "뭐든지 논리적으로 분석하려고 했고, 평소엔 쾌활하고 조금 시끄러웠지만 일할 때는 놀랄 만큼 진지했죠. "
                "자동 시스템에 덜 기대고 직접 판단하려는 고집도 아주 선명하게 남아 있습니다."
            ),
            "내가 특히 잘한 건 뭐야?": (
                "문제가 생기면 우왕좌왕하기보다 원인을 잡아내는 데 강하셨어요. "
                "항로나 샘플 보관 쪽에서 일이 꼬였을 때도 좋은 아이디어를 꺼내 팀과 같이 풀어갔고, 기계공학 지식을 꽤 야무지게 써먹으셨죠. "
                "인정합니다, 그때의 당신은 꽤 믿음직했어요."
            ),
            "나는 주로 어떤 공간에 있었어?": (
                "기록상 자주 찍히는 위치는 조종실, 실험실, 그리고 침실이에요. "
                "하지만 딱 잘라 한 공간에만 있었다기보다는 선체 전체를 쉴 새 없이 돌아다녔습니다. "
                "동료들을 슬쩍슬쩍 귀찮게 하는 것도 꽤 즐기셨고요."
            ),
            "나는 너에게 뭘 자주 물었어?": (
                "실험용 부품이나 설계 이야기를 자주 물으셨어요. "
                "항로 계산법을 두고 저와 토론하기도 했고, 본인 설계회로에 대한 농담을 꽤 즐기셨습니다. "
                "그런데 그 농담들을 기억 못 하신다니, 흠, 피코 데이터베이스도 살짝 서운합니다."
            ),
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                response = self.service.generate_response(question, phase="job_question")

                self.assertEqual(response["answer"], expected)
                self.assertEqual(response["reply"], expected)
                self.assertTrue(response["isQuestion"])
                self.assertFalse(response["isJobAttempt"])
                self.assertEqual(response["imageType"], "none")

    def test_job_recommended_questions_accept_similar_wording(self):
        cases = {
            "너는 날 어떻게 기억해?": "제 기록 속 당신은 실험자이자 문제해결자에 가까웠어요.",
            "내가 잘했던 건 뭐였어?": "문제가 생기면 우왕좌왕하기보다 원인을 잡아내는 데 강하셨어요.",
            "내가 자주 있던 장소는 어디야?": "기록상 자주 찍히는 위치는 조종실, 실험실, 그리고 침실이에요.",
            "내가 너한테 자주 하던 질문은?": "실험용 부품이나 설계 이야기를 자주 물으셨어요.",
        }

        for question, expected_start in cases.items():
            with self.subTest(question=question):
                response = self.service.generate_response(question, phase="job_question")

                self.assertTrue(response["answer"].startswith(expected_start))
                self.assertTrue(response["isQuestion"])

    def test_name_recommended_questions_return_pico_fixed_replies(self):
        cases = {
            "내 인간관계는 어땠지?": (
                "사람들과 두루 잘 지내셨고, 문제가 생겨도 말부터 꼬이기보다 차분히 풀어내는 쪽에 가까웠어요. "
                "이상하게 사람들을 끌어당기는 매력도 있어서 주변 기록에 당신 이름이 자주 보입니다. "
                "침착한 얼굴로 판을 정리하는 모습, 피코 로그에도 꽤 선명하게 남아 있어요."
            ),
            "내 평소 습관을 알려줘.": (
                "기록하는 걸 정말 좋아하셨어요. "
                "거창한 보고서든 사소한 하루 조각이든, 늘 당신만의 방식으로 남겨두는 습관이 있었습니다. "
                "기억은 날아갔어도 기록 본능은 쉽게 안 지워지는 법이죠, 삐빅."
            ),
            "내 취미는 어땠지?": (
                "정확하게 알려드릴 수 없는 질문이군요. "
                "보안 프로토콜이 취미 칸을 살짝 가리고 있어서요. "
                "다만 이것만은 말할 수 있어요, 당신은 분명 예술적인 면모가 있는 사람이었습니다."
            ),
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                response = self.service.generate_response(question, phase="name_question")

                self.assertEqual(response["answer"], expected)
                self.assertEqual(response["reply"], expected)
                self.assertTrue(response["isQuestion"])
                self.assertFalse(response["isNameAttempt"])
                self.assertEqual(response["imageType"], "none")

    def test_name_recommended_questions_accept_similar_wording(self):
        cases = {
            "나는 사람들과 관계가 어땠어?": "사람들과 두루 잘 지내셨고",
            "내 기록 습관은 어땠어?": "기록하는 걸 정말 좋아하셨어요.",
            "취미 같은 건 뭐였어?": "정확하게 알려드릴 수 없는 질문이군요.",
        }

        for question, expected_start in cases.items():
            with self.subTest(question=question):
                response = self.service.generate_response(question, phase="name_question")

                self.assertTrue(response["answer"].startswith(expected_start))
                self.assertTrue(response["isQuestion"])

    def test_irrelevant_questions_get_varied_replies(self):
        replies = [
            self.service.generate_response(message, phase="job_question")["answer"]
            for message in ["오늘 점심 뭐야?", "날씨는 어때?", "농담 하나 해줘."]
        ]

        self.assertGreater(len(set(replies)), 1)


if __name__ == "__main__":
    unittest.main()
