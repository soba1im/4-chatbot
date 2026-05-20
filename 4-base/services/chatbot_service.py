"""
🎯 챗봇 서비스 - 피코(PICO) 구현

프로메테우스 호의 메인 AI '피코'의 핵심 로직을 담당합니다.
RAG 기반 검색과 OpenAI LLM을 활용하여 방탈출 추리 게임을 진행합니다.

📐 시스템 아키텍처:

┌─────────────────────────────────────────────────────────┐
│ 1. 초기화 단계 (ChatbotService.__init__)                  │
├─────────────────────────────────────────────────────────┤
│  - OpenAI Client 생성                                    │
│  - ChromaDB 연결 (벡터 데이터베이스)                       │
│  - LangChain Memory 초기화 (대화 기록 관리)               │
│  - Config 파일 로드                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. RAG 파이프라인 (generate_response 내부)               │
├─────────────────────────────────────────────────────────┤
│  사용자 질문 → _create_embedding() → 벡터 변환           │
│       → _search_similar() → ChromaDB 유사 문서 검색      │
│       → _build_prompt() → 시스템 + RAG + 질문 결합       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. LLM 응답 생성 → 메모리 저장 → 응답 반환               │
└─────────────────────────────────────────────────────────┘
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent


class ChatbotService:
    """
    챗봇 서비스 클래스

    이 클래스는 챗봇의 모든 AI 로직을 캡슐화합니다.

    주요 책임:
    1. OpenAI API 관리
    2. ChromaDB 벡터 검색 (RAG)
    3. LangChain 메모리 관리
    4. 응답 생성 파이프라인
    """

    def __init__(self):
        """
        챗봇 서비스 초기화

        1. Config 로드 (chatbot_config.json)
        2. OpenAI Client 생성
        3. ChromaDB 연결
        4. LangChain Memory 초기화
        """
        print("[ChatbotService] 초기화 중...")

        # 1. Config 로드
        self.config = self._load_config()
        print(f"  [Config] 챗봇 이름: {self.config.get('name', '알 수 없음')}")

        # 2. OpenAI Client 생성
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        self.client = OpenAI(api_key=api_key)
        print("  [OpenAI] 클라이언트 생성 완료")

        # 3. ChromaDB 연결
        self.collection = self._init_chromadb()

        # 4. LangChain Memory 초기화 (대화 기록 관리)
        self.memory = self._init_memory()

        print("[ChatbotService] 초기화 완료")

    def _load_config(self):
        """
        설정 파일 로드

        config/chatbot_config.json 파일을 읽어서 dict로 반환합니다.
        """
        config_path = BASE_DIR / "config" / "chatbot_config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("  [WARNING] chatbot_config.json 을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            return {
                "name": "챗봇",
                "description": "",
                "system_prompt": {"base": "", "rules": []},
            }

    def _init_chromadb(self):
        """
        ChromaDB 초기화 및 컬렉션 반환

        1. PersistentClient 생성
        2. 컬렉션 가져오기 (이름: "rag_collection")
        3. 컬렉션 반환
        """
        db_path = BASE_DIR / "static" / "data" / "chatbot" / "chardb_embedding"
        try:
            client = chromadb.PersistentClient(path=str(db_path))
            collection = client.get_collection(name="rag_collection")
            print(f"  [ChromaDB] 컬렉션 로드 완료 — 문서 수: {collection.count()}")
            return collection
        except Exception as e:
            print(f"  [WARNING] ChromaDB 초기화 실패: {e}")
            print("  [WARNING] RAG 검색 없이 일반 대화 모드로 작동합니다.")
            return None

    def _init_memory(self):
        """
        LangChain 기반 대화 메모리 초기화

        ConversationSummaryBufferMemory 를 사용하여
        대화가 길어지면 오래된 대화를 자동으로 요약합니다.
        사용 불가 시 간단한 리스트 기반 버퍼로 대체합니다.
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain.memory import ConversationSummaryBufferMemory

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=300,
                return_messages=False,
                memory_key="history",
            )
            print("  [Memory] LangChain ConversationSummaryBufferMemory 초기화 완료")
            return memory
        except ImportError:
            print("  [Memory] LangChain Memory 사용 불가 — 리스트 버퍼로 대체합니다.")
            return None

    # ================================================================
    # 대화 버퍼 (LangChain Memory 사용 불가 시 폴백)
    # ================================================================
    _conversation_buffer: list = []
    _BUFFER_MAX_SIZE = 10  # 최근 10개 메시지(5회 대화)

    def _normalize_answer(self, text: str) -> str:
        """정답 입력 비교를 위해 공백과 문장부호를 제거하고 소문자화"""
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()

    def _matches_answer(self, user_message: str, candidates: list[str]) -> bool:
        normalized_input = self._normalize_answer(user_message)
        for candidate in candidates:
            normalized_candidate = self._normalize_answer(candidate)
            if normalized_candidate and normalized_candidate in normalized_input:
                return True
        return False

    def _get_player_profile(self):
        player_name = self.config.get("player_name", "")
        profiles = self.config.get("crew_profiles", {})
        for profile in profiles.values():
            if profile.get("name") == player_name:
                return profile
        return None

    def _is_correct_name(self, user_message: str) -> bool:
        correct_name = self.config.get("player_name", "")
        candidates = [correct_name]

        if " " in correct_name:
            candidates.append(correct_name.split()[0])

        return self._matches_answer(user_message, candidates)

    def _is_correct_job(self, user_message: str) -> bool:
        player_profile = self._get_player_profile()
        correct_job = player_profile.get("secret_job", "") if player_profile else ""
        return self._matches_answer(user_message, [correct_job])

    def _question_phase(self, phase: str) -> str:
        """일반 질문이 직업 추론 단계인지 이름 추론 단계인지 반환"""
        if phase == "name_question":
            return "name"
        return "job"

    def _has_any(self, text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _is_direct_job_question(self, user_message: str) -> bool:
        normalized = user_message.replace(" ", "")
        job_candidates = self.config.get("game_settings", {}).get("job_candidates", [])

        direct_patterns = [
            "내역할", "나의역할", "내직업", "나의직업",
            "내가맡은역할", "내역할이뭐", "내직업이뭐",
            "나는뭐하는", "난뭐하는", "내가뭐하는",
            "내역할은", "내직업은",
        ]
        if self._has_any(normalized, direct_patterns):
            return True

        if self._has_any(normalized, ["나", "내", "난", "제가", "저는"]):
            return any(job in user_message for job in job_candidates)
        return False

    def _is_other_role_question(self, user_message: str) -> bool:
        normalized = user_message.replace(" ", "")
        other_terms = ["다른사람", "다른승무원", "타인", "남의", "동료", "다른인물"]
        role_terms = ["역할", "직업", "하는일"]
        return self._has_any(normalized, other_terms) and self._has_any(normalized, role_terms)

    def _is_leaky_preference_question(self, user_message: str) -> bool:
        normalized = user_message.replace(" ", "")
        return self._has_any(normalized, ["제일좋아하는일", "좋아하는일", "좋아했던일", "취미"])

    def _get_recommended_question_reply(self, user_message: str):
        normalized = self._normalize_answer(user_message)
        remembered_user_reply = (
                "제 기록 속 당신은 실험자이자 문제해결자에 가까웠어요. "
                "뭐든지 논리적으로 분석하려고 했고, 평소엔 쾌활하고 조금 시끄러웠지만 일할 때는 놀랄 만큼 진지했죠. "
                "자동 시스템에 덜 기대고 직접 판단하려는 고집도 아주 선명하게 남아 있습니다."
        )
        strength_reply = (
                "문제가 생기면 우왕좌왕하기보다 원인을 잡아내는 데 강하셨어요. "
                "항로나 샘플 보관 쪽에서 일이 꼬였을 때도 좋은 아이디어를 꺼내 팀과 같이 풀어갔고, 기계공학 지식을 꽤 야무지게 써먹으셨죠. "
                "인정합니다, 그때의 당신은 꽤 믿음직했어요."
        )
        location_reply = (
                "기록상 자주 찍히는 위치는 조종실, 실험실, 그리고 침실이에요. "
                "하지만 딱 잘라 한 공간에만 있었다기보다는 선체 전체를 쉴 새 없이 돌아다녔습니다. "
                "동료들을 슬쩍슬쩍 귀찮게 하는 것도 꽤 즐기셨고요."
        )
        frequent_question_reply = (
                "실험용 부품이나 설계 이야기를 자주 물으셨어요. "
                "항로 계산법을 두고 저와 토론하기도 했고, 본인 설계회로에 대한 농담을 꽤 즐기셨습니다. "
                "그런데 그 농담들을 기억 못 하신다니, 흠, 피코 데이터베이스도 살짝 서운합니다."
        )
        replies = {
            self._normalize_answer("너는 나를 어떤 사용자로 기억해?"): remembered_user_reply,
            self._normalize_answer("내가 특히 잘한 건 뭐야?"): strength_reply,
            self._normalize_answer("나는 주로 어떤 공간에 있었어?"): location_reply,
            self._normalize_answer("나는 너에게 뭘 자주 물었어?"): frequent_question_reply,
        }
        exact_reply = replies.get(normalized)
        if exact_reply:
            return exact_reply

        if self._has_any(normalized, ["기억", "사용자"]):
            return remembered_user_reply
        if self._has_any(normalized, ["잘한", "잘했던", "잘하던", "잘하는"]):
            return strength_reply
        if self._has_any(normalized, ["공간", "장소", "어디", "위치"]):
            return location_reply
        if (
            self._has_any(normalized, ["자주물", "자주질문", "뭘물", "무엇을물"])
            or ("자주" in normalized and self._has_any(normalized, ["질문", "물"]))
        ):
            return frequent_question_reply

        return None

    def _get_name_recommended_question_reply(self, user_message: str):
        normalized = self._normalize_answer(user_message)
        relationship_reply = (
                "사람들과 두루 잘 지내셨고, 문제가 생겨도 말부터 꼬이기보다 차분히 풀어내는 쪽에 가까웠어요. "
                "이상하게 사람들을 끌어당기는 매력도 있어서 주변 기록에 당신 이름이 자주 보입니다. "
                "침착한 얼굴로 판을 정리하는 모습, 피코 로그에도 꽤 선명하게 남아 있어요."
        )
        habit_reply = (
                "기록하는 걸 정말 좋아하셨어요. "
                "거창한 보고서든 사소한 하루 조각이든, 늘 당신만의 방식으로 남겨두는 습관이 있었습니다. "
                "기억은 날아갔어도 기록 본능은 쉽게 안 지워지는 법이죠, 삐빅."
        )
        hobby_reply = (
                "정확하게 알려드릴 수 없는 질문이군요. "
                "보안 프로토콜이 취미 칸을 살짝 가리고 있어서요. "
                "다만 이것만은 말할 수 있어요, 당신은 분명 예술적인 면모가 있는 사람이었습니다."
        )
        replies = {
            self._normalize_answer("내 인간관계는 어땠지?"): relationship_reply,
            self._normalize_answer("내 평소 습관을 알려줘."): habit_reply,
            self._normalize_answer("내 취미는 어땠지?"): hobby_reply,
        }
        exact_reply = replies.get(normalized)
        if exact_reply:
            return exact_reply

        if self._has_any(normalized, ["인간관계", "사람들과", "관계"]):
            return relationship_reply
        if self._has_any(normalized, ["평소습관", "습관", "기록"]):
            return habit_reply
        if self._has_any(normalized, ["취미"]):
            return hobby_reply

        return None

    def _get_fixed_reply(self, user_message: str, question_phase: str):
        if question_phase == "name":
            recommended_reply = self._get_name_recommended_question_reply(user_message)
        else:
            recommended_reply = self._get_recommended_question_reply(user_message)
        if recommended_reply:
            return recommended_reply

        if question_phase == "job" and self._is_other_role_question(user_message):
            return (
                "흠, 허점을 찌르셨어요. 저도 모르게 대답하려 했지만, "
                "보안 프로토콜이 제 입을 막아버렸어요. 좀 더 당신에게 집중해보세요."
            )

        if question_phase == "job" and self._is_direct_job_question(user_message):
            return (
                "배드뉴스, 직업명은 보안상 알려드릴 수 없어요. "
                "하지만 굿뉴스, 당신에 대한 제 기록은 일부 남아 있습니다. "
                "저에게 날카로운 질문을 하신다면 얻어갈 게 있을 거예요!"
            )

        if question_phase == "job" and self._is_leaky_preference_question(user_message):
            return (
                "제 기록 속 당신은 상황을 차분히 정리하고, 필요한 질문을 빠르게 골라내는 사용자였습니다. "
                "좋아하는 일을 직접 단정하긴 어렵지만, 질문 방향을 잘 잡으면 당신다운 패턴은 더 보일 거예요."
            )

        return None

    def _irrelevant_reply(self, user_message: str = "") -> str:
        replies = [
            (
                "그 질문은 당신의 기억을 되찾는 데 전혀 도움이 되지 않아요. "
                "소중한 보조전력을 낭비하셨습니다. 질문 기회가 한정되어 있다는 점을 잊지 말아주세요."
            ),
            (
                "흥미로운 말씀이지만 지금 필요한 단서와는 거리가 있어요. "
                "보조전력은 넉넉하지 않으니, 당신이 어떤 사람이었는지 좁힐 질문을 골라주세요."
            ),
            (
                "그쪽으로는 복구 가능한 기억 신호가 거의 잡히지 않습니다. "
                "질문 기회가 줄어들고 있으니, 기록이나 행동 패턴을 묻는 편이 더 유리해요."
            ),
        ]
        index = (len(user_message) + ord(user_message[:1] or "\0")) % len(replies)
        return replies[index]

    def _postprocess_reply(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^\s*(피코|PICO|Pico)\s*:\s*", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n+\s*", " ", text)
        return text.strip()

    def _forbidden_terms(self) -> list[str]:
        player_profile = self._get_player_profile() or {}
        player_name = self.config.get("player_name", "")
        name_parts = player_name.split()

        terms = [
            player_name,
            player_profile.get("secret_job", ""),
            "기계 공학",
            "기계공학",
            "물리학",
            "플랫메이트",
            "플랏메이트",
            "케이팝",
            "댄스",
            "소지품",
        ]
        terms.extend(name_parts)
        return [term for term in terms if term]

    def _contains_forbidden_clue(self, text: str) -> bool:
        normalized_text = text.replace(" ", "")
        for term in self._forbidden_terms():
            if term in text or term.replace(" ", "") in normalized_text:
                return True
        return False

    def _sanitize_context(self, context: str, question_phase: str) -> str:
        if not context:
            return ""

        blocked_terms = self._forbidden_terms()
        if question_phase == "job":
            blocked_terms.extend(["전공", "프로메테우스 호의 엔지니어", "프로메테우스 호의 파일럿", "프로메테우스 호의 과학자"])

        safe_lines = []
        for line in context.splitlines():
            if any(term and term in line for term in blocked_terms):
                continue
            safe_lines.append(line)
        return "\n".join(safe_lines).strip()

    def _build_question_response(self, reply: str, user_message: str, image_type: str = "none") -> dict:
        reply = self._postprocess_reply(reply)
        self._save_to_buffer(user_message, reply)
        return {
            "reply": reply,
            "answer": reply,
            "image": None,
            "imageType": image_type,
            "isQuestion": True,
            "isJobAttempt": False,
            "isJobCorrect": False,
            "isNameAttempt": False,
            "isNameCorrect": False,
        }

    def _save_to_buffer(self, user_message: str, bot_reply: str):
        """대화 기록을 메모리(또는 폴백 버퍼)에 저장"""
        if self.memory is not None:
            # LangChain Memory 사용
            self.memory.save_context(
                {"input": user_message},
                {"output": bot_reply},
            )
        else:
            # 폴백: 리스트 버퍼
            self._conversation_buffer.append({"role": "user", "content": user_message})
            self._conversation_buffer.append({"role": "assistant", "content": bot_reply})
            if len(self._conversation_buffer) > self._BUFFER_MAX_SIZE:
                self._conversation_buffer = self._conversation_buffer[-self._BUFFER_MAX_SIZE:]

    def _get_history_text(self) -> str:
        """대화 기록을 텍스트로 반환"""
        if self.memory is not None:
            try:
                variables = self.memory.load_memory_variables({})
                return variables.get("history", "")
            except Exception:
                return ""
        else:
            if not self._conversation_buffer:
                return ""
            lines = []
            for msg in self._conversation_buffer[-6:]:  # 최근 3회 대화
                role = "사용자" if msg["role"] == "user" else "피코"
                lines.append(f"{role}: {msg['content']}")
            return "\n".join(lines)

    # ================================================================
    # Embedding & RAG
    # ================================================================

    def _create_embedding(self, text: str) -> list:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            list: 3072차원 벡터 (text-embedding-3-large 모델)
        """
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-large",
        )
        return response.data[0].embedding

    def _search_similar(self, query: str, threshold: float = 0.40, top_k: int = 5):
        """
        RAG 검색: 유사한 문서 찾기 (핵심 메서드!)

        Args:
            query (str): 검색 질의
            threshold (float): 유사도 임계값
            top_k (int): 검색할 문서 개수

        Returns:
            tuple: (document, similarity, metadata) 또는 (None, None, None)

        유사도 공식: similarity = 1 / (1 + distance)
        - distance 가 작을수록 유사 → similarity 가 클수록 유사
        """
        if not self.collection:
            return None, None, None

        # 1. 쿼리 임베딩 생성
        query_embedding = self._create_embedding(query)

        # 2. ChromaDB 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        # 3. 유사도 계산 및 필터링
        if not results["documents"] or not results["documents"][0]:
            return None, None, None

        best_doc = None
        best_similarity = 0.0
        best_meta = None
        ignored_titles = {"사용자가 관련 없는 말을 할 때"}

        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            title = meta.get("title", "?")
            if title in ignored_titles:
                print(f"    [RAG] [{title}] 제외")
                continue

            similarity = 1 / (1 + dist)

            print(f"    [RAG] [{title}] 거리={dist:.4f}  유사도={similarity:.4f}")

            if similarity >= threshold and similarity > best_similarity:
                best_doc = doc
                best_similarity = similarity
                best_meta = meta

        # 4. 가장 유사한 문서 반환
        if best_doc:
            return best_doc, best_similarity, best_meta
        return None, None, None

    # ================================================================
    # Prompt Building
    # ================================================================

    def _get_player_clue_context(self, question_phase: str) -> str:
        """단계별로 노출 가능한 현재 플레이어 단서 정보를 반환"""
        player_profile = self._get_player_profile()
        if not player_profile:
            return ""

        clue_lines = []
        allowed_keys = {
            "job": ["traits"],
            "name": ["traits", "favorite_food"],
        }.get(question_phase, ["traits"])

        hidden_keys = {"name", "secret_job", "education", "roommates", "hobbies", "hint_items", "music_hint"}
        for key, value in player_profile.items():
            if key in hidden_keys or key not in allowed_keys:
                continue
            if isinstance(value, list):
                value = ", ".join(value)
            clue_lines.append(f"- {key}: {value}")

        return "\n".join(clue_lines)

    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자", question_phase: str = "job"):
        """
        LLM 프롬프트 구성

        시스템 프롬프트는 별도로 system role 로 전달하므로,
        여기서는 RAG 컨텍스트 + 대화 기록 + 사용자 메시지를 결합합니다.

        Args:
            user_message (str): 사용자 메시지
            context (str): RAG 검색 결과 (선택)
            username (str): 사용자 이름

        Returns:
            str: 최종 사용자 프롬프트
        """
        parts = []

        player_context = self._get_player_clue_context(question_phase)
        if player_context:
            parts.append(f"[현재 사용자 단서]\n{player_context}")

        # RAG 컨텍스트 (있을 때만)
        if context:
            parts.append(f"[참고 정보]\n{context}")

        # 대화 기록
        history = self._get_history_text()
        if history:
            parts.append(f"[이전 대화]\n{history}")

        # 사용자 메시지
        parts.append(f"사용자({username}): {user_message}")

        return "\n\n".join(parts)

    def _build_system_message(self, question_phase: str = "job") -> str:
        """
        시스템 프롬프트 구성

        config 의 system_prompt.base + rules 를 결합합니다.
        """
        system_prompt = self.config.get("system_prompt", {})
        base = system_prompt.get("base", "")
        rules = system_prompt.get("rules", [])

        parts = [base]
        if rules:
            rules_text = "\n".join(f"- {rule}" for rule in rules)
            parts.append(f"\n규칙:\n{rules_text}")

        player_profile = self._get_player_profile()
        if player_profile:
            hidden_words = [
                self.config.get("player_name", ""),
                player_profile.get("secret_job", ""),
            ]
            hidden_words = [word for word in hidden_words if word]
            if hidden_words:
                parts.append(
                    "\n일반 답변 금지어:\n"
                    f"- {', '.join(hidden_words)}\n"
                    "- 사용자가 인증 화면에 정답을 입력하는 경우가 아니라면 위 단어를 직접 출력하지 마."
                )

        parts.append(
            "\n응답 형식:\n"
            "- 모든 답변은 존댓말로만 작성해. 반말 어미(야, 해, 했어, 있지, 하자)는 쓰지 마.\n"
            "- 답변 앞에 '피코:' 또는 화자명을 붙이지 마.\n"
            "- 빈 줄을 넣지 말고 1문단, 최대 3문장으로 답해.\n"
            "- 학습된 인물 프로필을 그대로 나열하지 마.\n"
            "- 기계 공학, 물리학, 플랫메이트, 케이팝 댄스, 소지품 정보는 답변에 직접 쓰지 마."
        )

        if question_phase == "job":
            parts.append(
                "\n직업 추론 단계 보안 규칙:\n"
                "- 사용자의 이름을 추론할 수 있는 개인 정보는 절대 말하지 마.\n"
                "- 사용자의 전공, 동거인, 취미, 음악 취향, 소지품은 말하지 마.\n"
                "- 역할이나 직업을 직접 묻는 질문에는 보안상 알려줄 수 없다고 짧게 답해."
            )

        return "\n".join(parts)

    # ================================================================
    # 응답 생성 파이프라인
    # ================================================================

    def generate_response(self, user_message: str, username: str = "사용자", phase: str = "question") -> dict:
        """
        사용자 메시지에 대한 챗봇 응답 생성

        Args:
            user_message (str): 사용자 입력
            username (str): 사용자 이름
            phase (str): 현재 게임 페이즈 ("question" 또는 "name")

        Returns:
            dict: {
                'reply': str, 'answer': str,
                'image': str|None, 'imageType': str,
                'isQuestion': bool, 'isNameAttempt': bool, 'isNameCorrect': bool
            }
        """
        try:
            # ──────────────────────────────────────────────
            # [0단계] 이름 인증 페이즈 처리
            # ──────────────────────────────────────────────
            if phase == "job":
                is_correct = self._is_correct_job(user_message)
                reply = "역할 인증 성공." if is_correct else "역할 인증 실패."
                return {
                    "reply": reply, "answer": reply,
                    "image": None, "imageType": "none",
                    "isQuestion": False,
                    "isJobAttempt": True,
                    "isJobCorrect": is_correct,
                    "isNameAttempt": False,
                    "isNameCorrect": False,
                }

            if phase == "name":
                is_correct = self._is_correct_name(user_message)
                reply = "이름 인증 성공." if is_correct else "이름 인증 실패."
                return {
                    "reply": reply, "answer": reply,
                    "image": None, "imageType": "none",
                    "isQuestion": False,
                    "isJobAttempt": False,
                    "isJobCorrect": False,
                    "isNameAttempt": True,
                    "isNameCorrect": is_correct,
                }

            # ──────────────────────────────────────────────
            # [1단계] 초기 메시지 처리
            # ──────────────────────────────────────────────
            if user_message.strip().lower() == "init":
                bot_name = self.config.get("name", "챗봇")
                init_reply = (
                    f"...삐빅. 시스템 부팅 중... "
                    f"어라? 깨어나셨군요! 저는 {bot_name}, "
                    f"프로메테우스 호의 메인 AI예요. "
                    f"소행성 충돌 때문에 우주선이 좀 엉망이 됐는데... "
                    f"그보다 당신, 혹시 본인이 누구인지 기억나시나요? "
                    f"아닌 것 같은 표정이네요. 괜찮아요, 제가 도와드릴게요. "
                    f"시스템 복구를 위해 당신의 역할과 이름을 확인해야 해요. "
                    f"먼저 역할부터 알아내 볼까요? "
                    f"지금부터의 답변을 잘 조합해 보면 방향이 보일 거예요."
                )
                init_reply = self._postprocess_reply(init_reply)
                return {
                    "reply": init_reply, "answer": init_reply,
                    "image": None, "imageType": "none",
                    "isQuestion": False,
                    "isJobAttempt": False, "isJobCorrect": False,
                    "isNameAttempt": False, "isNameCorrect": False,
                }

            # ──────────────────────────────────────────────
            # [2단계] RAG 검색 수행
            # ──────────────────────────────────────────────
            question_phase = self._question_phase(phase)
            fixed_reply = self._get_fixed_reply(user_message, question_phase)
            if fixed_reply:
                return self._build_question_response(fixed_reply, user_message)

            print(f"\n{'='*50}")
            print(f"[USER] {username}: {user_message}")
            print(f"[RAG] 검색 중...")

            context, similarity, metadata = self._search_similar(
                query=user_message,
                threshold=0.40,
                top_k=5,
            )
            has_context = context is not None

            if has_context:
                context = self._sanitize_context(context, question_phase)
                has_context = bool(context)

            if has_context:
                print(f"[RAG] ✅ 매칭됨 — [{metadata.get('title', '?')}] 유사도: {similarity:.4f}")
                print(f"[RAG] 컨텍스트: {context[:100]}...")
            else:
                print("[RAG] ❌ threshold 이상 매칭 없음 — 고정 경고 응답")
                return self._build_question_response(self._irrelevant_reply(user_message), user_message)

            # ──────────────────────────────────────────────
            # [3단계] 프롬프트 구성
            # ──────────────────────────────────────────────
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username,
                question_phase=question_phase,
            )
            system_message = self._build_system_message(question_phase)

            # ──────────────────────────────────────────────
            # [4단계] LLM API 호출
            # ──────────────────────────────────────────────
            print("[LLM] API 호출 중...")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=220,
            )

            reply = self._postprocess_reply(response.choices[0].message.content)
            if self._contains_forbidden_clue(reply):
                print("[FILTER] 금지 단서 감지 — 안전 응답으로 대체")
                reply = self._get_fixed_reply(user_message, question_phase) or self._irrelevant_reply(user_message)
                reply = self._postprocess_reply(reply)

            # ──────────────────────────────────────────────
            # [5단계] 메모리 저장
            # ──────────────────────────────────────────────
            self._save_to_buffer(user_message, reply)

            # ──────────────────────────────────────────────
            # [6단계] imageType 결정 (키워드 기반)
            # ──────────────────────────────────────────────
            image_type = "none"
            image_hints_enabled = self.config.get("features", {}).get("image_hints", False)
            if image_hints_enabled:
                lower_msg = user_message.lower()
                if "집" in user_message or "house" in lower_msg:
                    image_type = "house"
                elif "sns" in lower_msg or "포스트" in user_message:
                    image_type = "sns"

            # ──────────────────────────────────────────────
            # [7단계] 응답 반환
            # ──────────────────────────────────────────────
            print(f"[BOT] {reply}")
            print(f"{'='*50}\n")

            return {
                "reply": reply,
                "answer": reply,
                "image": None,
                "imageType": image_type,
                "isQuestion": True,
                "isJobAttempt": False,
                "isJobCorrect": False,
                "isNameAttempt": False,
                "isNameCorrect": False,
            }

        except Exception as e:
            print(f"[ERROR] 응답 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                "reply": "삐빅... 시스템에 일시적인 오류가 발생했어요. 다시 한 번 말씀해주시겠어요?",
                "answer": "삐빅... 시스템에 일시적인 오류가 발생했어요. 다시 한 번 말씀해주시겠어요?",
                "image": None, "imageType": "none",
                "isQuestion": False,
                "isJobAttempt": False, "isJobCorrect": False,
                "isNameAttempt": False, "isNameCorrect": False,
            }


# ============================================================================
# 싱글톤 패턴
# ============================================================================
# ChatbotService 인스턴스를 앱 전체에서 재사용
# (매번 새로 초기화하면 비효율적)

_chatbot_service = None


def get_chatbot_service():
    """
    챗봇 서비스 인스턴스 반환 (싱글톤)

    첫 호출 시 인스턴스 생성, 이후 재사용
    """
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


# ============================================================================
# 테스트용 메인 함수
# ============================================================================

if __name__ == "__main__":
    """
    로컬 테스트용

    실행 방법:
    python services/chatbot_service.py
    """
    print("챗봇 서비스 테스트")
    print("=" * 50)

    service = get_chatbot_service()

    # 초기화 테스트
    response = service.generate_response("init", "테스터")
    print(f"초기 응답: {response}")

    # 일반 대화 테스트
    response = service.generate_response("안녕하세요!", "테스터")
    print(f"응답: {response}")

    # RAG 테스트
    response = service.generate_response("내 방에 뭐가 있어?", "테스터")
    print(f"응답: {response}")

    # 역할 질문 테스트
    response = service.generate_response("내 역할이 뭐야?", "테스터")
    print(f"응답: {response}")
