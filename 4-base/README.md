# 🚀 Project Ariadne: AI 방탈출 챗봇 게임

---

# 📖 프로젝트 개요
**Project Ariadne**는 사용자가 갇힌 우주선 공간에서 미지의 AI 로봇 '피코(Pico)'와 대화하며 단서를 찾고 탈출하는 **웹 기반 인터랙티브 방탈출 게임**입니다. RAG(검색 증강 생성) 기술을 활용하여 AI가 게임의 세계관 내에서만 답변하도록 설계되었으며, 프레임워크 없이 순수 Vanilla JS와 CSS3만으로 몰입감 있는 SF UI를 구현했습니다.

---

# 🏗️ 시스템 아키텍처

## ## 🏗️ 전체 구조도
1. **사용자 브라우저**: `detail.html` (단일 파일 SPA)
2. **프론트엔드 엔진**: 씬 상태 머신 (`scenes[]` + `renderScene`)
3. **통신 레이어**: `Fetch API` ↔ `/api/chat`
4. **백엔드 서버**: `Flask (app.py)`
5. **핵심 서비스**: `ChatbotService` (싱글톤 패턴)
6. **AI 파이프라인**: 
   - `text-embedding-3-large` (OpenAI)
   - `ChromaDB` (벡터 검색)
   - `gpt-4o-mini` (응답 생성)

## ## 🔄 데이터 흐름 (Data Flow)
# ### 1단계: 사용자 입력
- 사용자가 채팅 입력 (현재 페이즈: `job` / `name` / `question` 정보를 포함)

# ### 2단계: Flask 서버 수신
- `/api/chat` 경로로 데이터 전송 및 `ChatbotService` 호출

# ### 3단계: RAG 파이프라인 가동
- **임베딩 생성**: 사용자 질문을 벡터로 변환 (`text-embedding-3-large`)
- **유사도 검색**: ChromaDB에서 관련 단서 문서 추출 (Threshold 0.40 적용)

# ### 4단계: 컨텍스트 결합 및 추론
- **프롬프트 빌딩**: [시스템 지침] + [RAG 검색 단서] + [이전 대화 기록] 결합
- **LLM 호출**: `gpt-4o-mini`를 통해 최적의 답변 생성

# ### 5단계: 응답 반환 및 렌더링
- **JSON 응답**: 답변 텍스트 및 정답 여부(`isCorrect`) 반환
- **씬 업데이트**: JS 상태 머신이 응답 결과에 따라 다음 씬으로 분기 렌더링

---

# # 📂 프로젝트 구조 (Project Structure)

```text
4-base/
├── app.py                               # Flask 애플리케이션 (라우팅)
├── build_embedding.py                   # 임베딩 빌드 스크립트
├── services/
│   ├── __init__.py
│   └── chatbot_service.py               # 핵심 AI 로직 (RAG, 챗봇 서비스)
├── config/
│   └── chatbot_config.json              # 챗봇 설정 (방 정보, Phase 설정)
├── static/
│   ├── data/
│   │   └── chatbot/
│   │       └── chardb_text/             # 텍스트 데이터 (RAG 소스)
│   │           ├── crew_profiles.txt
│   │           ├── game_scenarios.txt
│   │           ├── job_clues.txt
│   │           ├── locations_and_clues.txt
│   │           └── ship_info.txt
│   ├── images/
│   │   ├── chatbot/                     # 챗봇 이미지 (README용)
│   │   └── hateslop/                    # 게임 UI 이미지 (배경, 아이콘 등 49개)
│   │       ├── pico.png
│   │       ├── logo.png
│   │       ├── corridor_bg.jpg
│   │       └── ...
│   ├── videos/
│   │   └── hateslop/                    # 게임 영상 (오프닝, 엔딩 등 4개)
│   │       ├── opening.mp4
│   │       ├── happyending.mp4
│   │       └── ...
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── chatbot.js
│       └── chatbot_single.js            # 프론트엔드 핵심 상태 관리 로직
├── templates/
│   ├── index.html                       # 메인 페이지
│   ├── detail.html                      # 게임 플레이 상세 페이지 (SPA)
│   └── chat.html                        # 채팅 페이지 레이아웃
├── Dockerfile                           # 도커 이미지 생성 설정
├── docker-compose.yml                   # 멀티 컨테이너 관리 설정
├── requirements.txt                     # 필요 라이브러리 목록
├── .env.example                         # 환경변수 설정 가이드
├── README.md                            # 프로젝트 메인 가이드
├── ARCHITECTURE.md                      # 상세 시스템 구조 문서
├── ADVANCED_TOPICS.md                   # 고급 기술 설정 가이드
├── DOCKER-GUIDE.md                      # 도커 실행 가이드
└── RENDER-GUIDE.md                      # 배포 가이드

---

# 🛠️ 기술 스택

## ## 💻 Backend
- **Flask 3.0**: RESTful API 및 Jinja2 템플릿 엔진
- **OpenAI API**: gpt-4o-mini / text-embedding-3-large
- **ChromaDB**: PersistentClient 기반 로컬 벡터 DB
- **LangChain**: 대화 기록 요약 및 버퍼 메모리 관리

## ## 🎨 Frontend
- **Vanilla JavaScript**: ES6+ 기반 상태 관리 로직
- **HTML5 / CSS3**: Glassmorphism UI, Keyframe 애니메이션
- **Web APIs**: Fetch API, HTMLVideoElement

## ## 🌐 Infrastructure & VCS
- **Infrastructure**: Docker
- **Version Control**: Git, GitHub

---

# 💡 기술 선택 이유

## ## 💾 ChromaDB를 선택한 이유
- 파이썬 환경과 완벽하게 통합되며 별도 서버 구축 없이 임베디드 모드로 사용 가능합니다.
- 벡터 검색을 통해 사용자의 의도를 파악하고 가장 적절한 게임 단서를 제공하기에 최적입니다.

## ## 🧠 RAG 패턴을 적용한 이유
- LLM이 알지 못하는 게임 고유의 시나리오 지식을 실시간으로 주입하여 '환각 현상'을 방지하기 위함입니다.

## ## ⚡ Vanilla JS + 단일 HTML을 선택한 이유
- 외부 라이브러리 의존성을 줄이고, 브라우저의 기본 동작 원리와 상태 관리의 본질을 직접 구현하며 학습하기 위해 선택했습니다.

---

# ⚠️ 개발 중 문제점 & ✅ 해결 방법

## ## ⚠️ 문제 1: 씬 전환 시 인덱스 동기화 오류
- **현상**: 특정 장면으로 이동 후 다음 버튼 클릭 시 이전 인덱스로 되돌아가는 현상.
- **원인**: `renderScene` 함수 내에서 전역 변수 `currentSceneIndex` 업데이트 누락.
- **해결**: 함수 실행 시 인덱스 동기화 코드를 최상단에 배치하여 해결.

## ## ⚠️ 문제 2: 에셋 하드코딩으로 인한 UI 꼬임
- **현상**: 복도 씬에서 질문 시 배경이 조종실로 강제 전환되는 버그.
- **원인**: 동적 씬 생성 로직에서 배경 이미지를 조종실로 고정해둠.
- **해결**: 현재 페이즈(`currentPhase`)를 체크하여 배경을 조건부로 할당하도록 로직 수정.

## ## ⚠️ 문제 3: 게임 재시작 시 상태 유지 문제
- **현상**: 새로고침 없이 재시작 시 이전 질문 횟수가 남아있는 문제.
- **해결**: 게임 시작 함수 호출 시 모든 전역 변수와 배열 길이를 초기화하는 코드 추가.

---

# 🚀 성능 개선 노력
- **[항목 1]**: OpenAI API 호출 시 `max_tokens` 최적화를 통해 응답 대기 시간 단축.
- **[항목 2]**: 이미지 프리로딩(Preloading)을 통해 장면 전환 시 끊김 현상 제거.

---

# 😔 아쉬웠던 점
- **[항목 1]**: 프로젝트 규모가 커짐에 따라 Vanilla JS의 유지보수 한계를 느낌.
- **[항목 2]**: 시간 제약으로 인해 더 화려한 멀티미디어(사운드 효과 등) 요소 추가 미흡.

---

# 🤔 회고 및 성찰
- **기술적 성장**: RAG 구조를 밑바닥부터 구현하며 데이터 흐름의 전체 메커니즘을 완벽히 이해하게 되었습니다.
- **협업 경험**: Git Merge Conflict를 해결하며 팀원과의 코드 컨벤션과 소통의 중요성을 체감했습니다.
- **향후 계획**: 이번 프로젝트에서 느낀 상태 관리의 어려움을 발판 삼아, 다음 프로젝트에서는 **React**를 도입하여 컴포넌트 기반 개발을 시도할 예정입니다.

---

# 👥 Contributors
- **임소현 (@soba1im)**: front-end
- **이재경 (@rud472888-creator)**: back-end
- **김하은**: producer
- **이세정**: producer