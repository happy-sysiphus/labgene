# LabGene 하네스 작업 정리 — 2026-09-28

기준 스펙: `docs/superpowers/specs/2026-09-28-labgene-harness-design.md` v0.6 (sha256 `47c497b3…`, 변경 없음).
구현 계획: `docs/superpowers/plans/2026-09-28-labgene-harness-implementation-plan.md`.
세부 상태·증거: `TASKS.md` · 검증 기록 `docs/implementation/verification.md` · 설계 결정 `docs/implementation/decisions.md` · 실행법 `README.md`.

## 한눈에 보기

| 단계 | 상태 |
|---|---|
| T00–T07 구현 (하네스·과학 환경·기억·지식·공급자·연구원·상담·CLI·보고) | 완료, 오프라인 검증 통과 |
| T10 독립 검토 | 1차(핵심 모듈 5개)·2차(상담·보고·평가 모듈)·통합(연결·CLI) 검토 반영 완료 |
| T06-live Gemini 실연결 | 완료 (3/3 통과). 런타임은 U10으로 대체 |
| T06-sub 구독 CLI 전환 (연구원·상담 Codex Luna max, KG·요약 Claude Opus max) | 오프라인 완료·독립 검토 반영(353 passed), **실연결은 사용자 허가 대기** |
| T02-ext 실제 과학 환경 | aldenv·Summit 설치·재현 완료, **Summit 성공 규칙 결정 대기** |
| T08 자격검증·게이트·검색 검증·파일럿 | 도구 완성, **모델·예산·초안 검토 결정 대기** |
| T09 본평가 | 대기 (T08 통과 후) |

커밋: 브랜치 `harness-v0.6-offline`, `cc1e342` (167개 파일). 기존 스펙 수정과 `docs/superpowers/plans/`는 제외했다.
커밋 이후 변경(아직 커밋 안 함): `.env` 로딩(`config.py`, `cli.py`), live 테스트 사용량 기록, 문서 갱신, 이 문서.

오프라인 결과는 계약 검증일 뿐이며 연구 성능이나 제품 효용의 근거가 아니다.

## 1. 구현한 것

- **하네스(T01):** 50행동 원장. 카운터는 커밋된 기록에서만 계산하며 51번째 행동은 불가능하다. 같은 action_id 재전송은 재차감하지 않고, 강제 중단 후 재개와 연속 3회 프로토콜 오류 종료를 지원한다.
- **과학 환경(T02):** 격리 worker adapter(aldenv, Summit)와 과제 검증 보고서(공개/비공개 분리)를 구현했다.
- **기억(T03):** 종료 시 모든 실험 기록을 자기 조건 기억에 자동 저장한다. 한 트랜잭션으로 중복 없이 저장하고, 상태 디렉터리 전체를 스냅숏·복원한다.
- **지식(T04/T05):** 수집한 자료는 격리 영역 → 신원 검사 → 누수 게이트를 거쳐 승인 저장소에 들어간다. 차단 출처의 파생물은 전이적으로 무효화한다. 제품 RAG는 BM25+dense+RRF, 조건부 KG 경로, 증거 카드로 구성하고 BO는 넣지 않았다.
- **공급자·연구원·상담(T06):** Gemini Interactions와 내부 역할 adapter는 유한 재시도만 하고, 다른 모델로 대체하지 않으며, 반환 모델이 바뀌면 멈춘다. 연구원은 계획→검토→확정 구조이고, 두 상담은 같은 기본 모델과 같은 게이트 검색 도구를 쓴다.
- **평가 도구(T08용):** 연구원 자격검증(고정 상태 24 + 폐루프 3종), 누수 게이트 검증, 검색 개발 검증.
- **CLI·보고(T07):** `preflight`, `build-corpus`, `run-set`, `run-evaluation`, `resume`, `report`, `validate-task`, `validate-gate`, `validate-retrieval`, `qualify-researcher`, `freeze`. 비용 상한은 세션과 사전 구축을 넘어 누적 적용된다. `freeze`는 코퍼스·답안 번들·분석 계획 내용까지 고정한다.

## 2. 검증 결과

- 오프라인: `pytest tests/unit tests/integration tests/e2e` **284 passed** (구독 CLI 전환과 검토 반영 뒤 **353 passed**). 강제 중단 6개 경계 모두 재개 결과가 중단 없는 실행과 같다.
- 실제 과학 환경: `pytest tests/science` **6 passed**. aldenv 수치 재현, Summit 결정성 확인.
- Gemini 실연결: `tests/live` **3 passed**. 8번 호출 모두 반환 모델 일치, 사용량 기록 확인.
- 독립 검토에서 찾아 고친 결함(모두 회귀 테스트 추가):
  - 1차 (하네스·과학 환경·기억·지식·공급자/연구원): 주요 9 + 경미 24
  - 2차 (상담·보고·평가 도구): 차단 1 + 주요 6 + 경미 21
  - 통합 (연결·CLI): 주요 3 + 경미 8
  - 대표 사례: 재개 시 비용 상한 우회, freeze가 지식 입력을 고정하지 않음, 연구원 입력에 조건 이름 노출, 승인 판정이 답안 번들 변경 뒤 재사용됨.
- 자격검증 도구는 스크립트 fixture 연구원을 **불합격**으로 판정했다(구조화 행동 16/24). 기준이 느슨하지 않다는 확인이다.

## 3. 오늘 확인한 사실 (결정 근거)

### 비용 실측 (에피소드 중반, 관측 28개 상태)

| 구성 | 판단 1회 | 비고 |
|---|---|---|
| Gemini 3.1 Pro, thinking high (현재 스펙) | $0.105~0.114 | 비용 대부분이 thinking 토큰 |
| Gemini 3.1 Pro, thinking low | $0.082 | |
| Gemini 3.8 Flash, high, 분석 도구 상한 1 | **$0.054** | 상한이 없으면 도구 10회 호출로 $0.10 |
| Gemini 3.8 Flash, medium, 상한 1 | $0.043 | |
| 상담 1회 (Pro) | 일반 $0.095 / 제품 $0.069 | |
| 누수 게이트 호출 1회 | Pro $0.0032 / 3.1 Flash-Lite $0.0004 | |

- 에피소드 1회는 Pro 기준 약 $2~7, 3.8 Flash 기준 약 $1~3.5다. 빨리 성공하면 싸고, 50회를 다 쓰면 비싸다.
- 예산 **5만원**: 환율 1,400원·부가세 10%·카드 수수료 2%를 보수적으로 적용해 **정가 $30 한도**로 관리한다.
- 지금까지 지출은 **$0.773(약 1,200원)**: smoke $0.106, 비용 측정 $0.456 + $0.211.
- 5만원으로 가능한 것:
  - 3.8 Flash 기준: T08 전체 약 $13~41. 대부분 들어가지만, 폐루프 에피소드가 대부분 50회를 다 쓰면 넘을 수 있다. 그 경우 가드가 한도에서 멈춘다.
  - Pro 기준: T08만으로 약 $26~81이라 초과할 가능성이 크다.
  - 본평가: Flash 기준 세트 반복 3회(24 에피소드) 약 $24~84, 5회(40 에피소드) 약 $40~140이다. **추가 예산 약 4만~22만원이 필요**하다.

### 모델 비교 (Artificial Analysis Intelligence Index, 2026-09-28 확인)

| 모델 | 지표 | 단가 (입력/출력, 100만 토큰당) |
|---|---|---|
| Gemini 3.1 Pro Preview (현재 스펙) | 30 | $2 / $12 |
| **Gemini 3.8 Flash (high)** | **41** | $0.75 / $3.75 (2026-12-31까지, 이후 2배) |
| Claude Sonnet 5 | 38 | $2 / $10 (토크나이저가 약 30% 더 셈) |
| Claude Opus 5.5 | 54–58 | $4 / $20 |
| GPT-6 Astra | 53 | $10 / $50 |

Claude·GPT 중 성능이 비슷한 모델은 값이 같거나 더 비싸서 예산 문제를 풀지 못한다. 구글 안에서는 3.8 Flash가 더 높고 더 싸다. 일반 벤치마크 점수이므로 실제 연구원 역할은 T08 자격검증으로 확인해야 한다.

### Summit 과제 (54,264개 격자점 분석)

- 논문 목표는 "수율이 최대의 90% 이상이면서 TON 최대화"이고, 표 1 case I 최적점은 수율 82%, TON 69다.
- 모델에 논문 최적 조건을 넣으면 수율 78.67%, TON 68.40이 나온다.
- **문제:** 모델이 TON을 수율과 따로 예측해 "TON = 수율/촉매량" 관계가 깨진다(실측 데이터에서는 평균 오차 0.07로 성립). 현재 후보 규칙(수율 ≥ 77, TON ≥ 66)은 모델 오류 지점 5곳에서만 성공한다. TON을 정의대로 계산하면 성공점이 0개다.
- 선택지별 성공 영역(격자점 수 / 무작위 50회로 성공할 확률):
  - A (TON = 수율/촉매량, 기준 = 논문 최적 조건의 모델 값, 허용 오차 3): 5점 / 0.46%
  - A' (A와 같고 허용 오차 5): 14점 / 1.3%
  - B (논문 목적식을 모델 안에서 적용): 사실상 1점
  - C (현재 규칙 유지): 모델 오류로만 성공하므로 쓸 수 없다.

## 3-1. 누수 게이트 결정과 확인 결과 (추가)

- **누수 게이트는 Codex(ChatGPT 구독)로 결정**(U8), **추론 강도는 medium 이하**(U9). 스펙상 내부 역할은 공급자가 자유라 스펙 변경은 없다.
- Codex CLI를 0.158.0으로 업그레이드했다(0.144.6은 gpt-6-astra를 못 부름). 설정은 격리된 `codex exec`다: 빈 작업 폴더, 읽기 전용, 사용자 설정 무시, 세션 기록 안 남김, 도구 기능 끔, 게이트 지시문으로 기본 지시문 교체.
- 게이트 호출 1회: 입력 약 7,400토큰(도구를 끄고 지시문을 바꾸기 전 17,949), 출력 약 36토큰, 평균 5.5초. 실제 사용 모델은 CLI가 알려 주지 않아 기록에 '검증 불가'로 남긴다.
- 개발용 가짜 케이스 26건: 잘못 허용 0, 잘못 차단 0, 보류 0, 오류 0. 파이프라인 동작 확인일 뿐 실제 게이트 정확도는 아니다. 실제 과제의 정답 논문 기준 케이스로 다시 검증해야 한다.

## 3-2. 모델 역할을 구독으로 전환 (U10–U12, 추가)

- **결정(사용자):**
  - 연구원·두 상담: Codex CLI `gpt-6-luna`, 추론 강도 max, 세 역할이 같은 설정(U10). U9(medium 이하)의 예외다.
  - KG 추출·기억 요약: Claude Code headless `claude-opus-5-5`, max(U11).
  - 상담의 외부 웹 검색: 당분간 없음(U12). Codex 내장 웹 검색도 모든 호출에서 끈다.
  - 임베딩은 Gemini API 그대로 둔다(구독에 임베딩이 없음).
- **구독 과금만 쓰게 막은 부분:** 두 CLI를 부를 때 API 키 환경변수를 지운다. preflight는 Codex가 ChatGPT 로그인·0.158.0인지, Claude Code가 구독(`claude.ai`) 로그인인지 확인한다.
- **Codex에서 도구 쓰기:** `codex exec`에는 함수 호출 기능이 없다. 그래서 도구 설명을 지시문에 넣고, 출력 스키마로 `{"text", "tool_calls"}` 형식을 강제했다.
  - 대화 이력은 매 호출마다 JSON 한 덩어리로 다시 보낸다. 도구 결과 문자열이 가짜 대화 차례를 만들 수 없다.
  - 연구원·상담 코드는 바꾸지 않았다(원래 상태 없는 공급자를 지원).
  - 선택 필드는 null 허용으로 바꿨다가 null을 지워서 원래 모양으로 돌려준다.
- **격리:** Codex가 자기 도구를 쓰면(명령 실행, 파일 변경, 웹 검색, MCP, 계획 도구 등) 결과를 버리고 인프라 오류로 처리한다.
  - 무한 재연결 재시도와 fast mode는 끈다.
  - Claude Code는 `--safe-mode --tools "" --restricted`로 CLAUDE.md·스킬·플러그인·훅·MCP·도구를 모두 끄고, 호출한 세션의 환경변수도 넘기지 않는다.
- **모델 확인:**
  - Claude Code는 실제 실행 모델을 결과(`modelUsage`)로 확인한다.
  - Codex는 알려 주지 않는다. 보고서에 `unverified`로 표시한다(전에는 `ok`로 잘못 나왔다).
- **스펙과의 충돌:** 스펙 §10.1(Gemini Pro)·§11.5와 다르다. 스펙 파일에 사용자 수정이 있어 고치지 않았고, `decisions.md`에 충돌을 기록했다. 자격검증(T08)은 U10 설정으로 다시 해야 한다.
- **독립 검토 반영:**
  - Codex가 요청을 다른 모델로 돌린 흔적이 출력에 있으면 모델 변경으로 보고 실행을 멈춘다.
  - 빈 응답·형식이 깨진 응답은 어댑터 문제로 보고 재시도한다. 연구원의 프로토콜 위반으로 세지 않는다.
  - Codex가 몰래 붙이는 문맥(스킬 목록, 폴더·셸·날짜)을 껐다. 모델 호출 없이 `codex debug prompt-input`으로 확인했고, 7,854자에서 2,705자로 줄었다. 남은 것은 모델 목록이 강제로 넣는 멀티에이전트 안내문이고, 그 도구는 꺼져 있다.
  - Claude는 출력이 가장 많은 모델로 실제 모델을 판정한다.
  - 구독 모델도 가격 0을 명시해야 preflight를 통과한다.
- **아직 실제 호출은 한 번도 안 했다.** 오프라인 테스트 353개 통과. preflight는 두 live 프로파일 모두 통과했다(로컬 상태 명령만 실행).

## 4. 결정 대기 (사용자)

1. ~~연구원·두 상담 모델~~ → U10으로 결정됨(Codex `gpt-6-luna` max).
1-1. **실연결 허가**: canary 격리 확인 → live smoke(판단 1회, 조건별 상담 1회, KG 추출 1회). 구독 한도만 쓰고, API 비용은 임베딩뿐이다.
2. **Summit 성공 규칙**: A안(허용 오차 3) 추천. 모델에 노출되는 과제 ID도 중립 이름으로 바꿔야 한다. 동결 전에 파일럿으로 난이도를 확인한다.
3. **초안 독립 검토** (모두 `draft_unreviewed` 또는 개발용):
   - `configs/qualification/criteria.yaml`: 통과 기준
   - `configs/qualification/fixed_state/fs-*.yaml`: 고정 상태 과제 24개(`uc` 단위·제약, `og` 관측 근거, `hu` 가설 갱신, `na` 다음 행동)
   - `configs/qualification/fixed_state/tasks/`: 위 과제들이 쓰는 공개 과제 3개
   - `configs/qualification/closed_loop/cl_*.yaml`: 폐루프 3종
   - `configs/gate_cases/dev_fixture.yaml`, `configs/retrieval_dev/fixture_dev.yaml`: 가짜 fixture 기준이라 실제 과제가 정해지면 새로 작성
4. **관측 근거 채점 방식**: 지금은 연구원이 실제로 주장한 내용만 채점한다(연구 노트가 선택 사항이라서). 유지할지 결정.
5. **본평가 예산·규모**: 파일럿에서 에피소드당 실측 비용을 본 뒤 결정.
6. **보안**: 채팅에 붙여 넣은 API 키는 작업이 끝나면 재발급 권장.

## 5. 결정 후 다음 작업

1. (허가 후) canary 격리 확인과 live smoke. 실측 지연시간으로 `provider_timeout_s`를 정하고, T08용 상한을 산정한다.
2. ~~결정된 모델로 live 프로파일 갱신~~ → 완료(`configs/live.yaml`, git 제외). 이전 Gemini 설정은 `configs/live-gemini-smoke.yaml`.
3. Summit 과제를 재정의·재검증한다(`validate-task`).
4. T08을 진행한다: 고정 상태 자격검증 → 누수 게이트 검증 → 폐루프 자격검증 → Summit 파일럿 1~2회.
5. 실측 비용으로 본평가 규모를 산정하고 `freeze` → `run-evaluation`으로 간다.

## 6. 주요 파일 위치

| 무엇 | 어디 |
|---|---|
| API 키 | `.env` (git 제외) |
| live 프로파일 (구독 CLI, smoke 한도; 실행은 허가 후) | `configs/live.yaml` (git 제외) |
| 이전 Gemini smoke 프로파일 ($2 한도) | `configs/live-gemini-smoke.yaml` (git 제외) |
| 구독 CLI adapter | `src/labgene/providers/codex_cli.py`, `src/labgene/providers/claude_cli.py` |
| 실측 증거 | `artifacts/live-smoke/`, `artifacts/cost-probe/` (git 제외) |
| Summit 비공개 자료·분석 스크립트 | `private/` (git 제외, 평가자 전용) |
| 실제 과학 환경 | `.envs/aldenv`, `.envs/summit` (git 제외) |
| 과제 검증 보고서 (공개분) | `docs/implementation/task-validation/` |

## 7. 재실행 명령

```powershell
.venv\Scripts\python -m pytest tests/unit tests/integration tests/e2e -q
.venv\Scripts\python -m pytest tests/science -q
.venv\Scripts\python -m labgene run-set --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml --run-id <새 id>
.venv\Scripts\python -m labgene preflight --profile configs/live.yaml --plan configs/set_plans/smoke.yaml   # 모델 호출 없음
$env:LABGENE_LIVE="1"; $env:LABGENE_LIVE_PROFILE="configs/live-gemini-smoke.yaml"; .venv\Scripts\python -m pytest tests/live -q   # 유료, 약 $0.11
```
