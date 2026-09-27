# LabGene 실험 하네스 스펙 v0.1

> 이 문서는 **계약서**다. 구현 에이전트는 이 문서가 정한 프로토콜·스키마·규칙을 바꾸지 않는다. 바꿔야 하면 버전을 올리고 "변경 이력"에 이유를 적는다. 런타임 에이전트(연구원·판정자·오케스트레이터)의 프롬프트는 §5의 규칙을 그대로 구현한다.

## 0. 한 줄 요약

장비 매뉴얼 + 오픈소스 온톨로지로 만든 KG가 **연구자(Gemma 에이전트)가 golden answer에 도달하는 루프 수를 줄이는지**, 그리고 제품의 오케스트레이터가 그 루프를 **더** 줄이는지를 판정자 오라클 루프로 측정한다.

## 1. 가설

- **H1 (지식)**: B(온톨로지+KG) < A0(지식 없음) — 루프 수 중앙값.
- **H2 (구조화)**: B(온톨로지+KG) < A1(같은 매뉴얼 텍스트 RAG) — 구조화·정규화 자체의 기여.
- **H3 (오케스트레이션)**: C(B + 오케스트레이터) < B.
- **H4 (등급)**: H1–H3 효과가 초·중·고급 모두에서 같은 방향. 등급 간 격차가 KG로 줄어드는지는 탐색 항목(가설 아님).

성공 판정: 각 가설은 (태스크, 반복, 등급)으로 **짝지은** 비교에서 루프 수 차이(하위 조건 − 상위 조건, 예: A0−B)의 중앙값에 대한 95% 부트스트랩 CI 하한이 0보다 크고, 짝지은 부호검정 p<0.05. 효과 크기 목표는 사전 등록하지 않는다(파일럿 후 v0.2에서 정함).

## 2. 용어

| 용어 | 정의 |
|---|---|
| 태스크 | 논문 1편에서 만든 문제 1개: 연구원이 보는 프롬프트 + 숨은 golden answer + 채점 루브릭 |
| golden answer | 논문이 확립한 정답(원인·최적 조건·결론 중 하나; `answer_type`으로 표시). 형태는 논문 선정 후 태스크별로 확정 |
| 루프 | 연구원이 `submit_answer`를 호출하고 판정자가 판정을 돌려주는 1회. **하네스가 센다** (에이전트 자기보고 아님) |
| 스텝 | 루프 안에서 연구원이 하는 도구 호출 1회(검색·조회·상담). 루프당 상한 K=8 |
| 조건 | A0 / A1 / B / C (§3.1) |
| 등급 | L1 초급 / L2 중급 / L3 고급 = 연구원 프로파일 `levels/<id>.yaml` 1개 (§3.2, 정의는 잠정) |
| 런 | (태스크, 조건, 등급, 반복번호) 1개의 실행. 새 컨텍스트에서 시작 |

## 3. 실험 설계

### 3.1 조건 (도구 집합만 다르고 프롬프트 템플릿은 동일)

| 조건 | 연구원이 쓸 수 있는 도구 | 증명하는 것 |
|---|---|---|
| A0 지식 없음 | `submit_answer` | 모델 사전지식 바닥선 |
| A1 텍스트 RAG | A0 + `search_text`, `get_passage` (같은 매뉴얼 코퍼스, BM25) | 지식은 있으나 구조 없음 |
| B 온톨로지+KG | A1 + `search_entities`, `get_entity`, `find_paths` | 구조화·정규화·관계의 기여 |
| C B+오케스트레이터 | B + `consult(question)` | 제품 오케스트레이터의 기여 |

- B ⊇ A1 ⊇ A0: 상위 조건은 하위 조건의 정보를 전부 포함한다(정보량 단조). 그래서 차이는 "추가된 것"의 기여로 읽힌다.
- C의 `consult`는 오케스트레이터(§5.3)에게 질문하고 근거 브리프를 받는다. 스텝 1회로 세고, 루프에는 안 센다. 오케스트레이터 내부 도구 호출은 로그만 남긴다.

### 3.2 등급 — **잠정 (별도 리서치 과제 R2로 근거 확인 후 확정)**

"초급/중급/고급 연구자"를 무엇으로 모사할지는 선례 조사가 먼저다. LLM 크기·추론 설정·페르소나 프롬프트·지식 제한 중 어느 것이 "전문성 수준" 모사로 쓰인 사례가 있는지, 그 타당성 근거는 무엇인지 R2가 답한 뒤 이 절을 v0.2에서 확정한다. 아래는 R2 결과가 없을 때의 **임시 가설**이며 하네스는 어느 정의든 담을 수 있게 만든다(등급 = 연구원 어댑터 설정 프로파일 1개).

| 등급(임시) | 후보 정의 | 실행 위치(모델 크기로 갈 경우) |
|---|---|---|
| L1 초급 | Gemma 3 4B | 개인 서버 Ollama (RTX 4060 Ti 8GB, Tailscale 100.86.154.106) |
| L2 중급 | Gemma 3 12B | 개인 서버 Ollama (부분 CPU 오프로드, 느림 허용) |
| L3 고급 | Gemma 3 27B | Google AI Studio Gemma API(OpenAI 호환) 또는 Colab. 로컬 불가(VRAM 8GB·RAM 16GB) |

- 하네스 요구사항(등급 정의와 무관): 등급은 `levels/<id>.yaml` 프로파일(`model, base_url, api_key_env, temperature, thinking, persona_prompt?, seed_policy`) 하나로 표현되고, 런 로그에 프로파일 해시가 남는다. 정의가 바뀌어도 코드는 안 바뀐다.
- 연구원 어댑터는 **OpenAI 호환 chat 엔드포인트** 하나로 추상화한다. 어디서 돌리든 하네스는 같다.
- 도구 호출은 모델 네이티브 function-calling에 의존하지 않고 **JSON 행동 프로토콜**(§5.1)로 한다 — 모델 크기·호스팅에 무관, 전부 로그 가능.
- Phase 1(§3.5)은 등급 1개(L2 후보)로만 돌리므로 R2 결과를 기다리지 않는다.

### 3.3 태스크

- 출처: 논문(별도 리서치 과제로 선정). 목표 ≥5편, ≥10 태스크. 매뉴얼 코퍼스에 **논문 본문이 들어가면 안 된다**.
- 태스크 파일 `tasks/<id>.yaml`:
  ```yaml
  id: t001
  paper: {doi: "...", title: "..."}
  domain: ald            # ald | sputter | solgel | ...
  answer_type: cause     # cause | condition | conclusion | other
  prompt: |              # 연구원이 보는 전부. 논문 결과·결론은 없음
    ...
  max_loops: 10
  knowledge_scope: [manual:ald-xx, ontology:chmo]   # 커버리지 점검용
  ```
- golden·루브릭은 **다른 파일** `tasks/golden/<id>.yaml`에 두고 판정자 프로세스만 읽는다:
  ```yaml
  id: t001
  golden: |
    ...
  rubric:
    - {key: mechanism, must: true, desc: "..."}
    - {key: parameter, must: true, desc: "..."}
    - {key: magnitude, must: false, desc: "..."}
  ```
- 판정: `must` 항목 전부 충족 = correct.

### 3.4 반복·종료·지표

- 반복 R=3 (시드 고정: `seed = sha256(f"{task}:{condition}:{level}:{rep}")` 하위 32비트 — Python `hash()`는 프로세스마다 달라지므로 금지; Ollama `seed`+`temperature 0.7`, 시드 미지원 엔드포인트는 문서화).
- 종료: correct, 또는 `max_loops` 소진, 또는 총 스텝 상한(=max_loops×K), 또는 프로토콜 위반 3회 연속.
- **주지표**: `loops_to_correct` (정수). 미도달은 검열(`censored=true`, 값 = max_loops+1로 표기하되 별도 열로 보고).
- 부지표: `success@max_loops`, 루프당 스텝 수, 도구 종류별 호출 수, 토큰(입력/출력), 벽시계, `consult` 횟수(C), 힌트 사용 수(힌트 켠 경우).
- 격자: 조건 4 × 등급 3 × 태스크 ≥10 × 반복 3 = ≥360 런. 로컬 Gemma는 비용 0, Google/Claude 호출은 판정자·오케스트레이터에만.
- 통계(stdlib): (태스크, 반복, 등급) 짝지은 차이 → 중앙값·부트스트랩 95% CI(태스크 단위 클러스터 부트스트랩)·부호검정. 검열 런은 (a) 검열값 포함, (b) 제외 두 방식 모두 보고.

### 3.5 실행 순서 (사용자 계획 순서)

1. **Phase 1**: L2만, A0/A1/B — H1·H2 파일럿 → 태스크·루브릭·판정자 보정.
2. **Phase 2**: L1·L2·L3 전부, A0/A1/B — H4. (R2로 등급 정의를 확정한 뒤에만 시작)
3. **Phase 3**: C 추가 — H3.

## 4. 판정자 오라클

- 모델: Claude(구독 CLI, `claude -p --json-schema`). 골든·루브릭을 보는 **유일한** 프로세스.
- 입력: 태스크 프롬프트, 루브릭, golden, 연구원 제출 `{answer, rationale}`, 이전 판정 이력(루브릭 키 단위).
- 출력(스키마 고정): `{"verdict": "correct"|"incorrect", "rubric": {key: pass|fail}, "hint": null}`.
- **힌트 정책** `hint_policy`: 기본 `none`(맞다/틀리다만). 옵션 `aspect`: 실패한 루브릭 키 이름만(값·방향 없음). 힌트를 켜면 별도 조건으로 표기하고 주결과에는 섞지 않는다.
- 판정자는 golden 텍스트·수치를 어떤 출력에도 넣지 않는다(코드가 출력에서 golden n-gram 검출 시 런 무효).
- **판정자 검증**: 본실행 전 사람이 채점한 제출 ≥20건과 일치율 ≥90%. 미달이면 루브릭을 고치고 재검증. 검증 결과는 `eval/judge_validation.md`.

## 5. 에이전트 프로토콜

### 5.1 연구원 (모든 조건 공통)

- 시스템 프롬프트 템플릿 하나(`prompts/researcher.md`). 조건별로 바뀌는 건 `{tools}` 블록뿐. 등급별로 바뀌는 건 프로파일(§3.2)뿐.
- 행동 = JSON 한 줄. 하네스가 파싱·실행·결과 주입(ReAct 루프를 하네스가 주도).
  ```json
  {"action": "search_text", "args": {"query": "...", "k": 5}}
  {"action": "get_passage", "args": {"id": "p:ald-xx:12:3"}}
  {"action": "search_entities", "args": {"query": "...", "kind": "cause"}}
  {"action": "get_entity", "args": {"id": "cause:precursor-degradation"}}
  {"action": "find_paths", "args": {"from": "sym:thickness-low", "to_kind": "action", "max_hops": 3}}
  {"action": "consult", "args": {"question": "..."}}
  {"action": "submit_answer", "args": {"answer": "...", "rationale": "..."}}
  ```
- 파싱 실패 → 오류 메시지 주입, 재시도. 3회 연속 실패 = 런 종료(`outcome=protocol_error`).
- 루프당 스텝 K=8 소진 시 하네스가 "이제 submit_answer만 가능"을 주입.
- 연구원은 파일·웹·메모리 없음. 컨텍스트는 런마다 새로 시작.

### 5.2 도구 의미론 (조건 B 기준; A1은 앞 둘만)

| 도구 | 반환 |
|---|---|
| `search_text(query, k≤10)` | 패시지 `{id, doc, page, snippet, score}` 목록 (BM25) |
| `get_passage(id)` | 패시지 전문 + 문서·페이지 |
| `search_entities(query, kind?, k≤10)` | 노드 `{id, kind, label, alt[], iri?, score}` — 사전 매칭 우선, 그다음 BM25 |
| `get_entity(id)` | 속성 + 이웃 엣지 `{rel, dir, node}` + 근거 패시지 id 목록 |
| `find_paths(from, to?|to_kind, max_hops≤3)` | 경로 목록, 각 엣지에 근거 패시지 id |
| `consult(question)` (C만) | 오케스트레이터 브리프 `{summary, evidence: [{claim, passage_ids, entity_ids}], suggested_next: [...]}` |

- 모든 그래프 응답은 **근거 패시지 id**를 동반한다(KG는 원문으로 가는 색인). 근거 없는 엣지는 존재하지 않는다.

### 5.3 오케스트레이터 (조건 C)

- 제품의 일부. 모델은 제품 결정(기본 Claude CLI). 온톨로지·KG·RAG 전체 도구 + 자기 추론.
- 입력: 연구원의 질문, 현재 태스크 프롬프트, 이 런의 대화 이력(연구원 행동·도구 결과·판정 verdict). **golden·루브릭·판정자 내부는 절대 못 본다.**
- 출력: 브리프(§5.2). 브리프의 모든 claim은 패시지 id를 달아야 하고, 코드가 id 실재를 검증한다(없으면 그 claim 제거).
- 내부 도구 호출은 스텝·루프에 안 센다. 로그(`calls.caller='orchestrator'`)만.
- 변형 C2(선택): 판정 후마다 오케스트레이터가 브리프를 먼저 밀어주는 능동 모드. Phase 3 여유 시.

## 6. 시스템(SUT): 온톨로지 · KG · RAG

### 6.1 온톨로지 (ontology-research 09-28 보고서 기준; 그 보고서는 반박 검증 전이므로 v0.2에서 재확인)

| 역할 | 채택 | 비고 |
|---|---|---|
| 기법·장비 유형 | CHMO (CC BY 4.0) | ALD `CHMO:0001311`, 스퍼터 `CHMO:0001364` 등. 장비 유형도 CHMO |
| 재료·공정 실행 구조 | PMDco v3.1.x (CC BY 4.0) | BFO 기반, ChEBI·QUDT 재사용. ALD 클래스 없음 |
| 화학 물질 | ChEBI (+ PubChem CID) | TMA·TDMAT·TEMAH·TiN·HfO₂는 ChEBI에 없음 → PubChem |
| 단위·물리량 | QUDT | 단위 기호→IRI 정적 표 |
| 출처·워크플로 | PROV-O 개념만 | `wasDerivedFrom` 등 주석 수준 |
| ALD 공정 변수 이름 | ALD/ALE JSON Schema v4 (Zenodo 21772766) | 파라미터 정규 이름 |
| 증상·원인·조치 | **자체 설계** | 맞는 표준 없음(IOF-Maint에도 증상 클래스 없음). IRI 주석만 |

- OWL 파일을 통째로 들이지 않는다. **용어 ID(IRI)만** `node.iri`에 적는다. LLM은 IRI를 생성하지 않는다.
- 매핑 절차(구축 시, 배치): 후보 = 사전 완전일치 → OLS4 `/api/search` top-k → Claude가 "같은 의미인가" 판정 → `MAPS_TO`. 실패는 `AUTO:` 네임스페이스에 남기고 `status=proposed`.

### 6.2 KG 스키마 v1 (고정. 늘리려면 스펙 버전 업)

- 노드 종류(10): `equipment, component, material, process, parameter, unit, symptom, cause, action, document/passage`
- 노드 id: `{kind}:{slug}` (예 `equip:ald-savannah-s200`, `cause:precursor-degradation`), 불변. label·alt만 편집.
- 관계(12): `HAS_COMPONENT, USES_MATERIAL, HAS_PARAMETER, HAS_UNIT, SPEC_RANGE(qual: min,max,unit,condition), EXHIBITS(equipment|process→symptom), CAUSED_BY(symptom→cause), RESOLVED_BY(cause→action), STEP_OF(action→process, qual: order), PART_OF(process→process), BROADER(node→node), MAPS_TO(node→iri)`
- 모든 엣지에 `passage_id`(근거) 필수. `MAPS_TO`·`BROADER`만 예외(출처=온톨로지/사전).

### 6.3 구축 파이프라인 (Claude CLI, 배치)

1. **수집**: `data/manuals/<doc_id>.pdf` + `manifest.yaml`(장비, 출처, 라이선스). 논문은 넣지 않는다.
2. **텍스트화**: `pdftotext -layout`(poppler, 설치됨) → 페이지별 텍스트. 글자 수가 적은 페이지(스캔) → `pdftoppm` PNG → `claude -p`가 Read로 읽어 전사. 결과 `data/text/<doc_id>/<page>.txt`.
3. **패시지 분할**: 페이지 → 문단/절 단위(≤1,200자) → `passage(id, doc_id, page, text)`. FTS5(unicode61+porter; 한국어 매뉴얼이 생기면 bigram 채널 추가).
4. **추출**: 패시지 묶음 → `claude -p --json-schema` → `{entities:[{kind,name,alt[]}], relations:[{src_name,rel,dst_name,qual,passage_id}]}`. 스키마 밖 kind/rel은 코드가 버린다(자유 어휘 폭발 차단).
5. **정규화(grounding, 코드)**: NFKC → casefold → 공백/하이픈 제거 → 같은 kind 안에서 label∪alt 완전 일치 → 단위·수치 정규식 → 실패 시 `AUTO:` proposed 노드. 사전 `data/ontology/vocab.json`(시드 ≈100항목, 구축하며 증가).
6. **온톨로지 매핑**: §6.1 절차 → `MAPS_TO`.
7. **적재**: SQLite `data/kg.db` (WAL). 재구축은 `raw` 추출 JSON에서 결정론적으로(LLM 재호출 없음).
8. **품질 게이트**: 추출 표본 30 패시지 사람 검토(정밀도 ≥0.8), AUTO 비율 보고, 루브릭 커버리지(각 태스크 `knowledge_scope`가 KG에 존재).

### 6.4 저장

- SQLite 한 파일. 테이블: `document, passage, passage_fts, node, edge, extraction_raw`. 그래프 질의는 재귀 CTE(≤3홉). Postgres 호환 DDL 부분집합만 사용(나중 Supabase 이전 대비).
- 임베딩 없음(API 키 없음, 근거 부족). BM25 recall이 병목으로 측정되면 로컬 임베딩을 v0.2에서 검토.

## 7. 누수·오염 방지 규칙 (전 조건 공통, 위반 = 런 무효)

1. 프롬프트 템플릿 동일, 도구 블록만 차이. 템플릿·도구 설명은 해시로 `runs`에 기록; 바뀌면 영향 조건 전부 재실행.
2. 연구원·오케스트레이터 프로세스는 `tasks/golden/`을 읽을 수 없다(경로 분리 + 판정자만 로드).
3. 판정자 출력에 golden n-gram(≥6자) 검출 시 런 무효.
4. 매뉴얼 코퍼스에 논문·golden 문구 없음: `leak_check.py`가 golden 핵심구를 코퍼스에서 검색해 직접 일치 0건 단언. 결과 `eval/leak_check.md`.
5. 런마다 새 컨텍스트. Claude CLI 호출은 전용 `CLAUDE_CONFIG_DIR`(플러그인·훅·자동메모리 0) + `--strict-mcp-config` + 카나리 검사(이전 런 문자열이 새 런에 없음).
6. 루프·스텝 카운터는 하네스에만 있다.
7. 모델 버전·시드·온도·프롬프트 해시·KG 해시를 런마다 기록.
8. 본실행 전 이 스펙(가설·분석)을 커밋 = 사전 등록. 파일럿 후 태스크·루브릭 변경은 "변경 이력"에 기록하고 본실행은 변경 후 데이터만 쓴다.
9. 판정자 검증(§4) 통과 전 본실행 금지.

## 8. 로깅·재현성

- `eval/runs.db`: `runs(run_id, task, condition, level, rep, seed, model, endpoint, prompt_hash, kg_hash, started, ended, loops, success, censored, steps, tokens_in, tokens_out, outcome)`, `events(run_id, seq, ts, actor, type, payload_json)` — 연구원 행동·도구 결과·판정·오케스트레이터 내부 호출 전부.
- `python -m labgene report` → 마크다운 표(조건×등급 중앙값·CI·성공률) + 짝지은 검정.
- 전사(transcript)는 `eval/transcripts/<run_id>.jsonl`.

## 9. 저장소 구조·모듈 (구현 계획의 입력)

```
labgene/
  docs/superpowers/specs/2026-09-28-labgene-harness-design.md   ← 이 문서
  labgene/
    kg/        store.py  vocab.py  ground.py  ingest.py(pdf→passage)  extract.py  map_ontology.py  tools.py
    harness/   protocol.py(행동 JSON)  researcher.py  judge.py  orchestrator.py  runner.py  stats.py  leak_check.py
    llm/       openai_compat.py(연구원)  claude_cli.py(판정자·오케스트레이터·구축; horcrux llm.py의 stdin·utf-8·taskkill 패턴 재사용)
    prompts/   researcher.md  judge.md  orchestrator.md  extract.md
    cli.py     (ingest | extract | build | dryrun | run | report | leak-check | judge-validate)
  tasks/       *.yaml     tasks/golden/*.yaml
  levels/      L1.yaml  L2.yaml  L3.yaml   (§3.2 연구원 프로파일)
  data/        manuals/  text/  ontology/vocab.json  kg.db
  eval/        runs.db  transcripts/  reports/
  tests/
```

- 런타임 의존성 목표: stdlib + `pyyaml` + `httpx`(OpenAI 호환 호출). MCP 서버는 ①에서 만들지 않는다(하네스가 도구를 직접 호출). 사람/외부 에이전트용 MCP 노출은 후속.
- 모든 파일 I/O `encoding="utf-8"`.

## 10. 단계 계획 (4주) — 각 단계 완료 기준 포함

| 주 | 산출물 | 완료 기준 |
|---|---|---|
| W1 (9/28–10/4) | 저장소·스펙 커밋 / `store`·`vocab`·`ground` + 테스트 / `protocol`·`runner`·`stats` / **LLM 0회 dry-run**(스크립트 연구원 + 스크립트 판정자, 더미 태스크) / 서버 Ollama + Gemma 4B 접속 / `claude_cli` 래퍼 / 매뉴얼 1편 ingest→FTS | dry-run이 루프·스텝·검열을 정확히 세고 `report`가 표를 낸다. `search_text`가 매뉴얼 1편에서 동작 |
| W2 (10/5–10/11) | 매뉴얼 1–2편 추출→grounding→KG / 도구 6종 / 판정자 + 루브릭 / 파일럿 태스크 2개(손으로) / L1·L2로 A0/A1/B 파일럿 / 판정자 검증 시작 / (병렬) 논문 리서치 → 태스크 초안 | 파일럿 격자 실행 완료, 판정자 일치율 측정, 추출 정밀도 ≥0.8 |
| W3 (10/12–10/18) | 태스크 ≥10 확정(R1) + golden + leak-check / Phase 1 본실행(A0/A1/B × L2 × 3) / R2 확정 시 등급 프로파일 추가 → Phase 2(× L1·L3) / 분석 v1 | 사전 등록 커밋 후 실행, H1·H2 결과 표(H4는 R2 확정 시) |
| W4 (10/19–10/25) | 오케스트레이터 + `consult` / Phase 3(C × 등급 × 3) / 분석 v2 | H3 결과, 실패 유형 표본 20건 분류 |
| 마감 주 (10/26–10/30) | 최종 보고서·재현 스크립트 / 버퍼 (본실행 재실행 1회분) | **10/30 완료.** 재현 스크립트 1개로 전체 재실행 가능 |

## 11. 범위 밖 (이번 스펙)

사람용 UI · 다중 연구실 공유 · 논문/SOP/.eln의 **지식** 수집(논문은 태스크 원천으로만) · 임베딩 · 파인튜닝 · MCP 서버 노출 · 배포.

## 12. 별도 리서치 과제 (이 스펙 밖에서 수행, 결과가 v0.2에 반영)

| id | 질문 | 스펙에 미치는 영향 | 기한 |
|---|---|---|---|
| R1 | 우리 실험에 쓸 논문 선정: golden answer가 명확하고, 매뉴얼 지식이 답에 실제로 기여하며, 코퍼스에 누수되지 않는 논문 ≥5편 → 태스크·golden·루브릭 초안 | §3.3 태스크 | W2–W3 |
| R2 | "초급/중급/고급 연구자"를 LLM 에이전트로 모사한 선례가 있는가? (모델 크기·추론 설정·페르소나·지식 제한 중 무엇을 썼고 타당성 근거는?) 없으면 가장 방어 가능한 정의는? | §3.2 등급 | Phase 2 전 |
| R3 | 온톨로지 선정 보고서(09-28, 반박 검증 전) 재검증 | §6.1 | W2 |

## 13. 미결 (owner)

| 항목 | owner | 기한 |
|---|---|---|
| 장비 매뉴얼 PDF 확보(어느 장비, ≥2편) | 사용자 | W1 |
| 서버 WSL2에 Ollama + Gemma 설치 | 사용자(구현 에이전트 보조) | W1 |
| L3용 Google AI Studio 키/엔드포인트 (등급=모델 크기로 갈 경우) | 사용자 | Phase 2 전 |
| 힌트 정책 기본값(none 유지 여부) | 사용자 | W2 파일럿 후 |

## 14. 검증 방법 (end-to-end)

1. `pytest`: store CRUD·재귀 CTE 경로, grounding 규칙(호기·약어·단위), 프로토콜 파서, stats(부트스트랩·부호검정 고정 시드 값).
2. `python -m labgene dryrun`: 스크립트 연구원(oracle/naive/random) × 스크립트 판정자 → runs.db에 정확한 loops/steps/censored, report 표 생성. oracle=1루프, random 중앙값 > naive > oracle 단언.
3. `python -m labgene run --task t000 --conditions A0,A1,B --level L1 --reps 1`: Gemma 4B 스모크 3런, 전사에 golden 문자열 0건, 프로토콜 오류 0건.
4. `python -m labgene judge-validate`: 사람 라벨 20건 일치율 출력 ≥0.9.
5. `python -m labgene leak-check`: golden 핵심구 코퍼스 직접 일치 0건.
6. 격자 실행 후 `report`가 조건×등급 표·CI·검정을 낸다; 같은 시드로 재실행 시 runs.db의 loops가 동일(Ollama 시드 결정론 범위 내).

## 변경 이력

- v0.1 (2026-09-28): 최초. 설계 패널(4안·심사 3·반박 3, 09-27) 결과와 사용자 결정 반영.
