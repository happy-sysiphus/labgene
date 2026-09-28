# LabGene 제품 스펙 v0.1 — 연구실 지식 구축 스튜디오

작성일: 2026-09-28

상태: 하네스 스펙 v0.6의 검증이 성공한다는 전제 아래, LabGene을 시연 가능한 제품으로 만드는 설계다. 제품은 하네스 저장소 위의 얇은 층이며, 실험이 평가하는 바로 그 온톨로지·KG·RAG를 연구실 자료로부터 만든다. 하네스 스펙의 정책(연구원 약화 금지, 50행동, 누수 규칙)은 이 문서가 바꾸지 않는다.

## 1. 목적과 범위

**연구실이 자기 자료를 넣으면, 도메인 온톨로지에 맞춰 검토 가능한 KG가 만들어지고, 그 KG로 상담과 연구원 에이전트 실험이 돌아가는 것을 한 화면 흐름으로 시연한다.**

범위 안:

- 도메인 목록과 오픈소스 온톨로지 목록을 보여주고 선택하는 화면. 재료·공정·화학 도메인만 실제 동작.
- 선택한 온톨로지 번들의 로컬 확장(용어 추가, 별칭, 라벨 재정의·숨김). upstream 파일은 수정하지 않는다.
- 논문(PDF), 장비 매뉴얼(PDF), SOP(DOCX/PDF), 연구노트(MD/DOCX/HWPX), 실험 로그(XLSX/CSV), 발표 슬라이드(PPTX)의 일괄 투입.
- LLM 추출 → 코드 게이트 → 분담 정책에 따라 사람에게 한 번에 하나씩 질문하는 루프 → 승인된 사실만 하네스 저장소에 게시.
- 그래프 보기, 사람이 직접 하는 상담, 연구원 에이전트의 실험·상담 과정 픽셀 아트 리플레이.

범위 밖: 다중 사용자·인증·클라우드 배포, 다른 도메인의 실제 번들 구현, 온톨로지 전체 번역, upstream 온톨로지 편집, 자료 자동 수집(크롤링), BO 최적화기, 모바일.

## 2. 사용자 흐름

한 연구실 = 한 작업공간(`workspace/<lab_id>/`). 화면은 일곱 페이지다.

| 순서 | 페이지 | 하는 일 |
|---|---|---|
| 1 | 도메인 선택 | 도메인 목록 → 번들 구성(온톨로지 이름·라이선스·버전·규모·한 줄 설명) 확인 → 선택. 미구현 도메인은 "준비 중" 배지, 선택은 되지만 다음 단계로 못 감 |
| 2 | 자료 업로드 | 파일 여러 개 드롭 → 변환·청킹·색인·추출 진행률 → 문서별 상태(변환됨/추출됨/오류) |
| 3 | 질문 루프 | LLM이 고른 질문을 한 번에 하나씩. 원문 카드, 권장 답, 버튼, KG 영향 미리보기. 상단에 큐 유형별 완성도 |
| 4 | 검토 표 | 같은 항목을 표(st.data_editor)로 몰아서 처리. 보조 화면 |
| 5 | 그래프 | 승인·제안·충돌을 구분해 그림. 노드 선택 → 원문 카드·1-hop |
| 6 | 상담 | 사람이 질문을 치면 증거 카드가 붙은 답. 인용 노드를 그래프에서 하이라이트 |
| 7 | 연구원 리플레이 | 하네스 run의 원장을 픽셀 아트로 재생 |

시연 대본(약 7분): 도메인 선택 → 사전 인제스트된 자료 위에 SOP 1개 라이브 업로드 → 질문 5–6개 답변(약어 TMA, 단위 누락, 관계 방향, 로컬 용어 신설, 한글 별칭) → 그래프에서 새 노드·엣지 확인 → 상담 1회 → 픽셀 리플레이.

## 3. 구조

### 3.1 패키지와 앱

```
src/labgene/product/
  registry.py    도메인·번들 레지스트리, terms.jsonl 로드
  overlay.py     로컬 용어·별칭·오버라이드, Ontology YAML 내보내기
  convert.py     파일 → 하네스 구조 마크다운
  extract.py     닫힌 스키마 LLM 추출(2회 실행), draft 적재
  gates.py       원문 일치·단위·범위·사전 일치·OLS 조회·약어 다의성·2회 diff
  triage.py      분담 정책 + LLM 질문 작성·병합·정렬
  review.py      review.sqlite 스키마·상태 전이·트리거
  publish.py     verified → KnowledgeStore, 오버레이 YAML, 재색인, 거절 시 invalidate
  consult.py     사람 상담 래퍼(ProductAdvisor)
  lookup.py      OLS4·PubChem 클라이언트(로컬 캐시, 오프라인이면 캐시만)
  replay.py      ledger → 이벤트 JSON
  app/           Streamlit 페이지 7개 + 픽셀 리플레이 정적 HTML/JS
scripts/build_ontology_bundle.py   OBO/OWL/TTL/JSON Schema → terms.jsonl (오프라인 1회)
configs/domains.yaml               도메인·번들 레지스트리
```

CLI: `labgene-studio` (Streamlit 실행), `python -m labgene.product ingest|extract|triage|publish --workspace W` (같은 단계를 헤드리스로).

### 3.2 하네스 재사용과 변경

재사용: `Ontology`, `KnowledgeStore`, 구조 마크다운 파서·청커, BM25+dense+RRF 검색, `LLMKGExtractor` 패턴, `call_llm`과 공급자 어댑터, `ProductAdvisor`/`ConsultController`, 증거 카드, 원장 스키마.

변경은 두 가지다.

1. **제품 실행 모드.** 계약의 `ExecutionMode`를 바꾸지 않고 `live_development`로 실행한다. 누수 게이트는 통과형 검사기(`PassThroughChecker`: 항상 `allow`, `checker_id="product-passthrough"`, `policy_version="product-none"`, `development_only=False`)를 쓰고, 저장소가 요구하는 답안 번들은 `AnswerBundle(bundle_id="product-none", version="0", set_scope="<lab_id>", blocked_documents=[])` 하나를 넣는다. 제품 작업공간의 저장소는 평가용 상태 디렉터리와 섞지 않는다. (계약에 `"product"` 모드 값을 추가하는 것은 메인 구현자에게 별도 제안.)
2. **한글 경계.** `Ontology.mentions()`의 단어 경계를 `(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])`로 바꿔 라벨 뒤에 붙은 한글 조사("온도가")를 허용한다. 한글 복합어 안의 부분 일치는 알려진 한계로 두고, 별칭 테이블과 질문 루프로 교정한다.

모델 배치는 하네스 §10.1을 따른다. 사용자에게 보이는 상담·질문 문장은 Gemini, 추출·트리아지·게이트 보조 판정은 공급자 제한 없음, 임베딩은 `gemini-embedding-2`. 모든 호출은 `call_llm`으로 비용을 기록한다.

## 4. 도메인 레지스트리와 온톨로지 번들

`configs/domains.yaml`:

```yaml
domains:
  - id: materials-process-chem
    name: 재료·공정·화학
    implemented: true
    bundle: [pmdco, chmo, chebi-seed, rxno, mop, qudt-units, ald-schema-v4]
  - id: bio
    name: 생물·생명과학
    implemented: false
    bundle: [go, cl, uberon, obi, chebi-seed]
  # 이하 화학 합성·반도체·에너지·환경 등 4–6개, 모두 implemented: false
ontologies:
  pmdco: {name: PMDco, version: v3.1.1, license: CC BY 4.0, source: https://github.com/materialdigital/core-ontology/releases, format: owl}
  ...
```

표시용 메타(라이선스·유지관리자·등급·한 줄 설명)는 `science_ontologies_adoption.json`(72개)에서 가져오고, 거기 없는 것은 `science_ontologies.json`(379개) 항목으로 채운다. 미구현 도메인의 번들은 목록 표시만 하고 파일을 내려받지 않는다.

구현 번들(재료·공정·화학)은 온톨로지 선정 보고서의 결론과 겹침 방지 규칙을 그대로 따른다. 물질=ChEBI(없으면 PubChem CID·InChIKey + 가장 가까운 ChEBI 상위 클래스), 재료·시료·공정 구조=PMDco, 기법=CHMO, 반응=RXNO+MOP, 단위=QUDT, ALD 공정 변수=ALD/ALE 스키마 v4, 관계=RO, 상위=BFO/COB.

**로드 방식.** `scripts/build_ontology_bundle.py`가 CHMO·PMDco·RXNO·MOP·QUDT 단위·ALD 스키마를 `bundles/<name>/terms.jsonl`로 변환한다(필드: `id, label, synonyms, definition, parents, kind, deprecated, source, version`). 런타임은 이 JSONL만 읽는다. ChEBI는 통째로 싣지 않는다. 도메인 시드 목록(전구체·산화제·기판·박막 물질 수백 개)을 `chebi-seed/terms.jsonl`로 두고, 시드 밖은 `lookup.py`가 OLS4·PubChem을 조회해 캐시한다. 캐시는 `workspace/<lab_id>/lookup_cache.sqlite`에 남고, 오프라인 모드에서는 캐시만 쓴다. 폐기(deprecated) 용어는 링킹 후보에서 제외하고 대체 용어가 있으면 그것을 제안한다.

레지스트리는 번들 전체와 오버레이를 합쳐 하네스 `Ontology`가 읽는 프로파일 딕셔너리를 만든다. `profile_id`는 `<domain_id>@<bundle_hash>+overlay@<overlay_version>`이다.

## 5. 온톨로지 오버레이 — 추가·수정·별칭

upstream 파일은 읽기 전용이다. 연구실의 변경은 모두 오버레이에 쌓이고, 게시 때 YAML로 내보내 번들과 합쳐진다.

| 종류 | 필드 | 뜻 |
|---|---|---|
| 로컬 용어 | `local_id(lab:…)`, `label_en`, `label_ko`, `synonyms[]`, `parent_iri`, `kind`, `external_ids{pubchem_cid, inchikey, cas}`, `definition` | 온톨로지에 없는 물질·장비·공정 변수. 반드시 upstream 상위 IRI 하나에 매단다 |
| 별칭 | `surface_norm`, `surface_raw`, `lang(ko/en/formula/abbr)`, `term_id`, `verdict(positive/negative/unsure)` | 표면형 → 용어. `negative`는 "이 표면형은 이 용어가 아니다"를 기억해 재질문을 막는다 |
| 오버라이드 | `term_id`, `label_override`, `synonyms_add[]`, `hidden` | upstream 용어의 표시 라벨 재정의, 동의어 추가, 이 연구실에서 숨김 |

공통 필드: `status(draft/verified/rejected/held)`, `origin(llm/code/human)`, `model`, `prompt_sha`, `reviewer`, `reviewed_at`, `reason_code`. 파이프라인은 `draft`만 만들 수 있고, `verified`는 사람이 질문에 답하거나 검토 표에서 승인할 때만 된다.

**한국어 정책.** 온톨로지를 번역하지 않는다. 추출 시점에 LLM이 멘션의 영어 정규명·화학식·CAS 후보를 함께 내고, 링킹은 영어 라벨과 별칭 테이블로 한다. 한국어 표면형은 별칭으로만 축적되며, 사람이 질문에 답할 때 `positive`로 확정된다. UI에 보일 `label_ko`는 게시된 노드에 한해 LLM이 일괄 생성하고 표시 전용으로 둔다(사람이 별칭으로 승격하기 전에는 링킹에 쓰지 않는다). 정규화 규칙: NFKC, casefold, 공백·하이픈 제거, 트리/트라이·메탄/메테인 같은 국립국어원 복수 표준 표기를 같은 키로 접기.

**수정의 의미.** "온톨로지 수정"은 오버라이드다. upstream 용어의 정의·계층은 바꾸지 않고, 이 연구실에서 어떻게 부르고 보일지만 바꾼다. 번들을 새 버전으로 올릴 때 오버레이는 IRI 기준으로 다시 얹히고, 사라진 IRI는 `held`로 떨어져 질문이 된다.

## 6. 검토 DB 데이터 모델

`workspace/<lab_id>/review.sqlite`. 하네스 `knowledge.sqlite`와 분리하되, 원문 카드는 하네스 청크 ID(`doc:<stem>#<n>`)를 그대로 쓴다.

| 테이블 | 핵심 열 |
|---|---|
| document | `doc_id, filename, kind(paper/manual/sop/note/log/slides), sha256, converted_md_path, pages, status, error` |
| card | `card_id(=청크 id), doc_id, locator(json), text(사본), kind(paragraph/table/procedure/formula), uncertain` |
| mention | `mention_id, card_id, span, surface, normalized_en, formula, cas, entity_type, term_id, candidates(json: [{id,label,score,source}]), status, origin, …` |
| claim | `claim_id, card_id, quote, subject_mention, predicate, object_mention, value_min, value_max, unit_iri, conditions(json), claim_status(reported/observed/derived/hypothesis), status, origin, model, prompt_sha, run_agree(bool), gate_results(json), reviewer, reviewed_at, reason_code, published_relation_id` |
| question | `question_id, kind, target_table, target_id, priority, text_ko, recommended(json), options(json), impact_preview(json), status(open/answered/skipped), answer(json), answered_by, answered_at, follow_up_of` |
| audit_sample | `sample_id, target_table, target_id, blind_verdict, reviewer, at` |
| event_log | `at, page, action, target, seconds` (사람 작업 시간) |

엔티티 타입(초기): material, chemical, equipment, process_step, parameter, performance_metric, defect, unit. 술어(초기, 닫힌 목록): increases, decreases, saturates, optimum_window, no_effect, causes_defect, requires, uses. 목록 확장은 스키마 질문으로만 한다.

트리거·제약:

- `claim.status`, `mention.status`는 `CHECK IN ('draft','verified','rejected','held')`, 기본 `draft`.
- `verified`로 바뀌려면 `reviewer`가 있거나 `origin='code'`이고 `gate_results.auto_ok=true`여야 한다(트리거).
- `claim.quote`가 `card.text`에 그대로 없으면 INSERT/UPDATE를 `RAISE(ABORT, 'quote not in card')`로 막는다.
- `value_min <= value_max`, `unit_iri`는 QUDT 허용 목록 테이블 FK.
- 술어는 `predicate` 테이블 FK(닫힌 목록).

## 7. 구축 파이프라인

### 7.1 변환

| 입력 | 방법 | 출력 특성 |
|---|---|---|
| PDF, DOCX, PPTX, XLSX, CSV | docling | 제목 계층→`#`, 페이지→`<!-- page: N -->`, 표→파이프 표(헤더의 단위 괄호를 `[단위]` 행으로 이동), 수식→`$$`, OCR 의심→`ocr-suspect` |
| HWPX | zip 안 XML 직접 파싱 | 문단·표만. 그림 캡션은 텍스트로 |
| HWP(바이너리) | pyhwp가 설치된 경우만 | 없으면 "HWPX로 저장해 주세요" 안내 |
| MD, TXT | 그대로 | front matter 없으면 파일명·종류로 생성 |
| 슬라이드 | docling | 슬라이드 1장 = 페이지 1, 제목=절 |
| 엑셀 로그 | 시트마다 표 청크 | 첫 행 헤더, 단위 행 추론, 날짜·장비 호기 열은 `conditions`로 |

front matter에는 `title, kind, lab_id, material, equipment, date, source_file`을 넣는다. 변환 결과는 `workspace/<lab_id>/corpus/<doc_id>.md`에 저장되고 이후는 하네스 `ingest_document`(통과형 게이트)로 청킹·색인된다.

### 7.2 추출

청크마다 닫힌 스키마 JSON을 받는다.

```json
{"mentions":[{"surface":"트리메틸알루미늄","normalized_en":"trimethylaluminium","formula":"Al(CH3)3","cas":"75-24-1","entity_type":"chemical","span":[12,20]}],
 "values":[{"target_mention":0,"value":250,"value_max":null,"unit":"degC","quote":"기판 온도 250 °C"}],
 "claims":[{"subject":"substrate temperature","predicate":"increases","object":"growth per cycle","conditions":{"material":"Al2O3","range":{"degC":[150,250]}},"claim_status":"reported","quote":"…원문 구절…"}]}
```

같은 청크를 두 번 실행해 멘션·클레임 집합의 일치 여부를 `run_agree`로 기록한다. 프롬프트에는 번들 용어 라벨(청크와 관련된 타입만)과 연구실 별칭 테이블의 `positive` 항목을 예시로 넣는다. 호출은 `call_llm`(재시도·모델 변경 감지·비용 기록). 모든 결과는 `draft`다.

### 7.3 코드 게이트

| 게이트 | 통과 조건 | 실패 시 |
|---|---|---|
| 원문 일치 | `quote`가 카드 본문에 그대로 있음 | 클레임을 `held`, 질문 |
| 사전 일치 | 정규화된 표면형·정규명이 별칭 테이블 또는 번들 라벨·동의어와 정확히 하나에 일치 | 후보 조회로 |
| 후보 조회 | OLS4·PubChem·시드에서 top-5와 점수 | 후보 0개 → 로컬 용어 후보, 2개 이상 근접 → 질문 |
| 약어 다의성 | 대문자 5자 이하 표면형이 후보 2개 이상 | 항상 질문 |
| 단위 | QUDT 허용 목록에 있음 | 질문 |
| 범위 | `min<=max`, 파라미터별 물리 범위(번들의 ALD 스키마 값) 안 | 질문 |
| 2회 일치 | 두 실행이 같은 항목을 냄 | 불일치 항목은 질문 |
| 충돌 | 같은 주어·술어·목적어·조건에 다른 값이 다른 카드에 있음 | 충돌 질문 |

## 8. 분담 정책과 트리아지

**항상 사람이 결정하는 것(정책표).**

1. 파라미터→성능 관계 클레임 전부(술어가 increases/decreases/saturates/optimum_window/no_effect/causes_defect이고 목적어가 performance_metric 또는 defect).
2. 게이트에 걸린 수치·단위.
3. 동일성: 후보 2개 이상, 약어 다의성, 별칭 없는 한글 표면형, 장비 호기.
4. 두 번 이상 등장했거나 파라미터·재료 슬롯에 있는 미해결 멘션의 로컬 용어 신설.
5. 닫힌 술어 목록 밖의 새 관계 유형(스키마 질문), 출처 간 충돌.

**자동 승인(origin=code).** 사전에 정확히 하나로 일치한 멘션, 원문 일치·단위·범위·2회 일치를 모두 통과한 수치, 위 1번이 아닌 관계(uses/requires)로 게이트를 모두 통과한 클레임. 자동 승인 항목은 그래프에서 구분 표시되고 블라인드 감사 표본(시연 기본 30개)에 들어간다.

**LLM 트리아지.** 사람에게 갈 항목마다 (a) 한국어 질문 문장, (b) 권장 답과 한 줄 근거, (c) 선택지, (d) 답했을 때 생길 노드·엣지 미리보기를 만든다. 같은 표면형·같은 충돌은 질문 하나로 합치고 등장 횟수를 표시한다. 정렬은 정책표 순서 → 다른 질문의 전제가 되는 용어·별칭 질문 먼저 → 등장 횟수. 후속 질문은 답 하나당 최대 2개다(로컬 용어를 만들면 같은 표면형의 다른 멘션 일괄 적용 여부, 방향 오류로 거절하면 뒤집은 관계를 새 draft로 제안, 충돌을 한쪽으로 확정하면 다른 쪽 카드에 `superseded` 표시 여부).

질문 종류: identity, mapping, new_term, number_unit, relation, schema, conflict, audit.

## 9. 질문 루프 UI

한 번에 한 질문. 화면 구성:

- 왼쪽: 원문 카드(인용 구절 하이라이트, 문서·페이지·절), 표 청크는 표 그대로.
- 가운데: 질문 문장, 권장 답이 미리 선택된 버튼 `권장대로 / 아니오 / 수정 / 보류 / 건너뛰기`. `수정`은 항목 종류별 작은 폼(용어 선택 드롭다운, 값·단위, 술어·방향).
- 오른쪽: 1-hop 미니 그래프(≤50 노드)와 "이 답이 만들 변화" 미리보기.
- 상단: 큐 유형별 완성도(verified ÷ (verified+open)), 남은 질문 수, 세션 경과 시간.
- 카드 옆에 술어 정의(saturates와 increases의 차이)와 단위 규칙을 상시 표시. LLM의 자유 설명은 접힘 기본.

답은 즉시 반영된다. `권장대로`/`수정` → 대상 `verified`(검토자 기록), `아니오` → `rejected`(사유 코드 필수: 방향 오류/조건 누락/원문 불일치/다른 물질/기타), `보류` → `held`, `건너뛰기` → 질문만 뒤로. 후속 질문이 있으면 바로 이어 나온다. 검토 표 페이지는 같은 데이터를 표로 보여주고 다중 선택 승인·거절을 지원한다.

## 10. 게시·그래프·상담

**게시.** `verified` 클레임을 하네스 `KnowledgeStore.add_relation(subject=term_id, predicate, obj=term_id, parents=[card_id], created_by="human:<reviewer>" 또는 "auto:gates", conditions, claim_status)`로 쓴다. 로컬 용어와 별칭은 오버레이 YAML로 내보내고, 번들+오버레이를 합친 `Ontology`로 저장소를 다시 연다. 관계 ID는 내용 해시라 재게시는 멱등이다. 게시된 관계가 나중에 `rejected`되면 `store.invalidate(relation_id)`를 호출해 카드·경로까지 전이 무효화한다. 게시 단위는 "지금까지 승인된 전부"이며 버튼 한 번 또는 질문 N개마다 자동.

**그래프.** 저장소의 관계와 검토 DB 상태를 합친다. 선 스타일: 제안=점선, 승인=실선, 충돌=굵은 테두리. 색=술어. 자동 승인 노드·엣지에는 아이콘. 노드 선택 → 원문 카드 목록과 1-hop. 전체 그래프는 개요·고아 노드 확인용이고 편집은 질문 루프·검토 표에서만 한다.

**상담.** 사람이 친 질문을 `ConsultRequest`로 감싸 `ProductAdvisor`에 넣는다. 태스크는 `PublicTask(task_id="lab-free-question", parameters=[], metrics=[], success=[], simulator_id="none", simulator_version="none")`이고 관측은 빈 목록이다. 빈 파라미터·지표를 허용하도록 답변 검증 규칙을 조정한다. 답변의 `cited_source_ids`에 해당하는 노드·카드를 그래프에서 하이라이트한다.

## 11. 연구원 리플레이(픽셀 아트)

하네스 run의 `ledger.sqlite`에서 `actions`, `observations`, `consult_exchanges`를 읽어 시간순 이벤트 JSON을 만든다.

```json
{"t": 12, "kind": "consult", "question": "…", "answer": "…", "cited": 3}
{"t": 13, "kind": "experiment", "parameters": {"substrate_temp_degC": 250}, "result": {"gpc_nm": 0.11}, "best_so_far": 0.11, "target": 0.12, "success": false}
{"t": 14, "kind": "invalid_experiment", "reason": "out of range"}
```

정적 HTML/JS 페이지가 이 JSON을 받아 연구원 캐릭터가 상담 단말과 실험 장비 사이를 오가고, 말풍선에 질문·답·결과가 뜨며, 목표 대비 최선값 게이지가 오르는 장면을 그린다. 완료 run은 배속 슬라이더로 리플레이하고, 진행 중 run은 2초마다 폴링한다. 성공 시 축하 연출, 예산 소진 시 미도달 표시. 엔진·에셋·오픈소스 선택은 구현 계획에서 정한다. 리플레이는 원장의 실제 값만 그리며 새 값을 만들지 않는다. Gemini 라이브 run이 없을 때는 fixture run을 "예시 실행"으로 명시하고 재생한다.

## 12. 시연 코퍼스

각색한 가상 ALD 연구실 자료 약 12개(한·영 혼합). 실험 평가 코퍼스와 분리해 `data/demo_lab/`에 둔다.

| 종류 | 수 | 형식 | 심어 둔 빈칸 |
|---|---|---|---|
| 논문 | 2 | PDF(영) | 공정 창 관계, 표의 단위 |
| 장비 매뉴얼 | 1 | PDF(영) | 장비 클래스, 파라미터 물리 범위 |
| SOP | 2 | DOCX(한) | 약어 TMA, 절차 단계 |
| 연구노트 | 3 | MD·HWPX(한) | 트리/트라이메틸알루미늄 표기 변이, 장비 호기 ALD-02, 방향 모호한 관계 |
| 실험 로그 | 2 | XLSX | 단위 누락 열, 노트와 충돌하는 GPC 값 |
| 슬라이드 | 1 | PPTX(한·영) | 요약 수치, 새 성능 지표 후보 |

## 13. 구현 계획에서 정할 것

- 픽셀 리플레이 엔진·에셋·참고 오픈소스.
- docling 버전과 표 단위 행 추론 규칙, HWPX 파서 범위.
- Streamlit 그래프 컴포넌트(Cytoscape 계열) 선택.
- ChEBI 시드 목록의 초기 구성과 OLS4 조회 임계값.
- 술어·엔티티 타입의 초기 목록 확정과 ALD 스키마 v4에서 가져올 파라미터 물리 범위.
- 자동 게시 주기와 블라인드 감사 표본 크기의 기본값.

## 14. 참고

- 하네스 스펙 v0.6 §7(누수 규칙, 제품 모드에서는 통과), §10.1(모델 배치), §13(증거 카드·RAG).
- 온톨로지 선정: `C:\Users\지완\ontology-research\reports\재료 공정 화학 온톨로지 선정.md`
- KG 구축 분담·빌더 도구: `C:\Users\지완\claude\horcrux\reports\랩진 KG 구축 분담과 빌더 도구 선정.md` (분담 규칙·검토 큐·트리거·감사 규칙의 출처)
- 온톨로지 목록: `C:\Users\지완\ontology-research\science_ontologies.json`, `science_ontologies_adoption.json`
- 한국어 별칭 정책 근거(2026-09-28 웹 조사): XL-BEL(arXiv 2105.14398), BioELX(arXiv 2605.27380), 카카오헬스케어 SNOMED 매핑(PMC13010068), Sci-ZSEL(arXiv 2609.00228), Wikidata P683 한국어 라벨 2.1% 직접 조회.
