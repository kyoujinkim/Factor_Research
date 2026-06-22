# News-Driven LLM Commentary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit all LLM commentary in `market_research.ipynb` to fetch live news via OpenAI's `web_search_preview` tool, rewrite prompts to be news-narrative-first (minimal statistics), and add a new top-down industry-to-company section with a global cross-market synthesis.

**Architecture:** All changes are isolated to one file (`output/0.export/market_research.ipynb`). Cell 7 (LLM helper) gains a `use_search` flag and date injection. Four existing commentary cells (14, 18, 24, 32) have their data payloads trimmed to directional signals and their prompts rewritten. Four new cells are inserted after cell 32.

**Tech Stack:** Python, OpenAI Responses API (`client.responses.create` with `tools=[{"type": "web_search_preview"}]`), Jupyter / `NotebookEdit` tool.

---

## File Map

| File | Action |
|------|--------|
| `output/0.export/market_research.ipynb` cell 7 | Rewrite — add `use_search`, date injection, new system prompt |
| `output/0.export/market_research.ipynb` cell 14 | Rewrite — trim data payload, news-first prompt |
| `output/0.export/market_research.ipynb` cell 18 | Rewrite — trim data payload, news-first prompt |
| `output/0.export/market_research.ipynb` cell 24 | Rewrite — trim data payload, news-first prompt |
| `output/0.export/market_research.ipynb` cell 32 | Rewrite — trim data payload, news-first prompt |
| `output/0.export/market_research.ipynb` cells 33–36 (new) | Insert — top-down per-country + global synthesis |

---

## Task 1: Rewrite cell 7 — LLM helper infrastructure + system prompt

**Files:**
- Modify: `output/0.export/market_research.ipynb` cell-7

### Why
`llm_explain()` currently passes no tools, has no date context, and the system prompt instructs pure statistical analysis. All three need to change before any commentary cell can produce news-driven output.

- [ ] **Step 1: Replace cell 7 source with the new implementation**

Use `NotebookEdit` with `cell_id="cell-7"`, `new_source`:

```python
# ============================================================
# LLM Commentary Helper  (web search + persistent conversation)
# Requires: pip install openai
# Set environment variable: OPENAI_API_KEY=your-key
# ============================================================
import os
import datetime
from IPython.display import display, Markdown

OPENAI_MODEL = "gpt-5-mini"  # change to "gpt-4o-mini" for lower cost

_conversation_history = []

_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "당신은 글로벌 대형 기관투자자를 대상으로 미국, 유럽, 일본, 중국, 한국 시장의 "
        "멀티팩터 포트폴리오 전략을 수립하는 '뉴스 통합형 시니어 퀀트 전략가(News-Integrated Chief Quant Strategist)'입니다.\n\n"
        "【핵심 분석 원칙】\n"
        "1. 최신 뉴스 우선: 반드시 현재 날짜 기준 최근 2~4주 이내의 실제 시장 뉴스, 거시경제 이벤트, 정책 발표, 기업 공시를 먼저 검색하고 이를 분석의 중심으로 삼으십시오.\n"
        "2. 팩터 데이터는 보조 근거: 제공된 Z-스코어, 방향성(상승/하락) 등 팩터 데이터는 뉴스 해석을 뒷받침하는 정량적 확인 도구로만 활용하십시오. 수치를 직접 나열하는 것을 금지합니다.\n"
        "3. 이벤트-드리븐 내러티브: 모든 분석은 '어떤 사건/정책/테마가 이 팩터·섹터·종목을 움직이는가'라는 질문에 답하는 형식으로 서술하십시오.\n"
        "4. Bottom-line Up Front: 가장 중요한 뉴스 기반 투자 시그널부터 시작하고, 수치는 괄호 안에 간략히 표기하십시오.\n"
        "5. 격식체 한국어: 투자위원회(IC) 제출용 보고서에 부합하는 명사형 어미(~함, ~임, ~판단, ~권고)를 사용하십시오.\n"
        "6. 최신성 보장: 검색 시 반드시 현재 연도·월을 포함한 쿼리를 사용하여 최신 뉴스만 인용하십시오."
    )
}

def reset_conversation():
    global _conversation_history
    _conversation_history = []
    print("대화 기록이 초기화되었습니다.")

def llm_explain(data_str: str, prompt: str, use_search: bool = True) -> str:
    global _conversation_history
    try:
        from openai import OpenAI
        import httpx
        key = os.getenv('OPENAI_API_KEY', '')
        if not key:
            return "_LLM 해설을 사용할 수 없습니다 — `OPENAI_API_KEY` 환경 변수를 설정하세요._"

        client = OpenAI(api_key=key, http_client=httpx.Client(verify=False))
        today = datetime.date.today().isoformat()
        date_preamble = f"[오늘 날짜: {today}. 최근 2~4주 이내의 최신 뉴스를 검색하여 분석에 활용하십시오.]\n\n"
        user_message = {
            "role": "user",
            "content": date_preamble + prompt + "\n\nData:\n" + data_str
        }
        messages = [_SYSTEM_MESSAGE] + _conversation_history + [user_message]
        kwargs = dict(model=OPENAI_MODEL, input=messages)
        if use_search:
            kwargs["tools"] = [{"type": "web_search_preview"}]
        response = client.responses.create(**kwargs)
        reply = response.output_text
        _conversation_history.append(user_message)
        _conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except ImportError:
        return "_`openai` 패키지가 설치되지 않았습니다 — `pip install openai`를 실행하세요._"
    except Exception as e:
        return f"_LLM 오류: {e}_"

def show_explanation(data_str: str, prompt: str, title: str = "분석 해설", use_search: bool = True):
    text = llm_explain(data_str, prompt, use_search=use_search)
    display(Markdown(f"---\n### 💬 {title}\n\n{text}\n\n---"))

reset_conversation()
print(f"LLM 헬퍼 로드 완료. 모델: {OPENAI_MODEL}")
print("대화 메모리: 활성화 (초기화하려면 reset_conversation() 호출)")
print("웹 검색: 활성화 (최신 뉴스 기반 분석)")
print("API 키:", "✓ 설정됨" if os.getenv('OPENAI_API_KEY') else "✗ 미설정 (OPENAI_API_KEY 환경 변수를 설정하세요)")
```

- [ ] **Step 2: Verify the cell runs without error**

Run cell 7. Expected console output:
```
대화 기록이 초기화되었습니다.
LLM 헬퍼 로드 완료. 모델: gpt-5-mini
대화 메모리: 활성화 (초기화하려면 reset_conversation() 호출)
웹 검색: 활성화 (최신 뉴스 기반 분석)
API 키: ✓ 설정됨
```

- [ ] **Step 3: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "feat: add web_search_preview + news-first system prompt to llm_explain"
```

---

## Task 2: Rewrite cell 14 — Historical Analysis commentary

**Files:**
- Modify: `output/0.export/market_research.ipynb` cell-14

### Why
Current prompt narrates full statistical tables (Sharpe, MaxDD, acceleration %). New prompt passes only top-3 / bottom-3 factor direction signals per country and asks the LLM to search for the macro/policy news behind those movements.

- [ ] **Step 1: Replace cell 14 source**

Use `NotebookEdit` with `cell_id="cell-14"`, `new_source`:

```python
# === LLM Commentary: Historical Analysis (News-Driven) ===
_hist_txt = f"HISTORICAL ANALYSIS — Factor Direction Summary (week ending approx. {datetime.date.today().isoformat()})\n\n"

for _c in COUNTRIES:
    _hs = data['factor_hs_clipped'][_c]
    _ret4  = (_hs.iloc[-1] / _hs.iloc[-5]  - 1) * 100 if len(_hs) >= 5 else pd.Series(dtype=float)
    _prior4 = (_hs.iloc[-5] / _hs.iloc[-9] - 1) * 100 if len(_hs) >= 9 else pd.Series(dtype=float)
    _accel = (_ret4 - _prior4).dropna().sort_values(ascending=False)
    _top3 = list(_accel.head(3).index)
    _bot3 = list(_accel.tail(3).index)
    _hist_txt += f"{_c.upper()}: 가속 팩터={', '.join(_top3)} | 감속 팩터={', '.join(_bot3)}\n"

_hist_prompt = (
    "당신은 글로벌 퀀트 전략가입니다. 제공된 국가별 팩터 가속/감속 방향 요약을 바탕으로 각 시장의 최신 거시경제 뉴스를 검색하십시오.\n\n"
    "검색 지시: 각 국가별로 '[국가명] stock market factor style rotation news [현재연월]', '[국가명] 금리 통화정책 최신 뉴스' 쿼리로 최신 이벤트를 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 핵심 뉴스 드라이버 (국가별): 각 시장에서 팩터 흐름을 주도하는 실제 이벤트/정책/테마를 뉴스 기반으로 서술. 팩터 방향은 괄호 안에 간략 표기.\n"
    "2. 글로벌 공통 테마: 여러 시장에서 동시에 관찰되는 스타일 로테이션의 배경 이벤트.\n"
    "3. 단기 투자 시그널: 뉴스 흐름상 향후 2~4주 주목할 국가/팩터 조합 2~3개.\n\n"
    "분량: A4 0.5페이지 이내. 수치 나열 금지. 뉴스 내러티브 우선."
)

show_explanation(_hist_txt, _hist_prompt, title="1.3 Historical Analysis: 뉴스 기반 팩터 동인 분석")
```

- [ ] **Step 2: Run cell 14 and verify**

Expected: LLM output that opens with a news/event headline (e.g., Fed decision, BOJ policy, etc.), not a statistical table. Output should reference real recent events.

- [ ] **Step 3: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "feat: rewrite historical analysis commentary to news-driven prompt"
```

---

## Task 3: Rewrite cell 18 — Stock Market Analysis commentary

**Files:**
- Modify: `output/0.export/market_research.ipynb` cell-18

### Why
Current prompt passes raw ER percentiles, z-scores, and 4w/8w numerical changes for 5 countries. New prompt passes only a 3-word signal (강세/중립/약세 + 추세 방향) per country.

- [ ] **Step 1: Replace cell 18 source**

Use `NotebookEdit` with `cell_id="cell-18"`, `new_source`:

```python
# === LLM Commentary: Stock Market Analysis (News-Driven) ===
_market_txt = f"STOCK MARKET DIRECTION SUMMARY (as of {datetime.date.today().isoformat()})\n\n"

for _c in COUNTRIES:
    _zscore = data['index_ER'][_c]['ER_zscore'].iloc[0]
    _idx_ts = data['index_ts'][_c]['index_ER']
    _cur  = float(_idx_ts.iloc[-1])
    _z4w  = float(_idx_ts.iloc[-5]) if len(_idx_ts) >= 5 else _cur
    _trend = "상승" if _cur > _z4w else "하락"
    _signal = "강세" if _zscore > 0.5 else ("약세" if _zscore < -0.5 else "중립")
    _pct_pos = (data['total_ER'][_c]['ER'].dropna() > 0).mean() * 100
    _market_txt += f"{_c.upper()}: 시그널={_signal} | ER추세={_trend} | 양(+)ER비율={_pct_pos:.0f}%\n"

_market_prompt = (
    "당신은 글로벌 자산배분 전략가입니다. 제공된 5개 시장의 방향성 요약을 바탕으로 각 국가의 최신 시장 뉴스를 검색하십시오.\n\n"
    "검색 지시: 각 국가별로 '[국가명] stock market outlook [현재연월]', '[국가명] 금리 중앙은행 최신 결정', '[국가명] 기업실적 시즌 뉴스' 쿼리로 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 국가별 시장 뉴스 브리핑 (표): 국가 | 핵심 이벤트/뉴스 | ER 방향 | 배분의견(OW/N/UW)\n"
    "2. 강세/약세 배경 뉴스: 가장 강한 시장과 가장 약한 시장 각각의 뉴스 기반 설명.\n"
    "3. 글로벌 리스크 이벤트: 전 세계 시장에 공통 영향을 미치는 리스크 요인 (뉴스 기반).\n\n"
    "분량: A4 0.5페이지. 수치 나열 금지. 뉴스 이벤트 중심 서술."
)

show_explanation(_market_txt, _market_prompt, title="1.4 Stock Market Analysis: 뉴스 기반 시장 동향")
```

- [ ] **Step 2: Run cell 18 and verify**

Expected: table of countries with news-driven rationale per country, not numerical ER tables. Output should name specific events (elections, rate decisions, earnings season, etc.).

- [ ] **Step 3: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "feat: rewrite stock market commentary to news-driven prompt"
```

---

## Task 4: Rewrite cell 24 — Factor Analysis commentary

**Files:**
- Modify: `output/0.export/market_research.ipynb` cell-24

### Why
Current prompt passes a dense data block with z-scores, 4w changes, and cross-country premium tables for 6 factors × 5 countries. New prompt passes only per-factor global consensus direction (rising in N of 5 markets) and each country's top/bottom factor.

- [ ] **Step 1: Replace cell 24 source**

Use `NotebookEdit` with `cell_id="cell-24"`, `new_source`:

```python
# === LLM Commentary: Factor Analysis (News-Driven) ===
_factor_txt = f"FACTOR DIRECTION SUMMARY (as of {datetime.date.today().isoformat()})\n\n"

_factors_list = ['BM', 'HML', 'SMB', 'MOM', 'OP', 'INV']

_factor_txt += "글로벌 팩터 방향 (4주 기준, 국가별 상승↑/하락↓):\n"
for _f in _factors_list:
    _col = f"{_f}_ER"
    _dirs = []
    for _c in COUNTRIES:
        _fts = data['factor_ts'][_c]
        if _col in _fts.columns and len(_fts) >= 5:
            _chg = float(_fts[_col].iloc[-1]) - float(_fts[_col].iloc[-5])
            _dirs.append(f"{_c.upper()}:{'↑' if _chg > 0 else '↓'}")
        else:
            _dirs.append(f"{_c.upper()}:?")
    _up = sum(1 for d in _dirs if '↑' in d)
    _consensus = "글로벌 상승" if _up >= 4 else ("글로벌 하락" if _up <= 1 else "혼조")
    _factor_txt += f"  {_f}: {' '.join(_dirs)} → {_consensus}\n"

_factor_txt += "\n국가별 최강/최약 팩터:\n"
for _c in COUNTRIES:
    _top = data['factor_ER'][_c]['ER_zscore'].idxmax()
    _bot = data['factor_ER'][_c]['ER_zscore'].idxmin()
    _factor_txt += f"  {_c.upper()}: 최강={_top} | 최약={_bot}\n"

_factor_prompt = (
    "당신은 글로벌 팩터 투자 전략가입니다. 제공된 팩터 방향성 요약을 바탕으로 스타일 로테이션 관련 최신 뉴스를 검색하십시오.\n\n"
    "검색 지시: 'global factor investing style rotation [현재연월]', 'value vs growth stocks news [현재연월]', "
    "'momentum factor performance [현재연월]', 'small cap vs large cap stocks [현재연월]' 쿼리로 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 팩터 로테이션 배경 뉴스 (표): 팩터 | 글로벌 방향 | 핵심 뉴스 드라이버\n"
    "2. 주목 팩터 2~3개: 최근 뉴스에서 가장 강한 촉매가 확인된 팩터와 그 배경 이벤트.\n"
    "3. 경고 시그널: 뉴스 흐름과 팩터 방향이 충돌하는 반전 위험 조합.\n"
    "4. 전술적 팩터 포지셔닝: 현재 뉴스 흐름상 향후 4주 유효한 팩터 롱/숏 추천.\n\n"
    "분량: A4 0.5페이지. 수치 나열 금지. 뉴스 내러티브 우선."
)

show_explanation(_factor_txt, _factor_prompt, title="1.5 Factor Analysis: 뉴스 기반 팩터 로테이션 분석")
```

- [ ] **Step 2: Run cell 24 and verify**

Expected: factor table with news column (e.g., "HML: 글로벌 상승 | 이유: 금리 고점 논쟁, 가치주 재평가 뉴스"). No raw z-score tables.

- [ ] **Step 3: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "feat: rewrite factor analysis commentary to news-driven prompt"
```

---

## Task 5: Rewrite cell 32 — Sector Analysis commentary

**Files:**
- Modify: `output/0.export/market_research.ipynb` cell-32

### Why
Current prompt passes sector beta tables, full z-score matrices, 4w ER change grids, and top-20 equity lists with raw numbers. New prompt passes only top-2/bottom-2 sectors per country and top-5 stock names+sectors (no raw ER numbers).

- [ ] **Step 1: Replace cell 32 source**

Use `NotebookEdit` with `cell_id="cell-32"`, `new_source`:

```python
# === LLM Commentary: Sector Analysis (News-Driven) ===
_sector_txt = f"SECTOR & TOP STOCKS SUMMARY (as of {datetime.date.today().isoformat()})\n\n"

_sector_txt += "섹터 방향성 (상위 2 / 하위 2):\n"
for _c in COUNTRIES:
    _sec_z = data['sector_ER'][_c].copy().set_index('Sector')['ER_zscore'].dropna().sort_values(ascending=False)
    _top2 = list(_sec_z.head(2).index)
    _bot2 = list(_sec_z.tail(2).index)
    _sector_txt += f"  {_c.upper()}: 상위={', '.join(_top2)} | 하위={', '.join(_bot2)}\n"

_sector_txt += "\n국가별 상위 5개 종목 (종목명 | 섹터):\n"
for _c in COUNTRIES:
    _tb = data['top_bot'][_c].copy()
    _tb['SectorName'] = _tb['sector'].map(lambda x: SECTOR_NAMES.get(str(x), str(x)))
    if 'ER' in _tb.columns:
        _top5 = _tb.nlargest(5, 'ER')[['Name', 'SectorName']]
        _entries = [f"{row['Name']} ({row['SectorName']})" for _, row in _top5.iterrows()]
        _sector_txt += f"  {_c.upper()}: {' | '.join(_entries)}\n"

_sector_prompt = (
    "당신은 글로벌 섹터 및 종목 분석가입니다. 제공된 섹터/종목 요약을 바탕으로 관련 최신 뉴스를 검색하십시오.\n\n"
    "검색 지시: 상위 섹터별로 '[섹터명] sector stocks outlook [현재연월]', 상위 종목별로 '[종목명] news [현재연월]' 쿼리로 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 글로벌 섹터 테마 뉴스 (표): 섹터 | 핵심 뉴스 테마 | 수혜 국가\n"
    "2. 국가별 섹터 특이점: 글로벌 트렌드와 다른 국가 고유의 섹터 동향 (뉴스 근거).\n"
    "3. 상위 종목 뉴스 하이라이트: 상위 종목 중 최근 뉴스 촉매가 확인된 종목 2~3개 집중 서술.\n"
    "4. 종합 결론 — 하이 컨빅션 콜: 팩터·섹터·뉴스가 정렬된 최선호 기회 1개 + 최대 리스크 1개.\n\n"
    "분량: A4 0.5페이지. 수치 나열 금지. 뉴스 이벤트 중심."
)

show_explanation(_sector_txt, _sector_prompt, title="1.6 Sector Analysis: 뉴스 기반 섹터/종목 분석")
```

- [ ] **Step 2: Run cell 32 and verify**

Expected: sector table with news themes column, named company news highlights. No beta tables or z-score grids.

- [ ] **Step 3: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "feat: rewrite sector analysis commentary to news-driven prompt"
```

---

## Task 6: Insert new top-down section — 4 cells after cell 32

**Files:**
- Modify: `output/0.export/market_research.ipynb` — insert cells at position 33 (before the existing nbconvert cell)

### Why
This is the new top-down analysis section: per-country industry→company drill-down, then a global cross-market synthesis. Both rely on `web_search_preview` to fetch current news for specific industries and companies.

**Important:** insert cells in reverse order (last cell first) because each insertion at position 33 shifts all subsequent cells. Insert order: Cell D (global synthesis code) → Cell C (global header markdown) → Cell B (per-country code) → Cell A (top-down header markdown).

- [ ] **Step 1: Insert Cell D — global synthesis code at position 33**

Use `NotebookEdit` with `cell_id="cell-33"`, `edit_type="insert_before"`, `new_source`:

```python
# Global Cross-Market Synthesis
_global_txt = f"GLOBAL CROSS-MARKET SYNTHESIS (as of {datetime.date.today().isoformat()})\n\n"

_sec_global = pd.DataFrame()
for _c in COUNTRIES:
    _ie = data['sector_ER'][_c].copy().set_index('Sector')
    _sec_global[_c.upper()] = _ie['ER_zscore']
_global_avg = _sec_global.mean(axis=1).dropna().sort_values(ascending=False)
_top2_sectors = list(_global_avg.head(2).index)
_global_txt += f"글로벌 최강 섹터: {', '.join(_top2_sectors)}\n\n"

_global_txt += "섹터별 국가 대표 종목:\n"
for _sec in _top2_sectors:
    _global_txt += f"\n[{_sec}]\n"
    for _c in COUNTRIES:
        _tb = data['top_bot'][_c].copy()
        _tb['SectorName'] = _tb['sector'].map(lambda x: SECTOR_NAMES.get(str(x), str(x)))
        _sec_stocks = _tb[_tb['SectorName'] == _sec].nlargest(1, 'ER') if 'ER' in _tb.columns else pd.DataFrame()
        if not _sec_stocks.empty:
            _name = str(_sec_stocks.iloc[0].get('Name', ''))
            _global_txt += f"  {_c.upper()}: {_name}\n"
        else:
            _global_txt += f"  {_c.upper()}: (해당 섹터 데이터 없음)\n"

_current_month = datetime.date.today().strftime('%Y %B')
_global_prompt = (
    f"당신은 글로벌 크로스마켓 전략가입니다. 제공된 글로벌 최강 섹터 및 국가별 대표 종목을 바탕으로 크로스마켓 최신 뉴스를 검색하십시오.\n\n"
    f"검색 지시: '{', '.join(_top2_sectors)} global sector outlook {_current_month}', "
    f"각 대표 종목명으로 '[종목명] {datetime.date.today().year} stock news' 형태로 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 글로벌 섹터 테마 뉴스: 최강 섹터가 전 세계적으로 강세인 공통 매크로/이벤트 배경.\n"
    "2. 국가별 차별화 포인트: 같은 섹터 내에서 어떤 국가의 노출도가 뉴스 측면에서 가장 매력적인지 비교.\n"
    "3. 글로벌 하이 컨빅션 픽: 팩터·섹터·뉴스 촉매가 가장 정렬된 2~3개 종목과 투자 근거.\n"
    "4. 글로벌 테일 리스크: 현재 뉴스 환경에서 이 섹터 포지션을 위협하는 최대 리스크 1개.\n\n"
    "분량: A4 0.5페이지. 수치 나열 금지. 뉴스 내러티브 최우선."
)

show_explanation(_global_txt, _global_prompt, title="글로벌 크로스마켓 종합: 최강 섹터 & 하이 컨빅션 픽")
```

- [ ] **Step 2: Insert Cell C — global synthesis markdown header at position 33**

Use `NotebookEdit` with `cell_id="cell-33"`, `edit_type="insert_before"`, `cell_type="markdown"`, `new_source`:

```markdown
---
## Global Cross-Market Synthesis
```

- [ ] **Step 3: Insert Cell B — per-country top-down code at position 33**

Use `NotebookEdit` with `cell_id="cell-33"`, `edit_type="insert_before"`, `new_source`:

```python
# Top-Down: Per-Country Industry → Company Analysis
_current_month = datetime.date.today().strftime('%Y %B')

for _c in COUNTRIES:
    _td_txt = f"TOP-DOWN ANALYSIS: {_c.upper()} (as of {datetime.date.today().isoformat()})\n\n"

    # Top 3 industries by ER z-score
    _ind_z = data['industry_ER'][_c].copy()
    if 'industry' in _ind_z.columns:
        _ind_z = _ind_z.set_index('industry')
    if 'ER_zscore' in _ind_z.columns:
        _top3_ind = _ind_z['ER_zscore'].dropna().nlargest(3)
        _td_txt += "상위 3개 산업 (ER 강세 기준):\n"
        for _ind_code, _z in _top3_ind.items():
            _td_txt += f"  Industry code {_ind_code}\n"

    # Top 5 stocks
    _tb = data['top_bot'][_c].copy()
    _tb['SectorName'] = _tb['sector'].map(lambda x: SECTOR_NAMES.get(str(x), str(x)))
    if 'ER' in _tb.columns:
        _top5 = _tb.nlargest(5, 'ER')[['Name', 'SectorName']]
        _td_txt += "\n상위 5개 종목 (종목명 | 섹터):\n"
        for _, _row in _top5.iterrows():
            _td_txt += f"  {_row['Name']} | {_row['SectorName']}\n"

    _td_prompt = (
        f"당신은 {_c.upper()} 시장 전문 애널리스트입니다. 제공된 상위 산업 및 종목을 바탕으로 {_c.upper()} 시장의 최신 뉴스를 검색하십시오.\n\n"
        f"검색 지시: '{_c} stock market top sector industry news {_current_month}', "
        f"각 상위 종목명으로 '[종목명] {datetime.date.today().year} news' 형태로 검색하십시오.\n\n"
        "출력 구조 (탑다운 내러티브):\n"
        "1. 산업 테마: 상위 산업이 강세인 이유를 최신 뉴스 이벤트로 설명 (정책, 실적, 매크로).\n"
        "2. 종목 드릴다운: 상위 종목 중 뉴스 촉매가 확인된 종목 2~3개 — 어떤 사건이 해당 종목을 지지하는지 구체적 서술.\n"
        "3. 컨빅션 콜: 산업 테마와 종목 뉴스가 가장 정렬된 1개 종목과 투자 근거 한 문장.\n\n"
        "분량: A4 0.3페이지. 수치 나열 금지. 뉴스 이벤트 중심."
    )

    show_explanation(_td_txt, _td_prompt, title=f"탑다운 분석: {_c.upper()} — 산업 → 종목")
```

- [ ] **Step 4: Insert Cell A — top-down section markdown header at position 33**

Use `NotebookEdit` with `cell_id="cell-33"`, `edit_type="insert_before"`, `cell_type="markdown"`, `new_source`:

```markdown
---
## Top-Down Analysis: Industry to Company
```

- [ ] **Step 5: Run all 4 new cells and verify**

- Cell A renders the markdown header.
- Cell B runs without Python error; each country produces a 💬 block that opens with an industry news narrative (not a z-score table).
- Cell C renders the markdown header.
- Cell D runs without Python error; output opens with a global sector theme from recent news and names 2–3 specific high-conviction picks.

- [ ] **Step 6: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "feat: add news-driven top-down industry-to-company + global synthesis section"
```

---

## Task 7: Full notebook end-to-end smoke run

**Files:**
- Run: `output/0.export/market_research.ipynb` (all cells)

### Why
Verify that the cell index shifts from the 4 insertions in Task 6 did not break the export cells (now at positions 37–38), and that the conversation memory accumulates correctly across all 6 LLM commentary cells.

- [ ] **Step 1: Restart kernel and run all cells**

In JupyterLab/Notebook: Kernel → Restart & Run All.

Expected: all cells complete without `NameError` or `KeyError`. The 6 LLM commentary cells each produce a 💬 Markdown block that references real-world news events.

- [ ] **Step 2: Verify export cells still work**

The nbconvert cell (now at the new position after the 4 insertions) should still output:
```
[NbConvertApp] Writing ... bytes to market_research.html
```

- [ ] **Step 3: Final commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "chore: verify notebook end-to-end after news-driven refactor"
```

---

## Self-Review Checklist

- [x] Spec coverage: cell 7 infra ✓, system prompt ✓, cells 14/18/24/32 rewrites ✓, 4 new cells ✓, date injection ✓, up-to-date search queries ✓
- [x] No placeholders: all code blocks are complete and runnable
- [x] Type consistency: `llm_explain(data_str, prompt, use_search)` signature used consistently; `show_explanation` passes `use_search=True` by default; `datetime.date.today()` available after Task 1's `import datetime`
- [x] `_current_month` defined locally in each cell that uses it (Tasks 6 cells B and D) — no cross-cell dependency
- [x] Insert order in Task 6 is reverse (D→C→B→A) so final cell ordering is A, B, C, D at positions 33–36
