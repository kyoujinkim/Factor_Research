# Narrative Architecture Redesign — market_research.ipynb

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the notebook from data-type ordering to investment-logic ordering so that each LLM commentary builds explicitly on prior sections, converging to a single global thesis with regional implications.

**Architecture:** A one-shot Python rebuild script reads the original notebook JSON, reorganises cells into 6 narrative sections, rewrites all LLM prompt strings to chain conclusions, adds 2 new synthesis cells (Global Thesis + Final Conviction), and writes the result back in-place.

**Tech Stack:** Python stdlib (`json`, `copy`), Jupyter nbformat 4

## Global Constraints

- Only `market_research.ipynb` is modified (no other project files).
- Rebuild script is a one-time tool; delete after use.
- All existing data-loading and visualisation code is kept verbatim — only LLM prompt strings and cell ordering change.
- `use_search=False` on the old alignment cell → `use_search=True` in new Section 4.
- `_conversation_history` accumulation in Cell 6 is preserved (not reset mid-notebook).

---

## New Section Structure

| Section | Content | Source Cells (orig) |
|---------|---------|---------------------|
| Setup | Title, imports, data load, normalise, LLM helper | 0-6 (unchanged) |
| **[1] 과거 관찰** | TS plots, stats, correlations + LLM | 8-12 + rewrite cell 13 |
| **[2] 미래 신호** | Market ER, Factor ER, Factor Premium, Sector ER + LLM | 15-21, 27-28 + rewrite cell 22 |
| **[3] 글로벌 테제 합성** | NEW synthesis LLM (no data cell) | NEW |
| **[4] 섹터·팩터 함의** | Sector exposure heatmaps + LLM | 23-25 + rewrite cell 26 |
| **[5] 국가별 시사점** | Top stocks table + per-country loop LLM | 29-30 + rewrite cell 33 |
| **[6] 최종 통합 결론** | NEW final conviction LLM | NEW |
| Export | HTML + DOCX | 35-36 (unchanged) |

Dropped cells: 14, 17, 31, 32, 34 (redundant headers or replaced content).

---

## Task 1: Write and Run `rebuild_notebook.py`

**Files:**
- Create: `output/0.export/rebuild_notebook.py`
- Modify: `output/0.export/market_research.ipynb`

- [ ] **Step 1: Write the rebuild script**

Save the file below to `C:\Users\NHWM\PycharmProjects\Factor_Research\output\0.export\rebuild_notebook.py`.

```python
"""
rebuild_notebook.py
Restructures market_research.ipynb into 6-section narrative architecture.
Run once from: C:\Users\NHWM\PycharmProjects\Factor_Research\output\0.export\
"""
import json, copy, shutil
from pathlib import Path

NB = Path('market_research.ipynb')
shutil.copy(NB, NB.with_suffix('.ipynb.bak'))   # safety backup

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

orig = nb['cells']   # 37 cells, indices 0-36

# ── helpers ──────────────────────────────────────────────────────────────
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(src: str):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": src,
    }

def cp(i):
    c = copy.deepcopy(orig[i])
    c["outputs"] = []           # clear stale outputs
    c["execution_count"] = None
    return c

# ── new prompt strings ────────────────────────────────────────────────────

HIST_LLM_SRC = r"""# === Section 1 LLM: 과거 관찰 — 팩터 역사적 동인 ===
_hist_txt = f"HISTORICAL ANALYSIS — Factor Direction Summary (week ending approx. {datetime.date.today().isoformat()}\n\n"

for _c in COUNTRIES:
    _hs = data['factor_hs_clipped'][_c]
    _ret4   = (_hs.iloc[-1] / _hs.iloc[-5]  - 1) * 100 if len(_hs) >= 5  else pd.Series(dtype=float)
    _prior4 = (_hs.iloc[-5] / _hs.iloc[-9] - 1) * 100  if len(_hs) >= 9  else pd.Series(dtype=float)
    _accel  = (_ret4 - _prior4).dropna().sort_values(ascending=False)
    _top3   = list(_accel.head(3).index)
    _bot3   = list(_accel.tail(3).index)
    _hist_txt += f"{_c.upper()}: 가속 팩터={', '.join(_top3)} | 감속 팩터={', '.join(_bot3)}\n"

_hist_prompt = (
    "역할: 글로벌 퀀트 전략가. 이 섹션이 보고서의 첫 분석 단계입니다.\n\n"
    "제공된 국가별 팩터 가속/감속 방향 요약을 바탕으로 각 시장의 최신 거시경제 뉴스를 검색하십시오.\n\n"
    "검색 지시: 각 국가별로 '[국가명] 주식시장 팩터 [현재연월]' 쿼리로 최신 이벤트를 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 국가별 팩터 동향 관찰: 각 시장에서 팩터 흐름을 주도한 뉴스 이벤트/정책/테마. 팩터 방향은 괄호 안 간략 표기.\n"
    "2. 글로벌 공통 스타일 패턴: 여러 시장에서 동시에 관찰되는 스타일 로테이션과 배경 이벤트.\n"
    "3. 주목할 이상 신호: 글로벌 트렌드와 반하는 특이 시장과 그 이유.\n\n"
    "분량: A4 0.5페이지. 수치 나열 금지. 뉴스 내러티브 우선.\n"
    "※ 이 관찰은 다음 섹션(미래 신호)의 출발점으로 이어집니다."
)

show_explanation(_hist_txt, _hist_prompt, title="Section 1: 과거 관찰 — 팩터 역사적 동인 분석")
"""

FORWARD_LLM_SRC = r"""# === Section 2 LLM: 미래 신호 — 모델의 포워드 뷰 ===
_forward_txt = f"FORWARD SIGNALS — Factor & Market Expected Returns (as of {datetime.date.today().isoformat()})\n\n"

_factors_list = ['BM', 'HML', 'SMB', 'MOM', 'OP', 'INV']

_forward_txt += "글로벌 팩터 기대수익률 방향 (4주 기준, 국가별 상승↑/하락↓):\n"
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
    _forward_txt += f"  {_f}: {' '.join(_dirs)} → {_consensus}\n"

_forward_txt += "\n국가별 시장 ER 상황:\n"
for _c in COUNTRIES:
    _mz = data['index_ER'][_c]['ER_zscore'].iloc[0]
    _top = data['factor_ER'][_c]['ER_zscore'].idxmax()
    _bot = data['factor_ER'][_c]['ER_zscore'].idxmin()
    _forward_txt += f"  {_c.upper()}: 시장ER z={_mz:.2f} | 최강팩터={_top} | 최약팩터={_bot}\n"

_forward_txt += "\n글로벌 섹터 매력도 (상위3/하위3):\n"
_sec_z = {}
for _c in COUNTRIES:
    for _, _row in data['sector_ER'][_c].iterrows():
        _sec = _row.get('Sector', '')
        if _sec:
            _sec_z.setdefault(_sec, []).append(_row.get('ER_zscore', 0))
_sec_avg = {s: sum(v)/len(v) for s, v in _sec_z.items() if v}
_sec_sorted = sorted(_sec_avg.items(), key=lambda x: x[1], reverse=True)
_forward_txt += "  상위: " + ", ".join(s for s,_ in _sec_sorted[:3]) + "\n"
_forward_txt += "  하위: " + ", ".join(s for s,_ in _sec_sorted[-3:]) + "\n"

_forward_prompt = (
    "[이전 섹션에서 확인한 팩터 동향과 뉴스 드라이버를 이어받아 분석합니다.]\n\n"
    "역할: 글로벌 퀀트 전략가. 제공되는 데이터는 '기대수익률(Expected Return)' — 모델이 예측하는 미래 팩터·시장·섹터 수익률입니다.\n\n"
    "검색 지시: 각 국가별로 '[국가명] 경제 전망 [현재연월]' 쿼리로 최신 포워드 뷰 뉴스를 검색하십시오.\n\n"
    "출력 구조:\n"
    "1. 과거 추세와의 연속성/단절: 앞서 관찰한 팩터 모멘텀이 기대수익률에서도 지속되는가, 아니면 모델이 반전을 예고하는가? (국가별 비교)\n"
    "2. 국가별 미래 포지셔닝: 기대수익률 기준 가장 매력적/비매력적인 시장과 그 이유 (뉴스 근거 포함).\n"
    "3. 주목할 섹터 방향: 기대수익률 상위 섹터가 강세인 배경 이벤트와 그 지속 가능성.\n\n"
    "분량: A4 0.5페이지. 수치 나열 금지.\n"
    "※ 다음 섹션에서 이 분석을 바탕으로 글로벌 통합 테제를 선언합니다."
)

show_explanation(_forward_txt, _forward_prompt, title="Section 2: 미래 신호 — 모델의 포워드 뷰 분석")
"""

THESIS_LLM_SRC = r"""# === Section 3 LLM: 글로벌 테제 합성 (NEW) ===
_thesis_txt = (
    f"GLOBAL THESIS SYNTHESIS (as of {datetime.date.today().isoformat()})\n\n"
    "앞선 두 섹션에서 다음이 확인됨:\n"
    "  • Section 1: 국가별 팩터 가속/감속 동향 및 뉴스 드라이버\n"
    "  • Section 2: 팩터·시장·섹터 기대수익률의 포워드 신호\n"
    "이 두 관찰을 통합하여 글로벌 투자 테제를 도출하십시오.\n"
)

_thesis_prompt = (
    "[지금까지 두 섹션의 분석 — 과거 팩터 행동 관찰과 미래 기대수익률 신호 — 을 통합합니다.]\n\n"
    "역할: 수석 투자전략가(CIO). 이 섹션에서 공식 시장 테제를 선언합니다.\n\n"
    "출력 구조:\n"
    "1. 글로벌 시장 테제 (1-2문장): "
    "\"현재 글로벌 시장은 [X 국면]에 있으며, [Y 팩터/섹터/스타일]이 유리한 환경이다. "
    "핵심 드라이버는 [Z 이벤트/정책/매크로]이다.\"\n"
    "2. 지역별 테제 강도: 이 테제가 가장 강하게 적용되는 시장 vs. 예외·이탈 시장 (이유 포함).\n"
    "3. 테제 유효 기간과 무효화 조건: 어떤 이벤트/지표가 발생하면 이 테제를 수정해야 하는가.\n\n"
    "분량: A4 0.5페이지.\n"
    "※ 이 섹션은 보고서 전체의 중심 주장입니다 — 명확하고 단호하게 서술할 것. 수치 나열 금지."
)

show_explanation(_thesis_txt, _thesis_prompt, title="Section 3: 글로벌 테제 합성")
"""

SECTOR_IMPL_LLM_SRC = r"""# === Section 4 LLM: 섹터·팩터 함의 ===
_beta_cols_map = {'BM': 'beta_bm', 'HML': 'beta_hml', 'SMB': 'beta_smb',
                  'OP': 'beta_op', 'INV': 'beta_inv', 'MOM': 'beta_mom'}

_sector_impl_txt = "FACTOR-SECTOR ALIGNMENT ANALYSIS\n\n"

_f_zscores = {}
for _f in ['BM', 'HML', 'SMB', 'MOM', 'OP', 'INV']:
    _vals = [data['factor_ER'][_c]['ER_zscore'].get(_f, 0) for _c in COUNTRIES]
    _f_zscores[_f] = sum(_vals) / len(_vals)

_strong = [f for f, z in _f_zscores.items() if z > 0.3]
_weak   = [f for f, z in _f_zscores.items() if z < -0.3]
_sector_impl_txt += f"현재 강세 팩터: {', '.join(_strong) if _strong else '없음'}\n"
_sector_impl_txt += f"현재 약세 팩터: {', '.join(_weak) if _weak else '없음'}\n\n"

_sector_impl_txt += "국가별/섹터별 팩터 정렬 점수 상세:\n"
for _c in COUNTRIES:
    _sector_impl_txt += f"  [{_c.upper()}]\n"
    _sec_df = data['industry_ER'][_c].copy()
    if 'Sector' in _sec_df.columns:
        _sec_df = _sec_df.set_index('Sector')
    for _sec in _sec_df.index:
        _factor_details = []
        for _f, _bcol in _beta_cols_map.items():
            if _bcol in _sec_df.columns and _f in _f_zscores:
                _bval = _sec_df.loc[_sec, _bcol]
                _factor_details.append(f"{_f}:{_bval:.2f}")
        _sector_impl_txt += f"    - {_sec}: {', '.join(_factor_details)}\n"
    _sector_impl_txt += "\n"

_sector_impl_prompt = (
    "[확정된 글로벌 테제를 기반으로 섹터 배분 함의를 도출합니다.]\n\n"
    "역할: 글로벌 팩터-섹터 전략가.\n\n"
    "수혜 섹터 = 강세 팩터에 높은 베타 노출 (팩터 프리미엄 수령), 역풍 섹터 = 약세 팩터에 높은 베타 노출 (팩터 드래그).\n\n"
    "출력 구조:\n"
    "1. 테제 수혜 섹터 (표): 국가 | 섹터 | 수혜 근거 (팩터-섹터 정렬 + 뉴스 촉매) — 국가별 3개씩\n"
    "2. 테제 역풍 섹터 (표): 국가 | 섹터 | 역풍 근거 — 국가별 3개씩\n"
    "3. 글로벌 오버웨이트/언더웨이트 권고: 테제에 기반한 섹터 배분 방향 요약.\n\n"
    "수치 나열 금지. 섹터 배분 관점의 핵심만 서술.\n"
    "※ 다음 섹션에서 이 섹터 방향을 구체적인 종목으로 연결합니다."
)

show_explanation(_sector_impl_txt, _sector_impl_prompt,
                 title="Section 4: 섹터·팩터 함의 — 테제 기반 섹터 배분", use_search=True)
"""

COUNTRY_LLM_SRC = r"""# === Section 5 LLM: 국가별 시사점 (per-country loop) ===
for _c in COUNTRIES:
    _td_txt = f"TOP-DOWN ANALYSIS: {_c.upper()} (as of {datetime.date.today().isoformat()})\n\n"

    _ind_z = data['industry_ER'][_c].copy()
    if 'industry' in _ind_z.columns:
        _ind_z = _ind_z.set_index('industry')
    if 'ER_zscore' in _ind_z.columns:
        _top5_ind = _ind_z['ER_zscore'].dropna().nlargest(5)
        _td_txt += "상위 5개 산업 (ER 강세 기준):\n"
        for _ind_code, _z in _top5_ind.items():
            _td_txt += f"  Industry code {_ind_code}\n"

    _tb = data['top_bot'][_c].copy()
    _tb['SectorName'] = _tb['sector'].map(lambda x: SECTOR_NAMES.get(str(x), str(x)))
    if 'ER' in _tb.columns:
        _top10 = _tb.nlargest(10, 'ER')[['Name', 'SectorName']]
        _td_txt += "\n상위 10개 종목 (종목명 | 섹터):\n"
        for _, _row in _top10.iterrows():
            _td_txt += f"  {_row['Name']} | {_row['SectorName']}\n"

    _td_prompt = (
        f"[확정된 글로벌 테제와 섹터 배분 방향을 기반으로 {_c.upper()} 시장 시사점을 도출합니다.]\n\n"
        f"역할: {_c.upper()} 시장 전문 애널리스트.\n\n"
        f"검색 지시: '[{_c}] [상위 산업명] [{datetime.date.today().strftime('%Y %B')}]' 쿼리로, "
        f"각 상위 종목명으로 '[종목명] {datetime.date.today().year} news' 형태로 검색하십시오.\n\n"
        "출력 구조 (탑다운):\n"
        f"1. 테제의 {_c.upper()} 적용: 글로벌 테제가 이 시장에서 어떻게 강화 또는 완화되는가? 시장 특수성 설명.\n"
        "2. 수혜 산업 테마: 테제에 정렬된 상위 산업과 최신 뉴스 촉매.\n"
        "3. 컨빅션 픽 3개: 산업 테마 + 종목 뉴스가 가장 정렬된 종목. 각 투자 근거 한 문장.\n\n"
        "분량: A4 0.3페이지. 뉴스 이벤트 중심. 한국어로 작성."
    )

    show_explanation(_td_txt, _td_prompt, title=f"Section 5: {_c.upper()} 시장 시사점")
"""

FINAL_LLM_SRC = r"""# === Section 6 LLM: 최종 통합 결론 (NEW) ===
_final_txt = (
    f"FINAL SYNTHESIS (as of {datetime.date.today().isoformat()})\n\n"
    "모든 섹션 분석 완료:\n"
    "  • Section 1: 과거 팩터 관찰 + 뉴스 드라이버\n"
    "  • Section 2: 미래 기대수익률 포워드 뷰\n"
    "  • Section 3: 글로벌 테제 선언\n"
    "  • Section 4: 섹터·팩터 배분 함의\n"
    "  • Section 5: 5개 국가별 컨빅션 픽\n"
    "이 모든 분석을 하나로 통합하십시오.\n"
)

_final_prompt = (
    "[모든 섹션의 분석 — 과거 관찰, 미래 신호, 글로벌 테제, 섹터 함의, 국가별 시사점 — 을 통합합니다.]\n\n"
    "역할: 수석 투자전략가(CIO). 투자위원회(IC) 제출용 최종 결론 페이지입니다.\n\n"
    "출력 구조:\n"
    "1. 확정 글로벌 테제 (1문장): 모든 분석이 수렴한 최종 시장 관점.\n"
    "2. 국가별 비중 방향 (표): 국가 | 비중 방향(오버/뉴트럴/언더) | 핵심 근거 1줄\n"
    "3. 최선호 섹터·팩터 (글로벌 상위 3개): 각 한 줄 근거.\n"
    "4. 고확신 종목 (글로벌 상위 5-10개): 종목명 | 국가 | 핵심 투자 테제 (한 문장)\n"
    "5. 핵심 리스크: 테제를 무효화할 수 있는 이벤트 3개.\n\n"
    "분량: A4 1페이지. 의사결정자가 이 페이지만 읽어도 포지션을 잡을 수 있어야 함."
)

show_explanation(_final_txt, _final_prompt, title="Section 6: 최종 통합 결론 — IC 제출용 투자 결론")
"""

# ── assemble new cell list ────────────────────────────────────────────────
new_cells = []

# ── Setup (unchanged) ────────────────────────────────────────────────────
new_cells += [cp(i) for i in range(7)]          # cells 0-6

# ── Section 1: 과거 관찰 ────────────────────────────────────────────────
new_cells.append(md("---\n## Section 1: 과거 관찰 — 팩터 역사적 행동\n\n"
                    "과거 팩터 시계열을 분석하여 현재 시장 국면을 진단합니다."))
new_cells.append(cp(8))                          # Factor TS header
new_cells.append(cp(9))                          # Factor TS plots
new_cells.append(cp(10))                         # Factor stats
new_cells.append(cp(11))                         # Correlation header
new_cells.append(cp(12))                         # Correlation heatmaps
new_cells.append(code(HIST_LLM_SRC))             # LLM — rewritten

# ── Section 2: 미래 신호 ────────────────────────────────────────────────
new_cells.append(md("---\n## Section 2: 미래 신호 — 모델의 포워드 뷰\n\n"
                    "기대수익률(Expected Return)을 통해 모델이 예측하는 미래를 확인합니다."))
new_cells.append(cp(15))                         # Market ER header
new_cells.append(cp(16))                         # Market ER plots
new_cells.append(cp(18))                         # Factor ER header
new_cells.append(cp(19))                         # Factor ER table
new_cells.append(cp(20))                         # Factor Premium header
new_cells.append(cp(21))                         # Factor premium chart
new_cells.append(cp(27))                         # Sector ER header (moved)
new_cells.append(cp(28))                         # Sector ER (moved)
new_cells.append(code(FORWARD_LLM_SRC))          # LLM — rewritten

# ── Section 3: 글로벌 테제 합성 (NEW) ───────────────────────────────────
new_cells.append(md("---\n## Section 3: 글로벌 테제 합성\n\n"
                    "과거 관찰과 미래 신호를 통합하여 하나의 시장 테제를 도출합니다."))
new_cells.append(code(THESIS_LLM_SRC))           # LLM — NEW

# ── Section 4: 섹터·팩터 함의 ───────────────────────────────────────────
new_cells.append(md("---\n## Section 4: 섹터·팩터 함의\n\n"
                    "확정된 테제를 기반으로 섹터 배분 방향을 도출합니다."))
new_cells.append(cp(24))                         # Sector Factor Exposures header
new_cells.append(cp(25))                         # Sector heatmaps
new_cells.append(code(SECTOR_IMPL_LLM_SRC))     # LLM — rewritten (use_search=True)

# ── Section 5: 국가별 시사점 ─────────────────────────────────────────────
new_cells.append(md("---\n## Section 5: 국가별 시사점\n\n"
                    "글로벌 테제가 각 시장에서 어떻게 나타나는지, "
                    "그리고 그 안에서 가장 유망한 종목을 도출합니다."))
new_cells.append(cp(29))                         # Top equities header
new_cells.append(cp(30))                         # Top stocks tables
new_cells.append(code(COUNTRY_LLM_SRC))          # LLM — rewritten

# ── Section 6: 최종 통합 결론 (NEW) ──────────────────────────────────────
new_cells.append(md("---\n## Section 6: 최종 통합 결론\n\n"
                    "투자위원회(IC) 제출용 최종 결론입니다."))
new_cells.append(code(FINAL_LLM_SRC))            # LLM — NEW

# ── Export ───────────────────────────────────────────────────────────────
new_cells.append(cp(35))                         # HTML export
new_cells.append(cp(36))                         # DOCX export

# ── write back ───────────────────────────────────────────────────────────
nb['cells'] = new_cells
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. {len(new_cells)} cells written (was {len(orig)}).")
print("Backup saved to market_research.ipynb.bak")
```

- [ ] **Step 2: Run the script**

```bash
cd C:\Users\NHWM\PycharmProjects\Factor_Research\output\0.export
python rebuild_notebook.py
```

Expected output:
```
Done. 34 cells written (was 37).
Backup saved to market_research.ipynb.bak
```

- [ ] **Step 3: Delete the temporary script**

```bash
del rebuild_notebook.py
```

---

## Task 2: Verify the Restructured Notebook

- [ ] **Step 1: Confirm cell count and section headers**

Open `market_research.ipynb` in Jupyter or check via Python:

```python
import json
with open('market_research.ipynb') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    preview = src[:80].replace('\n', ' ')
    print(f"[{i:02d}] {c['cell_type'][:4]} | {preview}")
```

Expected: Sections 1-6 headers appear at indices ~7, ~14, ~24, ~27, ~31, ~36.

- [ ] **Step 2: Spot-check two key prompt changes**

Verify Cell 13 (old) no longer contains `"결론을 말하지 말고"`:

```python
src = ''.join(nb['cells'][13]['source'])
assert '결론을 말하지 말고' not in src, "Old restriction phrase still present!"
print("✓ Old restriction removed")
```

Verify the global thesis cell exists:

```python
thesis_found = any(
    '글로벌 테제 합성' in ''.join(c['source'])
    for c in nb['cells'] if c['cell_type'] == 'code'
)
assert thesis_found, "Thesis synthesis cell not found!"
print("✓ Global thesis cell present")
```

- [ ] **Step 3: Commit**

```bash
git add output/0.export/market_research.ipynb
git commit -m "refactor: restructure notebook into 6-section narrative architecture

- Data-type ordering → investment-logic ordering (observe → signal → thesis → sector → country → conclusion)
- Add Global Thesis Synthesis cell (Section 3) and Final Conviction cell (Section 6)
- All LLM prompts now chain conclusions from prior sections explicitly
- Remove 'do not state conclusions' restriction from Section 1
- Factor-sector alignment cell now uses web search (use_search=True)
- Dropped redundant cells: 14, 17, 31, 32, 34"
```

---

## Verification

Run the notebook top-to-bottom (Kernel → Restart & Run All) with a valid `OPENAI_API_KEY` and confirm:

1. Each LLM commentary references the prior section explicitly in its output text
2. Section 3 produces a clear 1-2 sentence global thesis statement
3. Section 6 produces a table of global conviction picks with country attribution
4. The narrative reads as a building argument, not independent silos
