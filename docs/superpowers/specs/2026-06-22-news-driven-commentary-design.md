# Design Spec: News-Driven LLM Commentary for market_research.ipynb

**Date:** 2026-06-22  
**Scope:** `output/0.export/market_research.ipynb`

---

## Goal

Replace the current statistics-heavy LLM commentary with news-and-event-driven interpretation. All LLM calls will fetch live, up-to-date web news via OpenAI's `web_search_preview` tool before generating commentary. Statistical data is retained only as directional signal, not as the primary narrative subject.

---

## Change 1: `llm_explain()` infrastructure (cell 7)

Add a `use_search: bool = True` parameter to `llm_explain()`.

When `use_search=True`:
- Pass `tools=[{"type": "web_search_preview"}]` to `client.responses.create()`
- The model autonomously searches the web during generation

The current date (`YYYY-MM-DD`) is injected into every user message preamble so the model knows to search for news from the past 2–4 weeks.

```python
def llm_explain(data_str: str, prompt: str, use_search: bool = True) -> str:
    ...
    today = datetime.date.today().isoformat()
    date_preamble = f"[Today's date: {today}. Search for news from the past 2-4 weeks.]\n\n"
    user_message = {"role": "user", "content": date_preamble + prompt + "\n\nData:\n" + data_str}
    kwargs = dict(model=OPENAI_MODEL, input=messages)
    if use_search:
        kwargs["tools"] = [{"type": "web_search_preview"}]
    response = client.responses.create(**kwargs)
    ...
```

`show_explanation()` passes `use_search=True` by default.

---

## Change 2: System prompt update (cell 7)

The system prompt is rewritten to define the role as a **news-integrated quant strategist**:

- Search current market news and macro events first, then anchor findings to the quant signals provided
- Statistical metrics (Z-score, Sharpe, 4w change) are used only to confirm direction and relative strength — not to be narrated as the headline
- All output in Korean, IC-report style
- Prioritize news from the past 2–4 weeks; use quant data as supporting evidence

---

## Change 3: Existing commentary prompt rewrites

Four cells are updated. Each prompt is shortened and restructured:

| Cell | Section | New prompt focus |
|------|---------|-----------------|
| 14 | Historical Analysis | Search for macro/policy events driving factor performance in each market over the past month |
| 18 | Stock Market Analysis | Search for country-level market news (rate decisions, earnings seasons, geopolitical) that explains ER direction |
| 24 | Factor Analysis | Search for style-rotation news (value vs growth, small cap, momentum) driving factor premiums globally |
| 32 | Sector Analysis | Search for sector-level themes (AI, energy transition, consumer sentiment) driving sector ER differences |

**Data payload for each cell is trimmed** to: top/bottom 3 signals per country (factor name + direction), not full statistical tables.

Prompt structure for all 4 cells:
1. Instruct to search news for the relevant market/factor/sector
2. Provide trimmed quant signals as grounding
3. Output: news-narrative first, quant confirmation in parentheses

---

## Change 4: New top-down section (4 new cells before export)

Inserted between cell 32 (Sector Analysis) and cell 33 (nbconvert export).

### Cell A — Markdown header
```
## Top-Down Analysis: Industry to Company
```

### Cell B — Per-country industry → company drill-down

For each of the 5 countries:
1. Identify top 3 industries by `industry_ER` z-score
2. Build a concise data string: `{country} | Top industries: {list} | Top stocks: {list with sector}`
3. Call `show_explanation()` with `use_search=True`

Prompt instructs the LLM to:
- Search recent news for those industries in that specific market (e.g., "Korean semiconductor industry news June 2026")
- Search for the top companies by name
- Output a narrative: industry theme → company-level rationale → conviction call

### Cell C — Markdown header
```
## Global Cross-Market Synthesis
```

### Cell D — Global synthesis

1. Identify top 2 sectors by global average ER z-score
2. Collect top company per country within those sectors
3. Call `show_explanation()` with `use_search=True`

Prompt instructs the LLM to:
- Search for the cross-market macro/thematic drivers of those sectors
- Identify which country's exposure to that sector has the most compelling recent newsflow
- Output: global theme → best country-sector combination → 2–3 highest-conviction stock picks across all markets

---

## Files Changed

- `output/0.export/market_research.ipynb` — only file modified

## Out of Scope

- No new API keys required
- No changes to data loading, normalization, or charting cells
- No changes to the export cells (33, 34)
- No changes to `src/` pipeline code
