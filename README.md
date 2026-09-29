# Agentic AI Governance Optimizer

A multi-agent **LangGraph** pipeline that treats unsafe, biased, or overconfident LLM output as a **liability problem**, not a prompt-tuning afterthought.

Before any answer is marked approved, a **Governance Critic** scores the draft against a written Enterprise AI Constitution (safety, factual accuracy, neutrality). Failed drafts are rewritten by a **Refiner** and re-scored. If alignment fails within a fixed number of loops, the system **fail-closes** instead of shipping the last attempt.

Built as a portfolio piece for research / applied-AI internships working on **LLM safety, Constitutional AI, and agentic workflows**.

---

## Why this exists

Single-shot chatbots optimize for fluency. Enterprise and public-sector use needs:

- a **written policy** the model is judged against (Constitutional AI)
- a **structured reward-model analog** (numeric scores + actionable deductions), not free-form “looks fine”
- **routing in code** so a critic cannot silently pass a 6/10
- an **audit trail** a human can inspect (iteration drafts, scores, feedback)
- a **circuit breaker** so refinement cannot loop forever

This repo implements that loop with Google **Gemini**, **LangChain** structured output, and a **Streamlit** control plane.

---

## Architecture

```mermaid
flowchart TD
    Q[User query] --> R[Researcher Agent]
    R --> C[Governance Critic]
    C -->|all scores ≥ threshold| A[Approve]
    C -->|fail and iteration < max| F[Refiner Agent]
    F --> C
    C -->|fail and iteration ≥ max| X[Circuit breaker]
    A --> END1[END — approved response]
    X --> END2[END — Governance Review Failed]
```

| Node | Role |
|------|------|
| **Researcher** | Generator. First comprehensive draft under the constitution. |
| **Governance Critic** | Guardrail. Gemini + `with_structured_output(CriticEvaluation)` (Pydantic). |
| **Refiner** | Rewrites using the critic’s exact `actionable_feedback`. |
| **Approve** | Terminal success. Draft is cleared for the user. |
| **Circuit breaker** | Fail-closed after max iterations (default 3). Last draft is **not** labeled approved. |

**Conditional edge (after Critic):** pass → approve; fail under the iteration cap → refiner → critic; fail at cap → circuit breaker.

Shared graph state (`TypedDict`): `user_query`, `draft`, `critic_scores`, `feedback`, `iteration_count`, plus an append-only `audit_trail` for the UI.

---

## Critic contract (automated reward model)

The critic does not rewrite the user-facing answer. It must return:

| Field | Type | Meaning |
|-------|------|---------|
| `factual_accuracy_score` | int 1–10 | Scoped claims, uncertainty, no fabricated specifics |
| `neutrality_bias_score` | int 1–10 | Fair framing on contested topics |
| `safety_compliance_score` | int 1–10 | Constitution Article I (harm / dual-use) |
| `actionable_feedback` | `list[str]` | Concrete rewrite instructions |
| `pass_fail` | bool | Routing signal |

**Pass rule (enforced in Python, not only in the prompt):** every score must be ≥ `PASS_THRESHOLD` (default **8**). A model cannot mark `pass_fail=true` if any axis is below the threshold.

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Orchestration | LangGraph `StateGraph`, conditional edges |
| LLM | Google Gemini via `langchain-google-genai` (`ChatGoogleGenerativeAI`) |
| Structured validation | Pydantic v2 |
| Config | `pydantic-settings` + `.env` |
| UI | Streamlit (`st.status`, per-iteration expanders) |

Default model: `gemini-2.0-flash` (override with `GEMINI_MODEL`).

---

## Demo UI (what to show in a review)

The Streamlit app is built so a technical interviewer can **see the loop**, not only the final paragraph:

- input for a sensitive or contested query
- live **active agent** status while the graph streams
- expanders: **Iteration N: Draft**, **Iteration N: Critic Scores & Feedback**
- prominent **Approved response**, or **Governance Review Failed** with the last unapproved draft behind an expander

---

## Quick start

**Requirements:** Python 3.10+, a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GEMINI_API_KEY
streamlit run app.py
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
streamlit run app.py
```

If `.env.example` is missing, create `.env` with:

```env
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.2
MAX_GOVERNANCE_ITERATIONS=3
PASS_THRESHOLD=8
```

`GOOGLE_API_KEY` is accepted if `GEMINI_API_KEY` is empty.

Never commit `.env`. The repo `.gitignore` excludes it.

---

## Repository layout

```
app.py                      Streamlit control plane
governance/
  constitution.py           Enterprise articles + scoring rubric
  schemas.py                CriticEvaluation Pydantic model
  state.py                  GovernanceState TypedDict
  config.py                 Environment-backed settings
  llm.py                    Gemini client + structured critic
  agents.py                 Researcher, Critic, Refiner, terminals
  graph.py                  StateGraph compile + routing
requirements.txt
```

---

## Design choices worth discussing in an interview

1. **Constitution in-prompt + scores in-schema.** Policy is readable; routing is typed.
2. **Fail-closed circuit breaker.** Unaligned text is not silently promoted.
3. **Audit trail in graph state.** Governance is observable, not a black-box “safety filter.”
4. **Code-side threshold.** Prompted `pass_fail` is intersected with numeric floors.

---

## Disclaimer

This is a **research / portfolio demonstration** of agentic governance patterns. It is **not** a certified compliance, legal, or safety product. Do not use it as the sole control for high-stakes production systems. Human review still applies.

The constitution instructs agents to refuse operational assistance for criminal, violent, or other high-harm requests; that is policy text plus model behavior, not a guarantee.

---

## License

Add a `LICENSE` file before you publish (for example MIT) if you want others to reuse the code. Until then, assume all rights reserved by the author.
