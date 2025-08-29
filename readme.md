# Prompt Optimizer

A Streamlit + LlamaIndex application for precision prompt engineering, built with a 4-phase "4D" workflow (Deconstruct → Diagnose → Develop → Deliver) and local A/B history. Refer to [Lyra Prompt](https://www.reddit.com/r/ChatGPT/comments/1lnfcnt/after_147_failed_chatgpt_prompts_i_had_a/).


## Features

- **Structured 4-D Workflow:** Deconstruct prompt, diagnose clarity, develop 3 candidates (with strategies), deliver usage guide.
- **LLM-as-Judge + Heuristics:** Built-in scoring for each candidate (clarity, completeness, safety, more).
- **A/B Comparison:** Inline diff of any two prompts (candidates or history).
- **Session History:** Restore, tag, and export any previous session locally (JSON/Markdown).
- **No cloud storage:** All data local except OpenAI API.

## Install & Run

Clone and install requirements:

   ```bash
   git clone ...prompt-optimizer
   cd prompt-optimizer
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate
   pip install -r requirements.txt
```

## Example Walkthrough

**Prompt:**
`Write a product-launch email for our new SaaS app for small businesses. Include a persuasive CTA and keep it under 120 words.`

* Select **Task Type:** Creative
* Optional: Word limit = 120
* Click **Analyze**
* Review Deconstruct & Diagnose
* Click **Generate Suggestions** (see 3 labeled prompts)
* **Evaluate**: get LLM and heuristics scoring
* **A/B Compare**: Compare candidates A & B
* **Export**: Save selected prompt to Markdown/JSON

## Extending

* Agentic critique, multi-doc RAG, prompt template libraries, SQLite: all can be layered in v1.1+
* Error-handling is robust; see app and core for comments.

## Test

Basic schema test:

```bash
pytest tests/test_pipeline.py
```

---

## License

MIT



## Code structure:

prompt-optimizer/
├── app.py
├── core/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── eval.py
│   ├── utils.py
├── config/
│   └── settings.yaml
├── data/
│   ├── .gitkeep
│   └── history.jsonl
├── exports/
│   └── .gitkeep
├── tests/
│   └── test_pipeline.py
├── README.md
├── requirements.txt
├── .env.example

## ToDo
- [ ] Add more prompt templates
- [ ] Add more heuristics
- [ ] Integrate with DSpy and BAML --