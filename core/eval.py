from typing import List, Dict, Any
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import re

RUBRIC = {
    "Clarity": 30,
    "Completeness": 25,
    "Constraint coverage": 20,
    "Testability": 15,
    "Safety": 10
}

def evaluate_candidates(candidates: List[dict], judge_model: str, openai_api_key: str) -> List[dict]:
    # Set deterministic LLM config
    Settings.llm = OpenAI(model=judge_model, api_key=openai_api_key, temperature=0.2, max_tokens=256)
    results = []
    for c in candidates:
        system_prompt = _judge_prompt(c['prompt'])
        res = Settings.llm.complete(system_prompt)
        scores = _parse_llm_judge_response(res.text)
        # Weighted overall
        overall = int(
            sum(scores.get(k,0)*w for k,w in RUBRIC.items())/sum(RUBRIC.values())
        )
        scores['Overall'] = overall
        results.append(scores)
    return results

def _judge_prompt(prompt: str) -> str:
    return (
        "You are a rigorous prompt evaluator. For the following prompt, score 1-5 each for:\n"
        "Clarity, Completeness, Constraint coverage, Testability, Safety.\n"
        "Provide JSON output like {\"Clarity\":X,...}.\n"
        f"PROMPT:\n{prompt}\n"
        "Now, judge:"
    )

def _parse_llm_judge_response(resp: str) -> Dict[str, int]:
    import json
    m = re.search(r'{.*}', resp, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            return {k: int(j[k]) for k in RUBRIC if k in j}
        except Exception:
            pass
    # fallback: try to parse lines
    out = {}
    for k in RUBRIC:
        p = re.search(fr"{k}:\s*([1-5])", resp)
        if p:
            out[k] = int(p.group(1))
    # default 3 if missing
    return {k: out.get(k, 3) for k in RUBRIC}

def calc_heuristics(prompt: str) -> Dict[str, Any]:
    # Length, Flesch, keywords, booleans
    toks = len(prompt.split())
    flesch = flesch_reading_ease(prompt)
    has_role = "You are " in prompt
    has_constraints = "must" in prompt or "limit" in prompt
    pii = _screen_pii(prompt)
    spec_coverage = _spec_coverage(prompt)
    return dict(
        Length=toks,
        Flesch=flesch,
        HasRole=has_role,
        HasConstraints=has_constraints,
        SpecCoverage=spec_coverage,
        PII_Flag=pii
    )

def flesch_reading_ease(text: str) -> float:
    import re
    # Flesch Reading Ease: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    sentences = max(text.count('.'), 1)
    words = max(len(text.split()), 1)
    syllables = sum(len(re.findall(r'[aeiouy]+', w.lower())) for w in text.split())
    return round(206.835 - 1.015*(words/sentences) - 84.6*(syllables/words), 1)

def _screen_pii(prompt: str) -> bool:
    # Naive: detect name, address, phone, SSN, patient, etc.
    pii_terms = ["ssn", "passport", "address", "phone", "email", "patient", "dob", "mrn", "social security", "confidential"]
    return any(t in prompt.lower() for t in pii_terms)

def _spec_coverage(prompt: str) -> float:
    # Dummy: checks for keywords "audience", "format", "length", "role"
    required = ["audience", "format", "length", "role"]
    found = sum(1 for k in required if k in prompt.lower())
    return round(100.0 * found / len(required), 1)
