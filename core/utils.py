import os
import json
import difflib
import tiktoken
from typing import List, Dict

def estimate_token_count(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.5)

def flesch_reading_ease(text: str) -> float:
    import re
    sentences = max(text.count('.'), 1)
    words = max(len(text.split()), 1)
    syllables = sum(len(re.findall(r'[aeiouy]+', w.lower())) for w in text.split())
    return round(206.835 - 1.015*(words/sentences) - 84.6*(syllables/words), 1)

def inline_diff(a: str, b: str) -> str:
    d = difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='')
    return '\n'.join(list(d))

def load_history(history_path: str) -> List[Dict]:
    if not os.path.exists(history_path):
        return []
    with open(history_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_session(session: dict, history_path: str):
    with open(history_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(session, ensure_ascii=False) + '\n')

def export_prompt(session: dict, exports_dir: str):
    os.makedirs(exports_dir, exist_ok=True)
    ts = session.get('timestamp', '').replace(':','').replace(' ','_')
    md_path = os.path.join(exports_dir, f"{ts}_optimized.md")
    json_path = os.path.join(exports_dir, f"{ts}_optimized.json")
    # Write markdown (prompt + scores + metadata)
    c = session['candidates'][session['chosen_idx']]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Optimized Prompt\n\n{c['prompt']}\n\n## Metadata\n")
        json.dump(session, f, ensure_ascii=False, indent=2)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return md_path, json_path

def find_session_by_id(history: List[Dict], sid: str) -> dict:
    for s in history:
        if s.get('session_id') == sid:
            return s
    return {}

def generate_session_id() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
