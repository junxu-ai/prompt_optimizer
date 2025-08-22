import os
import sys
import streamlit as st
import datetime
import json
from core.pipeline import run_4d_pipeline, build_candidates, get_session_snapshot
from core.eval import evaluate_candidates, calc_heuristics
from core.utils import (estimate_token_count, flesch_reading_ease, inline_diff,
                        load_history, save_session, export_prompt, find_session_by_id, generate_session_id)
from config.settings import get_config, get_openai_api_key

import sys
from pathlib import Path

# Add the parent directory of the current file to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 

# --- Initialization ---
st.set_page_config(page_title="Prompt Optimizer", layout="wide")
config = get_config()
os.environ["OPENAI_API_KEY"] = get_openai_api_key()
HISTORY_PATH = "data/history.jsonl"
EXPORTS_DIR = "exports"

# --- Welcome Banner ---
st.markdown(
    """
    <div style="padding:1rem;background:#20233a;border-radius:1rem;margin-bottom:1rem;">
    <b>Hello! I'm Lyra, your AI prompt optimizer.</b> I transform vague requests into precise, effective prompts for all AI platforms.<br>
    <i>Enter your rough prompt, pick a task type, set any constraints, and Lyra will analyze, suggest, and help you compare versions. No data ever leaves your device except via OpenAI API.</i>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- UI State Management ---
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = generate_session_id()
if 'history' not in st.session_state:
    st.session_state['history'] = load_history(HISTORY_PATH)

# --- Inputs ---
with st.form(key='main_form'):
    prompt = st.text_area("Original Prompt", height=180, key="prompt", value=st.session_state.get('last_prompt', ""))
    task_type = st.selectbox("Task Type", ["Creative", "Technical", "Educational", "Complex"])
    col1, col2, col3 = st.columns(3)
    with col1:
        word_limit = st.number_input("Word Limit (optional)", min_value=0, max_value=3000, step=10, value=0)
    with col2:
        tone = st.text_input("Tone (optional)", value="")
    with col3:
        style = st.text_input("Style/Format (optional)", value="")
    audience = st.text_input("Target Audience (optional)", value="")
    priority = st.selectbox("Priority (optional)", ["None", "Latency", "Cost"], index=0)
    submit = st.form_submit_button("Analyze")

# -- Preload session state if restoring --
if submit or not st.session_state.get('pipeline'):
    # Compose constraint dictionary
    constraints = {
        "word_limit": word_limit if word_limit > 0 else None,
        "tone": tone or None,
        "style": style or None,
        "audience": audience or None,
        "priority": None if priority == "None" else priority,
    }
    # Run 4D pipeline
    try:
        pipeline_out = run_4d_pipeline(
            prompt,
            task_type=task_type,
            constraints=constraints,
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        st.session_state['pipeline'] = pipeline_out
        st.session_state['constraints'] = constraints
        st.session_state['last_prompt'] = prompt
    except Exception as e:
        st.error(f"Failed to analyze prompt: {str(e)}")
        st.stop()

# --- Panels ---
if 'pipeline' in st.session_state:
    out = st.session_state['pipeline']

    with st.expander("1. Deconstruct", expanded=True):
        st.markdown(f"**Intent:** {out['deconstruct']['intent']}")
        st.markdown(f"**Entities:** {', '.join(out['deconstruct']['entities']) if out['deconstruct']['entities'] else 'N/A'}")
        st.markdown(f"**Context:** {out['deconstruct']['context'] or 'N/A'}")
        st.markdown(f"**Output Specs:** {out['deconstruct']['output_specs'] or 'N/A'}")
        st.markdown(f"**Constraints:** {json.dumps(out['deconstruct']['constraints'], indent=1)}")
        st.markdown(f"**Missing Info:** {', '.join(out['deconstruct']['missing']) or 'None'}")

    with st.expander("2. Diagnose", expanded=True):
        st.markdown("**Clarity Gaps, Ambiguity & Structure Needs:**")
        for item in out['diagnose']['issues']:
            st.markdown(f"- {item}")

    with st.expander("3. Develop", expanded=True):
        # On button, call LLM to build candidates
        if st.button("Generate Suggestions"):
            try:
                candidates = build_candidates(
                    out['deconstruct'],
                    task_type=task_type,
                    constraints=st.session_state['constraints'],
                    openai_api_key=os.environ["OPENAI_API_KEY"]
                )
                print(f"Generated {len(candidates)} candidates \n ", candidates)
                st.session_state['candidates'] = candidates
            except Exception as e:
                st.error(f"Failed to generate candidates: {str(e)}")
        if 'candidates' in st.session_state:
            for idx, c in enumerate(st.session_state['candidates'], 1):
                st.markdown(f"**Candidate {chr(64+idx)}** *(Strategy: {c['technique']}, Est. Tokens: {c['token_estimate']})*")
                st.code(c['prompt'], language='markdown')
                st.markdown(f"*Rationale:* {c['rationale']}")

    with st.expander("4. Deliver", expanded=True):
        # Candidate selection, usage, risk notes
        if 'candidates' in st.session_state:
            chosen = st.radio("Select best candidate", [f"Candidate {chr(65+i)}" for i in range(len(st.session_state['candidates']))], horizontal=True)
            chosen_idx = ord(chosen[-1]) - 65
            best = st.session_state['candidates'][chosen_idx]
            st.markdown("**Usage Guide**")
            st.markdown(f"- Target AI: {task_type} scenario\n- Estimated cost: {best['token_estimate']} tokens\n- Apply as: input for LLM, API, or workflow.")
            st.markdown("**Risk Notes**: Avoid sharing sensitive/regulated data; always review LLM outputs for critical use cases.")

            # Save/export section
            if st.button("Evaluate"):
                evals = evaluate_candidates(
                    st.session_state['candidates'],
                    judge_model="gpt-4o",
                    openai_api_key=os.environ["OPENAI_API_KEY"]
                )
                heuristics = [calc_heuristics(c['prompt']) for c in st.session_state['candidates']]
                for i, (ev, h) in enumerate(zip(evals, heuristics), 1):
                    st.markdown(f"**Candidate {chr(64+i)}**")
                    st.json({"LLM Judge": ev, "Heuristics": h})

            if st.button("Export"):
                session_meta = get_session_snapshot(
                    prompt=prompt,
                    deconstruct=out['deconstruct'],
                    diagnose=out['diagnose'],
                    candidates=st.session_state['candidates'],
                    chosen_idx=chosen_idx,
                    constraints=st.session_state['constraints'],
                    task_type=task_type
                )
                md_path, json_path = export_prompt(session_meta, EXPORTS_DIR)
                st.success(f"Exported to {md_path} and {json_path}")
                save_session(session_meta, HISTORY_PATH)
                # reload history
                st.session_state['history'] = load_history(HISTORY_PATH)

# --- A/B Compare Section ---
with st.expander("A/B Compare", expanded=False):
    if 'candidates' in st.session_state:
        # Current session only for v1 (historical in advanced)
        a = st.selectbox("Prompt A", options=[f"Candidate {chr(65+i)}" for i in range(len(st.session_state['candidates']))])
        b = st.selectbox("Prompt B", options=[f"Candidate {chr(65+i)}" for i in range(len(st.session_state['candidates'])) if f"Candidate {chr(65+i)}" != a])
        idx_a, idx_b = ord(a[-1]) - 65, ord(b[-1]) - 65
        diff = inline_diff(st.session_state['candidates'][idx_a]['prompt'], st.session_state['candidates'][idx_b]['prompt'])
        st.markdown(f"**A/B Diff between {a} and {b}:**")
        st.code(diff, language="diff")
    else:
        st.info("Generate suggestions to enable A/B compare.")

# --- History & Restore ---
with st.expander("History", expanded=False):
    history = st.session_state['history']
    if history:
        for h in reversed(history):
            tag = h.get('tags', "No tag")
            ts = h.get('timestamp', "")
            st.markdown(f"**{ts}** | **{tag}**")
            st.markdown(f"- **Prompt:** {h.get('prompt','')[:60]}...")
            if st.button(f"Restore session {ts}", key=ts):
                st.session_state['pipeline'] = {
                    'deconstruct': h['deconstruct'],
                    'diagnose': h['diagnose']
                }
                st.session_state['candidates'] = h['candidates']
                st.session_state['constraints'] = h.get('constraints', {})
                st.session_state['last_prompt'] = h.get('prompt', "")
                st.success(f"Restored session from {ts}")
    else:
        st.info("No history found.")

st.caption("© Lyra Prompt Optimizer v1 | Local only | OpenAI API required.")
