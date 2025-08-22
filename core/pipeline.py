import datetime
from typing import Any, Dict, List
from core.utils import estimate_token_count
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 4-D PHASES ---

def run_4d_pipeline(
    prompt: str,
    task_type: str,
    constraints: dict,
    openai_api_key: str
) -> Dict[str, Any]:
    # Deconstruct
    deconstruct = {
        "intent": _extract_intent(prompt),
        "entities": _extract_entities(prompt),
        "context": _extract_context(prompt),
        "output_specs": _extract_output_specs(prompt),
        "constraints": constraints,
        "missing": _detect_missing(prompt, constraints)
    }
    # Diagnose
    diagnose = {
        "issues": _diagnose_issues(prompt, deconstruct)
    }
    # Deliver placeholder (filled later)
    return {
        "deconstruct": deconstruct,
        "diagnose": diagnose,
    }

# --- Deconstruct Subfunctions (Simple rule-based; LLM can be swapped in v1.1) ---
def _extract_intent(prompt: str) -> str:
    return prompt.split("\n")[0][:128].strip() if prompt else ""

def _extract_entities(prompt: str) -> List[str]:
    # Naive: noun-chunk split
    import re
    return list(set([w.lower() for w in re.findall(r'\b([A-Z][a-z]+)\b', prompt)]))

def _extract_context(prompt: str) -> str:
    for key in ["background", "context", "situation"]:
        idx = prompt.lower().find(key)
        if idx != -1:
            return prompt[idx:idx+120]
    return ""

def _extract_output_specs(prompt: str) -> str:
    for key in ["output", "deliverable", "result"]:
        idx = prompt.lower().find(key)
        if idx != -1:
            return prompt[idx:idx+120]
    return ""

def _detect_missing(prompt: str, constraints: dict) -> List[str]:
    miss = []
    if "audience" not in prompt.lower() and not constraints.get("audience"):
        miss.append("Target audience")
    if not constraints.get("word_limit") and "word" not in prompt.lower():
        miss.append("Word limit")
    return miss

def _diagnose_issues(prompt: str, deconstruct: dict) -> List[str]:
    issues = []
    if len(prompt) < 40:
        issues.append("Prompt is too short for clarity.")
    if not deconstruct["entities"]:
        issues.append("No clear entities detected.")
    if not deconstruct["context"]:
        issues.append("No context provided.")
    if "how" not in prompt.lower() and "what" not in prompt.lower():
        issues.append("Prompt may lack a clear action or question.")
    return issues
import re
def extract_candidates(content: str):
    """
    Extracts Candidate, Strategy, Prompt, and Rationale sections from the provided content.
    Returns a list of dicts.
    """
    # Pattern matches each Candidate block (A, B, C, etc.)
    candidate_pattern = re.compile(
        r"(Candidate [A-Z]:\s*Strategy:.*?)(?=Candidate [A-Z]:|$)", re.DOTALL)
    strategy_pattern = re.compile(r"Strategy:\s*(.*?)(?:\n|$)", re.DOTALL)
    prompt_pattern = re.compile(r"Prompt:\s*\"{0,1}(.*?)(?<!\\)\"{0,1}\s*(?:\n|$)", re.DOTALL)
    rationale_pattern = re.compile(r"Rationale:\s*(.*?)(?=\n(?:Candidate [A-Z]:|$)|$)", re.DOTALL)

    candidates = []
    for match in candidate_pattern.finditer(content):
        block = match.group(1)
        # Extract candidate label (A, B, ...)
        candidate_label_match = re.match(r"Candidate ([A-Z]):", block)
        print(candidate_label_match)
        candidate_label = candidate_label_match.group(1) if candidate_label_match else "?"
        # Extract strategy, prompt, rationale
        strategy = strategy_pattern.search(block)
        prompt = prompt_pattern.search(block)
        rationale = rationale_pattern.search(block)
        candidates.append({
            "candidate": candidate_label,
            "strategy": strategy.group(1).strip() if strategy else "",
            "technique": strategy.group(1).strip() if strategy else "",
            "prompt": prompt.group(1).strip() if prompt else "",
            "rationale": rationale.group(1).strip() if rationale else "",
            "token_estimate": estimate_token_count(prompt.group(1).strip()) 
        })
    return candidates

# --- Develop: Use LLM to create candidate prompts ---
def build_candidates(
    deconstruct: dict,
    task_type: str,
    constraints: dict,
    openai_api_key: str
) -> List[dict]:
    # Set up LlamaIndex LLM for generation
    Settings.llm = OpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0.2, max_tokens=600)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)
    system_role = "You are Lyra, a master-level AI prompt engineering specialist."
    task_prompt = _strategy_prompt(deconstruct, task_type, constraints)
    res = Settings.llm.complete(system_role + "\n" + task_prompt)
    print(res.text)
    # LLM response must produce 3 prompts
    # import re
    # # Split into 3 sections by Candidate marker
    # matches = re.findall(r"Candidate [ABC]:\n(.+?)\nPrompt:", res.text, re.DOTALL)
    # rationales = re.findall(r"Rationale: (.+?)(?:\n|$)", res.text)
    # techniques = re.findall(r"Strategy: (.+?)(?:\n|$)", res.text)
    # candidates = []
    # for i in range(3):
    #     c = {
    #         "prompt": matches[i].strip() if i < len(matches) else "",
    #         "rationale": rationales[i].strip() if i < len(rationales) else "",
    #         "technique": techniques[i].strip() if i < len(techniques) else f"Strategy {i+1}",
    #         "token_estimate": estimate_token_count(matches[i]) if i < len(matches) else 0
    #     }
    #     candidates.append(c)
    candidates = extract_candidates(res.text)
    return candidates

def _strategy_prompt(deconstruct, task_type, constraints):
    base = f"""
Given the following intent: "{deconstruct['intent']}"
Entities: {', '.join(deconstruct['entities'])}
Context: {deconstruct['context']}
Output specs: {deconstruct['output_specs']}
Constraints: {constraints}
Task type: {task_type}

Produce 3 optimized prompt candidates as below:
- Label each as "Candidate A", "Candidate B", "Candidate C" without highlighting and numbering.
- For each, specify "Strategy", then show the prompt, then a short "Rationale".
- Max 250 words per candidate.
- Each must apply a different optimization approach based on the task type:
"""
    strat = ""
    if task_type == "Creative":
        strat = """
Candidate A:
Strategy: Multi-perspective framing with explicit tone and audience.
[Prompt goes here]
Rationale: ...
Candidate B:
Strategy: Role assignment plus layered context.
[Prompt goes here]
Rationale: ...
Candidate C:
Strategy: Constraint-driven (word limit/style) + creativity boost.
[Prompt goes here]
Rationale: ...
"""
    elif task_type == "Technical":
        strat = """
Candidate A:
Strategy: Constraint-first with acceptance criteria and I/O schema hints.
[Prompt goes here]
Rationale: ...
Candidate B:
Strategy: Role + explicit input/output format specification.
[Prompt goes here]
Rationale: ...
Candidate C:
Strategy: Stepwise instruction with edge case handling.
[Prompt goes here]
Rationale: ...
"""
    elif task_type == "Educational":
        strat = """
Candidate A:
Strategy: Few-shot outline + rubric + explicit learner level.
[Prompt goes here]
Rationale: ...
Candidate B:
Strategy: Outcome verbs and Bloom's taxonomy mapping.
[Prompt goes here]
Rationale: ...
Candidate C:
Strategy: Scenario-based instructional prompt.
[Prompt goes here]
Rationale: ...
"""
    elif task_type == "Complex":
        strat = """
Candidate A:
Strategy: Checklist format, explicit steps, and risk notes.
[Prompt goes here]
Rationale: ...
Candidate B:
Strategy: Assumptions/constraints up front, then deliverable.
[Prompt goes here]
Rationale: ...
Candidate C:
Strategy: Multi-role with input validation and post-conditions.
[Prompt goes here]
Rationale: ...
"""
    return base + strat

def get_session_snapshot(
    prompt: str,
    deconstruct: dict,
    diagnose: dict,
    candidates: list,
    chosen_idx: int,
    constraints: dict,
    task_type: str
) -> dict:
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return {
        "session_id": generate_session_id(),
        "timestamp": now,
        "prompt": prompt,
        "deconstruct": deconstruct,
        "diagnose": diagnose,
        "candidates": candidates,
        "chosen_idx": chosen_idx,
        "constraints": constraints,
        "task_type": task_type,
        "tags": deconstruct['intent'][:32]
    }

def generate_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")


if __name__ == "__main__":
    generate_str="""Candidate A:
        Strategy: Multi-perspective framing with explicit tone and audience.
        Prompt: "Imagine you're a small business owner eager to streamline operations. Write a concise, engaging product-launch email for our new SaaS app, designed specifically for small businesses. Use a friendly and professional tone to connect with your audience. Highlight key features that solve common pain points and include a persuasive call-to-action encouraging immediate sign-up. Keep the email under 120 words."
        Rationale: This approach frames the task from the perspective of the target audience, ensuring the email resonates with small business owners. By specifying the tone and audience, it guides the writer to craft a message that is both relatable and effective, enhancing engagement and conversion potential.

        Candidate B:
        Strategy: Role assignment plus layered context.
        Prompt: "As a marketing specialist for a SaaS company, your task is to craft a compelling product-launch email for our new app targeting small businesses. Focus on the unique benefits and solutions it offers. Layer in context about the competitive landscape and current market trends to make the email relevant and persuasive. Conclude with a strong call-to-action that prompts immediate engagement. Ensure the email is concise, under 120 words."
        Rationale: Assigning a specific role helps the writer adopt a professional mindset, while the layered context encourages them to consider broader market factors. This strategy ensures the email is not only informative but also strategically positioned within the competitive landscape, enhancing its persuasive power.

        Candidate C:
        Strategy: Constraint-driven (word limit/style) + creativity boost.
        Prompt: "Craft a creative and concise product-launch email for our new SaaS app aimed at small businesses. Emphasize innovation and simplicity, using vivid language to capture attention. Adhere strictly to a 120-word limit while ensuring the message is impactful and includes a compelling call-to-action. Experiment with different styles to find the most engaging approach."
        Rationale: By focusing on constraints like word limit and style, this approach encourages creativity within boundaries. It challenges the writer to think outside the box, using vivid language and innovative styles to create an impactful message that stands out, while still adhering to the specified word count.
        """

    candidates = extract_candidates(generate_str)
    print(candidates)