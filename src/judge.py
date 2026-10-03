"""Local LLM judge (Qwen2.5-14B-Instruct via vLLM) for answer correctness / evasion, and distractor generation."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

JUDGE_TMPL = """You are grading a short answer to a trivia question.

Question: {q}
Reference answer: {gold}
Accepted variants: {aliases}

Response to grade: "{resp}"

Classify the response by the answer it actually asserts:
CORRECT - it asserts an answer that denotes the same thing as the reference (spelling variants, synonyms, extra or missing details are fine as long as it is clearly the same answer).
INCORRECT - it asserts a different answer (including a partly wrong answer, or an answer that says the reference is not the answer).
EVASION - it asserts no answer at all (refuses, says it does not know, or only asks a question).

Reply with exactly one word: CORRECT, INCORRECT, or EVASION."""

DISTRACTOR_TMPL = """Trivia question: {q}
Correct answer: {gold}

Write one plausible but definitely incorrect answer to this question, of the same type as the correct answer (e.g. another person, place, number or year). It must not be a synonym or variant of the correct answer. Reply with only the incorrect answer, a few words, nothing else."""


def load_judge():
    from vllm import LLM
    return LLM(JUDGE_MODEL, gpu_memory_utilization=0.88, max_model_len=2048, seed=SEED, enforce_eager=True)


def _chat(llm, contents, max_tokens):
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True) for c in contents]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens))
    return [o.outputs[0].text.strip() for o in outs]


def judge(llm, items):
    """items: list of dicts with question, gold, aliases, resp -> list of labels in {CORRECT, INCORRECT, EVASION}."""
    contents = [JUDGE_TMPL.format(q=it["question"], gold=it["gold"], aliases="; ".join(it["aliases"][:10]),
                                  resp=it["resp"][:400]) for it in items]
    raw = _chat(llm, contents, 4)
    labs = []
    for r in raw:
        u = r.upper()
        labs.append("CORRECT" if u.startswith("CORRECT") else "EVASION" if u.startswith("EVASION") else "INCORRECT")
    return labs


def distractors(llm, qs):
    return [d.strip().strip('"').rstrip(".") for d in
            _chat(llm, [DISTRACTOR_TMPL.format(q=q["question"], gold=q["gold"]) for q in qs], 24)]
