"""
Given a detector's score and any available evidence (e.g. attention weights
if using MHFAHead, or simple acoustic descriptors), ask an LLM to produce a
plain-language rationale for the flagged classification. This is the piece
that reuses your Truist RAG-eval / multi-agent experience: the LLM here acts
as an explanation-generation agent, and a second call (see score_rationale)
acts as a judge scoring whether the explanation is faithful to the evidence.
"""

import json
from dataclasses import dataclass, asdict

import anthropic


@dataclass
class DetectionEvidence:
    file_id: str
    spoof_probability: float
    pitch_variance: float          # acoustic descriptor, compute upstream with librosa
    spectral_flatness: float       # higher = more noise-like / synthetic
    high_attention_time_ranges: list  # e.g. [[2.1, 2.6]] seconds, from MHFAHead weights


RATIONALE_SYSTEM_PROMPT = """You are a forensic audio analyst assistant. Given
detector evidence for a spoken audio clip, write a 2-3 sentence rationale
explaining why the clip was flagged as likely synthetic or likely genuine.
Ground every claim strictly in the numeric evidence provided. Do not invent
acoustic details that are not in the evidence. If the evidence is weak or
ambiguous, say so explicitly rather than overstating confidence."""


def generate_rationale(client: anthropic.Anthropic, evidence: DetectionEvidence,
                        model: str = "claude-sonnet-4-6", max_tokens: int = 300) -> str:
    prompt = f"Detector evidence:\n{json.dumps(asdict(evidence), indent=2)}\n\nWrite the rationale."
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=RATIONALE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


JUDGE_SYSTEM_PROMPT = """You are grading whether a rationale for an audio
deepfake detection is faithful to the evidence it was given. Score 1-5:
5 = every claim is directly supported by the evidence, no invented details.
3 = mostly supported but with some vague or unsupported claims.
1 = rationale contradicts the evidence or invents details not present.
Respond with only a JSON object: {"score": <int>, "reason": "<one sentence>"}"""


def score_rationale(client: anthropic.Anthropic, evidence: DetectionEvidence,
                     rationale: str, model: str = "claude-sonnet-4-6") -> dict:
    prompt = (
        f"Evidence:\n{json.dumps(asdict(evidence), indent=2)}\n\n"
        f"Rationale to grade:\n{rationale}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=150,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"unparseable judge output: {text}"}


if __name__ == "__main__":
    # example usage — replace with real evidence pulled from your detector run
    client = anthropic.Anthropic()
    example_evidence = DetectionEvidence(
        file_id="p2v_00042_3",
        spoof_probability=0.87,
        pitch_variance=0.9,
        spectral_flatness=0.72,
        high_attention_time_ranges=[[2.1, 2.6]],
    )
    rationale = generate_rationale(client, example_evidence)
    print("Rationale:", rationale)
    judged = score_rationale(client, example_evidence, rationale)
    print("Judge score:", judged)
