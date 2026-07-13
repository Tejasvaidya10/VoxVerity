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
    pitch_variance: float             # Hz^2 over voiced frames (librosa pyin)
    spectral_flatness: float          # 0..1, higher = noisier/synthetic
    spectral_rolloff_95_hz: float     # 95% energy roll-off frequency
    unvoiced_energy_ratio: float      # energy share in unvoiced frames (breaths/silences)
    high_band_energy_fraction: float  # energy share above 4 kHz
    high_attention_time_ranges: list  # empty while LinearHead is the classifier


RATIONALE_SYSTEM_PROMPT = """You are a forensic audio analyst assistant. Given
detector evidence for a spoken audio clip, write a 2-3 sentence rationale
explaining why the clip was flagged as likely synthetic or likely genuine.
Evidence fields: spoof_probability (detector output), pitch_variance (Hz^2,
natural speech wobbles; near-zero suggests synthesis), spectral_flatness
(0-1, higher = noise-like), spectral_rolloff_95_hz (where 95% of energy sits),
unvoiced_energy_ratio (energy in breaths/silences, where vocoder artifacts
hide), high_band_energy_fraction (>4 kHz share; vocoders often leave a shelf
or smear there).
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
        spectral_rolloff_95_hz=6200.0,
        unvoiced_energy_ratio=0.31,
        high_band_energy_fraction=0.18,
        high_attention_time_ranges=[[2.1, 2.6]],
    )
    rationale = generate_rationale(client, example_evidence)
    print("Rationale:", rationale)
    judged = score_rationale(client, example_evidence, rationale)
    print("Judge score:", judged)
