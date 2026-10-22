"""k-sample attribution aggregator (intake-2332#record, MP-Bench protocol, on local judges).

Each sample is one all-at-once multi-step attribution at temperature > 0 (default k=5 per family at
0.7, across every judge family supplied, two in HIL-17). A step's vote share = samples naming it /
samples drawn (a sample that fails to parse still counts in the denominator, and is reported).
Steps are ranked by vote share. Comparator: ONE temperature-0 pass of the first family, its named
steps all scored 1.0 (so their order is a tie, charged at expected DCG). Baseline: a uniformly random
ranking of all attributable steps, exact expectation plus a seeded Monte Carlo twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.trace.attribution.judge import JudgeSpec, TokenMeter, as_int, extract_json, metered
from src.trace.attribution.metrics import ndcg, random_ndcg_expected, random_ndcg_mc
from src.trace.attribution.prompts import PromptConfig, multi_request
from src.trace.attribution.schema import AgentTrajectory


@dataclass
class AggregatorConfig:
    k: int = 5  # samples per family
    temperature: float = 0.7
    seed_base: int = 0  # request seed = seed_base + sample number (server may ignore)
    ndcg_k: int = 5
    mc_draws: int = 2000
    prompt: PromptConfig = field(default_factory=PromptConfig)


def parse_steps(text: str, valid: set[int]) -> tuple[set[int] | None, int]:
    """(named steps, n_out_of_range). None = the sample did not parse."""
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        return None, 0
    named, bad = set(), 0
    for it in obj["steps"]:
        s = as_int(it.get("step")) if isinstance(it, dict) else as_int(it)
        if s is None:
            continue
        if s in valid:
            named.add(s)
        else:
            bad += 1
    return named, bad


def vote_share(samples: list[set[int] | None]) -> dict[int, float]:
    if not samples:
        return {}
    votes: dict[int, int] = {}
    for s in samples:
        for step in s or ():
            votes[step] = votes.get(step, 0) + 1
    return {step: v / len(samples) for step, v in votes.items()}


@dataclass
class AggregateResult:
    case_id: str
    vote_share: dict[int, float]
    temp0_steps: set[int] | None
    per_family: dict[str, dict]
    tokens_ksample: TokenMeter
    tokens_temp0: TokenMeter
    ndcg: dict[str, float | None]

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "vote_share": {str(k): v for k, v in sorted(self.vote_share.items())},
            "temp0_steps": None if self.temp0_steps is None else sorted(self.temp0_steps),
            "per_family": self.per_family,
            "tokens_ksample": self.tokens_ksample.as_dict(),
            "tokens_temp0": self.tokens_temp0.as_dict(),
            "ndcg": self.ndcg,
        }


def aggregate_case(
    case_id: str,
    traj: AgentTrajectory,
    families: list[JudgeSpec],
    relevance: dict[int, float],
    cfg: AggregatorConfig | None = None,
    reference: str | None = None,
    mc_seed: int = 0,
) -> AggregateResult:
    cfg = cfg or AggregatorConfig()
    if not families:
        raise ValueError("aggregate_case needs at least one judge family")
    valid = set(traj.step_indices())
    k_meter, t0_meter = TokenMeter(), TokenMeter()
    samples: list[set[int] | None] = []
    per_family: dict[str, dict] = {}
    n = 0
    for fam in families:
        call = metered(fam.judge, k_meter)
        parsed = fails = oor = 0
        for _ in range(cfg.k):
            named, bad = parse_steps(
                call(
                    multi_request(traj, cfg.prompt, reference, cfg.temperature, cfg.seed_base + n)
                ).text,
                valid,
            )
            n += 1
            samples.append(named)
            oor += bad
            if named is None:
                fails += 1
            else:
                parsed += 1
        per_family[fam.name] = {
            "family": fam.family,
            "quantised": fam.quantised,
            "samples": cfg.k,
            "parsed": parsed,
            "parse_failures": fails,
            "out_of_range_steps": oor,
        }
    vs = vote_share(samples)
    t0, _ = parse_steps(
        metered(families[0].judge, t0_meter)(
            multi_request(traj, cfg.prompt, reference, 0.0, cfg.seed_base)
        ).text,
        valid,
    )
    steps = traj.step_indices()
    res: dict[str, float | None] = {}
    for kind in ("exp", "linear"):
        res[f"ksample_{kind}"] = ndcg(vs, relevance, cfg.ndcg_k, kind)
        res[f"temp0_{kind}"] = ndcg({s: 1.0 for s in (t0 or ())}, relevance, cfg.ndcg_k, kind)
        res[f"random_expected_{kind}"] = random_ndcg_expected(steps, relevance, cfg.ndcg_k, kind)
        res[f"random_mc_{kind}"] = random_ndcg_mc(
            steps, relevance, cfg.ndcg_k, kind, cfg.mc_draws, mc_seed
        )
    return AggregateResult(case_id, vs, t0, per_family, k_meter, t0_meter, res)
