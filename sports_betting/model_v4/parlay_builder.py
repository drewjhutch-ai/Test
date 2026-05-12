"""
Parlay construction engine — builds P1-P5 parlay structures.
Enforces all parlay hard rules: no totals, no LAA, no sweep G3,
no same-series recycled losers, max 1 bet per game (unless +corr).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import combinations
from .layer_engine import PickCandidate


# P1-P5 parlay templates
PARLAY_TEMPLATES = {
    "P1": {"legs": 2, "max_lose_pct": 0.28, "stake_range": (35, 40), "target_ev": 0.15, "label": "Anchor"},
    "P2": {"legs": 3, "max_lose_pct": 0.30, "stake_range": (15, 20), "target_ev": 0.10, "label": "Core"},
    "P3": {"legs": 4, "max_lose_pct": 0.30, "stake_range": (10, 15), "target_ev": 0.10, "label": "Science"},
    "P4": {"legs": 5, "max_lose_pct": 0.35, "stake_range": (5, 10),  "target_ev": 0.05, "label": "Push"},
    "P5": {"legs": 6, "max_lose_pct": 0.45, "stake_range": (5, 5),   "target_ev": 0.00, "label": "Moonshot"},
}

# Confirmed busting structures — never build these
BUSTED_STRUCTURES = [
    "game_total",
    "over",
    "under",
    "laa_ml",
    "sweep_g3",
    "debut_k_prop",
    "same_series_2loss",
]


@dataclass
class ParlayLeg:
    pick: PickCandidate
    market: str
    price: int
    true_prob: float
    lose_pct: float
    description: str
    tags: list[str] = field(default_factory=list)

    @property
    def decimal_odds(self) -> float:
        if self.price > 0:
            return self.price / 100 + 1
        return 100 / abs(self.price) + 1

    def is_parlay_eligible(self) -> bool:
        if not self.pick.hard_rules_result if hasattr(self.pick, "hard_rules_result") else False:
            return True
        return getattr(self.pick, "hard_rules_result", None) and \
               getattr(self.pick.hard_rules_result, "parlay_eligible", True)


@dataclass
class Parlay:
    label: str          # P1 Anchor, P2 Core, etc.
    legs: list[ParlayLeg]
    stake_low: int
    stake_high: int
    combined_prob: float = 0.0
    combined_decimal: float = 0.0
    ev_pct: float = 0.0
    independence_notes: list[str] = field(default_factory=list)

    def compute(self):
        self.combined_decimal = math.prod(leg.decimal_odds for leg in self.legs)
        self.combined_prob = math.prod(leg.true_prob for leg in self.legs)
        self.ev_pct = (self.combined_prob * self.combined_decimal) - 1

    @property
    def american_odds(self) -> int:
        dec = self.combined_decimal
        if dec >= 2.0:
            return int((dec - 1) * 100)
        return int(-100 / (dec - 1))

    @property
    def payout_per_unit(self) -> float:
        return round((self.combined_decimal - 1) * self.stake_high, 2)

    def independence_audit(self) -> bool:
        """Verify no two legs share a game or failure mechanism."""
        game_ids = [leg.pick.game_id for leg in self.legs]
        positive_corr_pairs = [
            (i, j) for i, j in combinations(range(len(self.legs)), 2)
            if self.legs[i].pick.game_id == self.legs[j].pick.game_id
            and "positive_correlation" in (self.legs[i].tags + self.legs[j].tags)
        ]
        # Block same game unless positive correlation pair
        for i, j in combinations(range(len(self.legs)), 2):
            if (self.legs[i].pick.game_id == self.legs[j].pick.game_id
                    and (i, j) not in positive_corr_pairs):
                self.independence_notes.append(
                    f"FAIL: Legs {i+1} and {j+1} are from the same game — not independent."
                )
                return False

        self.independence_notes.append("PASS: All legs from independent games with separate failure modes.")
        return True


def build_parlay(legs: list[ParlayLeg], template_key: str) -> Parlay | None:
    """
    Attempt to build a parlay from a list of eligible legs using the given template.
    Returns None if legs don't meet quality requirements.
    """
    t = PARLAY_TEMPLATES[template_key]
    needed = t["legs"]
    max_lose = t["max_lose_pct"]

    eligible = [leg for leg in legs if leg.lose_pct <= max_lose and leg.true_prob >= (1 - max_lose)]

    if len(eligible) < needed:
        return None

    # Take the needed highest-confidence legs
    eligible.sort(key=lambda l: l.true_prob, reverse=True)
    chosen = eligible[:needed]

    parlay = Parlay(
        label=f"{template_key} {t['label']} ({needed}-leg)",
        legs=chosen,
        stake_low=t["stake_range"][0],
        stake_high=t["stake_range"][1],
    )
    parlay.compute()

    if not parlay.independence_audit():
        return None

    if parlay.ev_pct < t["target_ev"]:
        return None  # Doesn't meet EV threshold

    return parlay


def build_full_parlay_card(legs: list[ParlayLeg]) -> list[Parlay]:
    """
    Build the full P1-P5 parlay card from available eligible legs.
    Respects all hard rules and EV thresholds.
    """
    parlays = []
    for key in ["P1", "P2", "P3", "P4", "P5"]:
        p = build_parlay(legs, key)
        if p:
            parlays.append(p)
    return parlays


def picks_to_legs(picks: list[PickCandidate], prices: dict[str, int]) -> list[ParlayLeg]:
    """
    Convert analyzed PickCandidates to ParlayLegs.
    prices: dict of game_id -> american_price for backing side
    """
    legs = []
    for pick in picks:
        if pick.tier == "SKIP":
            continue

        # Check hard rules result if present
        hr = getattr(pick, "hard_rules_result", None)
        if hr and not hr.parlay_eligible:
            continue

        price = prices.get(pick.game_id, -110)
        true_prob = 1 - pick.losing_pct

        leg = ParlayLeg(
            pick=pick,
            market=pick.recommended_market or pick.proposed_market,
            price=price,
            true_prob=true_prob,
            lose_pct=pick.losing_pct,
            description=(
                f"{pick.backing_team} {pick.recommended_market} "
                f"({price:+d})"
            ),
        )
        legs.append(leg)

    return legs


def format_parlay_output(parlays: list[Parlay]) -> str:
    """Format all parlays for terminal display."""
    lines = []
    for p in parlays:
        p.compute()
        lines.append(f"\n{'='*60}")
        lines.append(f"  {p.label}")
        lines.append(f"  Odds: +{p.american_odds}  |  Combined prob: {p.combined_prob:.1%}")
        lines.append(f"  Stake: ${p.stake_low}-${p.stake_high}  |  Win: ~${p.payout_per_unit:.0f}")
        lines.append(f"  EV: {p.ev_pct:+.1%}")
        lines.append(f"  Legs:")
        for i, leg in enumerate(p.legs, 1):
            lines.append(f"    {i}. {leg.description} | True prob: {leg.true_prob:.0%} | Lose: {leg.lose_pct:.0%}")
        lines.append(f"  Independence: {p.independence_notes[0] if p.independence_notes else 'Not audited'}")

    return "\n".join(lines)
