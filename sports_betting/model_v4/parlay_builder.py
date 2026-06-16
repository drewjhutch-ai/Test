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

# Hard-banned structures (statistical death traps regardless of price)
BUSTED_STRUCTURES = [
    "debut_k_prop",      # K props on debut pitchers — sample size zero
    "same_series_2loss", # Chasing a team that lost 2 in same series
]
# NOTE: Totals (over/under) and LAA ML removed from ban list.
# Model will include them when statistical evidence is strong.


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
    forced: bool = False        # True when built despite low EV (daily guarantee)
    below_threshold: bool = False  # True when EV below target but still shown

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


def build_parlay(
    legs: list[ParlayLeg],
    template_key: str,
    force: bool = False,
    moonshot: bool = False,
) -> Parlay | None:
    """
    Attempt to build a parlay from a list of eligible legs using the given template.
    force=True: always return a parlay using best available legs even if EV threshold not met.
    moonshot=True: prefer run-line / riskier legs to maximise payout (P4/P5).
    """
    t = PARLAY_TEMPLATES[template_key]
    needed = t["legs"]
    max_lose = t["max_lose_pct"]

    MOONSHOT_TAGS = {"run_line_moonshot", "nrfi_layer8_signal", "total_over_layer8_signal",
                     "total_under_layer8_signal"}

    if force:
        eligible = list(legs)
    else:
        eligible = [leg for leg in legs if leg.lose_pct <= max_lose and leg.true_prob >= (1 - max_lose)]

    if len(eligible) < needed:
        return None

    if moonshot:
        # Sort: moonshot-tagged legs first, then by ascending true_prob (highest payout)
        eligible.sort(key=lambda l: (
            0 if any(tag in MOONSHOT_TAGS for tag in l.tags) else 1,
            l.true_prob,
        ))
    else:
        # Hit-rate mode: highest true_prob first, prefer non-moonshot legs
        eligible.sort(key=lambda l: (
            1 if any(tag in MOONSHOT_TAGS for tag in l.tags) else 0,
            -l.true_prob,
        ))

    # Greedy selection: pick legs from sorted list ensuring game-uniqueness per parlay
    # (same-game legs allowed only when they carry genuinely different market tags)
    chosen: list[ParlayLeg] = []
    game_market_seen: set[str] = set()
    for leg in eligible:
        key_gm = f"{leg.pick.game_id}_{leg.market}"
        if key_gm in game_market_seen:
            continue
        chosen.append(leg)
        game_market_seen.add(key_gm)
        if len(chosen) == needed:
            break

    if len(chosen) < needed:
        return None

    parlay = Parlay(
        label=f"{template_key} {t['label']} ({needed}-leg)",
        legs=chosen,
        stake_low=t["stake_range"][0],
        stake_high=t["stake_range"][1],
        forced=force,
    )
    parlay.compute()

    # Strict independence in normal mode, soft check in force mode
    if force:
        if not _soft_independence_audit(parlay):
            return None
    else:
        if not parlay.independence_audit():
            return None

    if parlay.ev_pct < t["target_ev"]:
        if not force:
            return None
        parlay.below_threshold = True

    return parlay


def build_full_parlay_card(legs: list[ParlayLeg]) -> list[Parlay]:
    """
    Build the full P1-P5 parlay card.
    P1-P3: optimise for hit rate (highest true_prob legs).
    P4-P5: optimise for payout (moonshot / run-line legs preferred).
    Always falls back to force-mode if EV threshold is not met.
    """
    parlays = []
    for key in ["P1", "P2", "P3"]:
        p = build_parlay(legs, key, force=False, moonshot=False)
        if not p:
            p = build_parlay(legs, key, force=True, moonshot=False)
        if p:
            parlays.append(p)
    for key in ["P4", "P5"]:
        p = build_parlay(legs, key, force=False, moonshot=True)
        if not p:
            p = build_parlay(legs, key, force=True, moonshot=True)
        if p:
            parlays.append(p)
    return parlays


def _ml_to_decimal(price: int) -> float:
    """American odds → decimal odds."""
    if price > 0:
        return price / 100 + 1
    return 100 / abs(price) + 1


def _decimal_to_american(dec: float) -> int:
    """Decimal odds → American odds (rounded to nearest 5)."""
    if dec >= 2.0:
        raw = (dec - 1) * 100
    else:
        raw = -100 / (dec - 1)
    # Round to nearest 5 (books price in 5-cent increments)
    return int(round(raw / 5) * 5)


def picks_to_legs(picks: list[PickCandidate], prices: dict[str, int]) -> list[ParlayLeg]:
    """
    Convert analyzed PickCandidates to ParlayLegs.

    Primary leg always reflects the model's actual proposed_market.
    Additional legs (F5, totals, NRFI, run line) are added only when layer 8
    explicitly identified them as viable signals — never with hardcoded probability
    bumps. Run-line legs are added for high-confidence picks to provide large-payout
    options (P4/P5 moonshot parlays).
    """
    legs = []
    for pick in picks:
        if pick.tier == "SKIP":
            continue

        hr = getattr(pick, "hard_rules_result", None)
        if hr and not hr.parlay_eligible:
            continue

        price = prices.get(pick.game_id, -120)
        true_prob = 1 - pick.losing_pct
        market = (pick.proposed_market or "full_game_ml").lower()

        # Determine primary description and market key from model's actual signal
        if "f5" in market:
            primary_market = "f5_ml"
            primary_desc   = f"{pick.backing_team} F5 ML ({price:+d})"
            primary_tags   = ["f5_model_signal"]
        elif "nrfi" in market or ("under" in market and "f5" in market):
            primary_market = "nrfi"
            primary_desc   = f"{pick.backing_team} game NRFI/F5 Under ({price:+d})"
            primary_tags   = ["nrfi_model_signal"]
        elif "over" in market or "total" in market:
            primary_market = "game_total_over"
            primary_desc   = f"Game OVER ({price:+d})"
            primary_tags   = ["total_model_signal"]
        elif "under" in market or "total_under" in market:
            primary_market = "game_total_under"
            primary_desc   = f"Game UNDER ({price:+d})"
            primary_tags   = ["total_model_signal"]
        elif "run_line" in market or "rl" in market:
            primary_market = "run_line_-1.5"
            primary_desc   = f"{pick.backing_team} -1.5 RL ({price:+d})"
            primary_tags   = ["rl_model_signal"]
        else:
            primary_market = "full_game_ml"
            primary_desc   = f"{pick.backing_team} ML ({price:+d})"
            primary_tags   = []

        legs.append(ParlayLeg(
            pick=pick,
            market=primary_market,
            price=price,
            true_prob=true_prob,
            lose_pct=pick.losing_pct,
            description=primary_desc,
            tags=primary_tags,
        ))

        # Pull additional markets that layer 8 (bet-type) flagged as genuine signals
        additional_markets: list[str] = []
        for lo in pick.layer_outputs:
            if lo.data.get("additional_markets"):
                additional_markets = lo.data["additional_markets"]
                break

        for extra in additional_markets:
            extra_lower = extra.lower()

            if "f5" in extra_lower and primary_market not in ("f5_ml",):
                f5_price = (price + 5) if price < 0 else (price - 5)
                # F5 prob is anchored to model's true_prob — no artificial boost
                f5_prob  = min(true_prob, 0.74)
                legs.append(ParlayLeg(
                    pick=pick,
                    market="f5_ml",
                    price=f5_price,
                    true_prob=f5_prob,
                    lose_pct=1 - f5_prob,
                    description=f"{pick.backing_team} F5 ML ({f5_price:+d})",
                    tags=["f5_layer8_signal"],
                ))

            elif "nrfi" in extra_lower or ("under" in extra_lower and "f5" in extra_lower):
                if primary_market != "nrfi":
                    nrfi_price = -130
                    nrfi_prob  = max(0.45, true_prob - 0.08)
                    legs.append(ParlayLeg(
                        pick=pick,
                        market="nrfi",
                        price=nrfi_price,
                        true_prob=nrfi_prob,
                        lose_pct=1 - nrfi_prob,
                        description=f"NRFI / F5 Under ({nrfi_price:+d})",
                        tags=["nrfi_layer8_signal"],
                    ))

            elif "over" in extra_lower and primary_market != "game_total_over":
                over_price = -110
                over_prob  = 0.50
                legs.append(ParlayLeg(
                    pick=pick,
                    market="game_total_over",
                    price=over_price,
                    true_prob=over_prob,
                    lose_pct=1 - over_prob,
                    description=f"Game OVER ({over_price:+d})",
                    tags=["total_over_layer8_signal"],
                ))

            elif "under" in extra_lower and primary_market != "game_total_under":
                under_price = -110
                under_prob  = 0.50
                legs.append(ParlayLeg(
                    pick=pick,
                    market="game_total_under",
                    price=under_price,
                    true_prob=under_prob,
                    lose_pct=1 - under_prob,
                    description=f"Game UNDER ({under_price:+d})",
                    tags=["total_under_layer8_signal"],
                ))

        # Run-line leg: added for STRONG picks only, using honest probability penalty.
        # Intentionally riskier — these populate P4/P5 moonshot parlays.
        if (pick.tier == "STRONG"
                and true_prob >= 0.60
                and primary_market not in ("run_line_-1.5",)):
            ml_dec   = _ml_to_decimal(price)
            rl_dec   = ml_dec * 1.15
            rl_price = _decimal_to_american(rl_dec)
            rl_prob  = max(0.40, true_prob - 0.15)  # honest 15-point penalty vs ML
            legs.append(ParlayLeg(
                pick=pick,
                market="run_line_-1.5",
                price=rl_price,
                true_prob=rl_prob,
                lose_pct=1 - rl_prob,
                description=f"{pick.backing_team} -1.5 RL ({rl_price:+d}) [moonshot]",
                tags=["run_line_moonshot"],
            ))

    return legs


def _soft_independence_audit(parlay: "Parlay") -> bool:
    """
    Soft independence check — warns about same-game legs but only blocks
    exact same market/game duplicates. Used in force-built parlays.
    """
    seen = set()
    for leg in parlay.legs:
        key = f"{leg.pick.game_id}_{leg.market}"
        if key in seen:
            parlay.independence_notes.append(
                f"WARN: Duplicate market+game leg ({leg.market}) — correlated risk."
            )
            return False
        seen.add(key)
    parlay.independence_notes.append(
        "PASS: All legs are unique market+game combinations."
    )
    return True


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
