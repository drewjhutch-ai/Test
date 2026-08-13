"""
Parlay construction engine — rebuilt around real, market-anchored probabilities.

The old builder manufactured legs with hardcoded probabilities (OVER/UNDER = 0.50,
NRFI = ml - 0.08, run-line = ml - 0.15), priced correlated same-game legs as if
independent, and force-built a full P1-P5 card every day. All three are classic
ways to bleed money (see The Logic of Sports Betting; Pinnacle/Unabated on the
correlation tax; NJ parlay hold of 19-24%).

This version follows the research:
  * A parlay is +EV **iff every leg is individually +EV** — it multiplies edge,
    it does not create it. So legs come only from qualifying straight picks that
    already cleared the market edge gate.
  * Legs are ML bets on different games (max 1 pick/game is enforced upstream),
    so they are independent and the joint probability is the product of the
    marginals. No fabricated sub-market probabilities, no correlation trap.
  * We never force a card. Thin slate ⇒ few or zero parlays. That is correct.
  * If two legs ever DID share a game, compute() applies the correlation
    formula P(A∩B)=P(A)P(B)+ρ·√[...] instead of the naive product.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import combinations
from .layer_engine import PickCandidate

# Same-game correlation coefficient used only as a defensive fallback if two
# legs from the same game ever reach a parlay. Cross-game legs use ρ = 0.
DEFAULT_SAME_GAME_RHO = 0.35

# Parlay ladder: build one parlay at each leg count so the user can pick their
# own risk/payoff rung. Legs come from the model's +EV picks (underdogs = big
# payoffs), so every rung is a longshot with real edge on each leg — not -EV
# chalk. Higher rungs pay more and hit less; that's the explicit tradeoff.
LADDER_LEG_COUNTS = [2, 3, 4, 5, 6]
MAX_LEGS = 6


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

    @property
    def ev(self) -> float:
        """EV per unit at this leg's price and true probability."""
        return self.true_prob * (self.decimal_odds - 1) - (1 - self.true_prob)


@dataclass
class Parlay:
    label: str
    legs: list[ParlayLeg]
    stake_low: int
    stake_high: int
    combined_prob: float = 0.0
    combined_decimal: float = 0.0
    ev_pct: float = 0.0
    independence_notes: list[str] = field(default_factory=list)

    def compute(self):
        self.combined_decimal = math.prod(leg.decimal_odds for leg in self.legs)
        self.combined_prob = _joint_probability(self.legs)
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


def _joint_probability(legs: list[ParlayLeg]) -> float:
    """
    Joint probability of all legs hitting. Independent (cross-game) legs
    multiply; any pair sharing a game is combined with the correlation formula
    so we never price correlated legs as if independent.
    """
    if not legs:
        return 0.0
    # Start from the independent product...
    prob = math.prod(leg.true_prob for leg in legs)
    # ...then correct any same-game pair. In practice legs are cross-game so this
    # loop is a no-op, but it keeps the math honest if that ever changes.
    for i, j in combinations(range(len(legs)), 2):
        if legs[i].pick.game_id == legs[j].pick.game_id:
            a, b = legs[i].true_prob, legs[j].true_prob
            indep = a * b
            corr = a * b + DEFAULT_SAME_GAME_RHO * math.sqrt(
                a * (1 - a) * b * (1 - b)
            )
            if indep > 0:
                prob *= corr / indep
    return round(min(max(prob, 0.0), 1.0), 4)


def picks_to_legs(picks: list[PickCandidate], prices: dict[str, int]) -> list[ParlayLeg]:
    """
    Convert qualifying straight picks into ML parlay legs using their REAL
    market-anchored blended probability and REAL price. No fabricated markets.

    Only +EV legs are returned (they should all be +EV since they cleared the
    edge gate, but we re-check defensively).
    """
    legs: list[ParlayLeg] = []
    for pick in picks:
        if pick.tier == "SKIP":
            continue

        hr = getattr(pick, "hard_rules_result", None)
        if hr and not getattr(hr, "parlay_eligible", True):
            continue

        market = getattr(pick, "market", None)
        # True probability: prefer the blended, market-anchored number; fall back
        # to (1 - losing_pct) only if market data is somehow absent.
        if market and market.get("p_blend"):
            true_prob = float(market["p_blend"])
        else:
            true_prob = 1 - pick.losing_pct

        price = getattr(pick, "backing_price", None)
        if price is None:
            price = prices.get(pick.game_id, -120)

        leg = ParlayLeg(
            pick=pick,
            market="full_game_ml",
            price=int(price),
            true_prob=round(true_prob, 4),
            lose_pct=round(1 - true_prob, 4),
            description=f"{pick.backing_team} ML ({int(price):+d})",
            tags=["ml_market_anchored"],
        )
        if leg.ev > 0:
            legs.append(leg)
    return legs


def _select_unique_game_legs(sorted_legs: list[ParlayLeg], n: int) -> list[ParlayLeg]:
    """Greedily take n legs, one per game, from an already-sorted list."""
    chosen: list[ParlayLeg] = []
    seen_games: set[str] = set()
    for leg in sorted_legs:
        if leg.pick.game_id in seen_games:
            continue
        chosen.append(leg)
        seen_games.add(leg.pick.game_id)
        if len(chosen) == n:
            break
    return chosen


def hit_expectation(prob: float) -> str:
    """Plain-language honest read on how often a parlay actually hits."""
    if prob <= 0:
        return "almost never hits"
    if prob >= 0.50:
        return f"hits ~{prob:.0%} — better than a coin flip"
    one_in = max(2, round(1 / prob))
    return f"hits ~{prob:.0%} (about 1 in {one_in}) — parlays lose most of the time"


def build_full_parlay_card(legs: list[ParlayLeg]) -> list[Parlay]:
    """
    Build a 2-to-6 leg parlay ladder from the model's +EV picks. Each rung uses
    the N strongest available legs (best hit-rate for that leg count); payouts
    escalate and hit-rate drops as you climb. Every leg is individually +EV
    (underdog value), so the parlays are longshots with real edge — not -EV
    chalk. Nothing is forced: rungs beyond the available leg count are skipped.
    """
    eligible = [leg for leg in legs if leg.ev > 0]
    if len(eligible) < 2:
        return []

    by_prob = sorted(eligible, key=lambda l: l.true_prob, reverse=True)  # best legs first
    max_available = len(by_prob)

    parlays: list[Parlay] = []
    for n in LADDER_LEG_COUNTS:
        if n > max_available:
            break  # not enough distinct legs for this rung (or any higher one)
        chosen = _select_unique_game_legs(by_prob, n)
        if len(chosen) < n:
            break

        stake_high = {2: 20, 3: 15, 4: 10, 5: 5, 6: 5}.get(n, 5)
        parlay = Parlay(
            label=f"{n}-Leg Parlay",
            legs=chosen,
            stake_low=max(5, stake_high // 2),
            stake_high=stake_high,
        )
        parlay.compute()
        parlay.independence_notes.append(hit_expectation(parlay.combined_prob))
        parlays.append(parlay)

    return parlays  # 2-leg first (best hit-rate) → 6-leg (biggest payout)


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
            lines.append(
                f"    {i}. {leg.description} | True prob: {leg.true_prob:.0%} "
                f"| EV: {leg.ev:+.0%}"
            )
        lines.append(f"  {p.independence_notes[0] if p.independence_notes else ''}")
    return "\n".join(lines)
