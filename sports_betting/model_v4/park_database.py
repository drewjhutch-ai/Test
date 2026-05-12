"""
Enhanced park factor database with LHB/RHB splits.
Replaces the basic park factors in config.py.
"""

PARK_DATABASE: dict[str, dict] = {
    "Great American Ball Park (CIN)": {
        "type": "outdoor", "hr_factor": 1.28,
        "lhb_hr_factor": 1.35, "rhb_hr_factor": 1.22,
        "run_factor": 1.15,
        "notes": "#1 LHB HR park. 8th shallowest RF. YRFI lean.",
    },
    "Coors Field (COL)": {
        "type": "outdoor", "hr_factor": 1.40,
        "run_factor": 1.38, "altitude_ft": 5200,
        "notes": "#1 run-scoring park. Wind direction critical.",
    },
    "Globe Life Field (TEX)": {
        "type": "dome", "hr_factor": 1.05, "run_factor": 1.02,
        "notes": "Dome. No weather. Slight hitter lean.",
    },
    "Tropicana Field (TB)": {
        "type": "dome", "hr_factor": 0.95, "run_factor": 0.97,
        "notes": "Dome. Slight pitcher lean. McClanahan 86% NRFI here.",
    },
    "Rogers Centre (TOR)": {
        "type": "dome", "hr_factor": 1.02, "run_factor": 1.00,
        "notes": "Dome. Neutral. No weather variance.",
    },
    "T-Mobile Park (SEA)": {
        "type": "retractable", "hr_factor": 0.82, "run_factor": 0.82,
        "notes": "LOWEST park factor in MLB (0.816). Strong NRFI lean.",
    },
    "Minute Maid Park (HOU)": {
        "type": "dome", "hr_factor": 1.08, "run_factor": 1.05,
        "notes": "Dome. Hitter-friendly. LF hill unique.",
    },
    "Petco Park (SD)": {
        "type": "outdoor", "hr_factor": 0.92, "run_factor": 0.93,
        "notes": "Pitcher-friendly. SD 0.11 first-inning R/game historically.",
    },
    "American Family Field (MIL)": {
        "type": "dome", "hr_factor": 1.06, "run_factor": 1.03,
        "notes": "Dome. Moderate hitter lean.",
    },
    "Wrigley Field (CHC)": {
        "type": "outdoor", "hr_factor": 1.12, "run_factor": 1.08,
        "notes": "#1 wind-affected park. Check wind direction EVERY game.",
    },
    "Camden Yards (BAL)": {
        "type": "outdoor", "hr_factor": 1.05,
        "lhb_hr_factor": 1.12, "rhb_hr_factor": 1.00,
        "notes": "LHB-friendly RF fence. Check wind.",
    },
    "Fenway Park (BOS)": {
        "type": "outdoor", "hr_factor": 1.03,
        "notes": "Green Monster helps RHB. LHB pulls to RF bullpen area.",
    },
    "Yankee Stadium (NYY)": {
        "type": "outdoor", "hr_factor": 1.10,
        "notes": "Short RF porch. Wind out to LF historically favorable.",
    },
    "Sutter Health Park (ATH)": {
        "type": "outdoor", "hr_factor": 1.18, "run_factor": 1.12,
        "notes": "Minor league park. Hot temps. Wind out amplifies all HR.",
    },
    "loanDepot Park (MIA)": {
        "type": "dome", "hr_factor": 0.97, "run_factor": 0.98,
        "notes": "Dome. Pitcher lean. No weather.",
    },
    "Chase Field (AZ)": {
        "type": "retractable", "hr_factor": 1.06, "run_factor": 1.04,
        "notes": "High altitude (1100 ft). Retractable roof.",
    },
    "Oracle Park (SF)": {
        "type": "outdoor", "hr_factor": 0.90, "run_factor": 0.91,
        "notes": "Pitcher-friendly. SF UNDER in 14/16 May games historically.",
    },
    "Citizens Bank Park (PHI)": {
        "type": "outdoor", "hr_factor": 1.06, "run_factor": 1.04,
        "notes": "Near sea-level. Check wind. Moderate hitter lean.",
    },
    "Dodger Stadium (LAD)": {
        "type": "outdoor", "hr_factor": 1.00, "run_factor": 0.99,
        "notes": "Neutral park. Afternoon shadows a factor in day games.",
    },
    "Target Field (MIN)": {
        "type": "outdoor", "hr_factor": 0.98, "run_factor": 0.97,
        "notes": "Cold early season suppresses offense. May temps key.",
    },
    "Truist Park (ATL)": {
        "type": "outdoor", "hr_factor": 1.02, "run_factor": 1.01,
        "notes": "Neutral. ATL offense is elite regardless of park.",
    },
    "PNC Park (PIT)": {
        "type": "outdoor", "hr_factor": 0.95, "run_factor": 0.95,
        "notes": "Pitcher-friendly. Check rain probability.",
    },
    "Busch Stadium (STL)": {
        "type": "outdoor", "hr_factor": 0.97, "run_factor": 0.96,
        "notes": "Pitcher-friendly. Wind direction matters.",
    },
    "Kauffman Stadium (KC)": {
        "type": "outdoor", "hr_factor": 0.95, "run_factor": 0.96,
        "notes": "Large outfield. Pitcher lean. Check wind.",
    },
    "Progressive Field (CLE)": {
        "type": "outdoor", "hr_factor": 1.00, "run_factor": 0.99,
        "notes": "Neutral park.",
    },
    "Guaranteed Rate Field (CWS)": {
        "type": "outdoor", "hr_factor": 1.02, "run_factor": 1.00,
        "notes": "Near neutral.",
    },
    "Oriole Park (BAL)": {
        "type": "outdoor", "hr_factor": 1.05, "run_factor": 1.02,
        "notes": "See Camden Yards entry.",
    },
    "Nationals Park (WSH)": {
        "type": "outdoor", "hr_factor": 1.01, "run_factor": 1.00,
        "notes": "Near neutral.",
    },
    "Angel Stadium (LAA)": {
        "type": "outdoor", "hr_factor": 1.01, "run_factor": 1.00,
        "notes": "Near neutral. NOTE: LAA PARLAY BANNED regardless of park.",
    },
}

# Aliases: team name -> park key
TEAM_TO_PARK: dict[str, str] = {
    "Cincinnati Reds":       "Great American Ball Park (CIN)",
    "Colorado Rockies":      "Coors Field (COL)",
    "Texas Rangers":         "Globe Life Field (TEX)",
    "Tampa Bay Rays":        "Tropicana Field (TB)",
    "Toronto Blue Jays":     "Rogers Centre (TOR)",
    "Seattle Mariners":      "T-Mobile Park (SEA)",
    "Houston Astros":        "Minute Maid Park (HOU)",
    "San Diego Padres":      "Petco Park (SD)",
    "Milwaukee Brewers":     "American Family Field (MIL)",
    "Chicago Cubs":          "Wrigley Field (CHC)",
    "Baltimore Orioles":     "Camden Yards (BAL)",
    "Boston Red Sox":        "Fenway Park (BOS)",
    "New York Yankees":      "Yankee Stadium (NYY)",
    "Oakland Athletics":     "Sutter Health Park (ATH)",
    "Athletics":             "Sutter Health Park (ATH)",
    "Miami Marlins":         "loanDepot Park (MIA)",
    "Arizona Diamondbacks":  "Chase Field (AZ)",
    "San Francisco Giants":  "Oracle Park (SF)",
    "Philadelphia Phillies": "Citizens Bank Park (PHI)",
    "Los Angeles Dodgers":   "Dodger Stadium (LAD)",
    "Minnesota Twins":       "Target Field (MIN)",
    "Atlanta Braves":        "Truist Park (ATL)",
    "Pittsburgh Pirates":    "PNC Park (PIT)",
    "St. Louis Cardinals":   "Busch Stadium (STL)",
    "Kansas City Royals":    "Kauffman Stadium (KC)",
    "Cleveland Guardians":   "Progressive Field (CLE)",
    "Chicago White Sox":     "Guaranteed Rate Field (CWS)",
    "Washington Nationals":  "Nationals Park (WSH)",
    "Los Angeles Angels":    "Angel Stadium (LAA)",
    "New York Mets":         "Citi Field (NYM)",
}


def get_park(home_team: str) -> dict:
    """Get park data for a home team. Returns empty dict if unknown."""
    park_key = TEAM_TO_PARK.get(home_team, "")
    return PARK_DATABASE.get(park_key, {})


def is_dome(home_team: str) -> bool:
    park = get_park(home_team)
    return park.get("type") in ("dome", "retractable")


def get_hr_factor(home_team: str, batter_hand: str = "neutral") -> float:
    park = get_park(home_team)
    if batter_hand == "L" and park.get("lhb_hr_factor"):
        return park["lhb_hr_factor"]
    if batter_hand == "R" and park.get("rhb_hr_factor"):
        return park["rhb_hr_factor"]
    return park.get("hr_factor", 1.00)


def get_run_factor(home_team: str) -> float:
    return get_park(home_team).get("run_factor", 1.00)


def wrigley_wind_signal(wind_speed: float, wind_dir_label: str) -> str:
    """Special Wrigley wind rule — most wind-affected park in baseball."""
    if wind_speed >= 10 and wind_dir_label == "out":
        return "STRONG YRFI — STRONG OVER"
    if wind_speed >= 10 and wind_dir_label == "in":
        return "STRONG NRFI — STRONG UNDER"
    if wind_speed >= 6 and wind_dir_label == "out":
        return "YRFI LEAN"
    if wind_speed >= 6 and wind_dir_label == "in":
        return "NRFI LEAN"
    return "NEUTRAL"


# ── Signal 10: Precise stadium wind orientations ──────────────────────
# Home plate → CF compass bearing (degrees).
# Wind bearing vs stadium bearing = true in/out/cross direction.
# Source: stadium blueprints and satellite measurement.

STADIUM_CF_BEARING: dict[str, float] = {
    "Great American Ball Park (CIN)": 30,    # CF points NNE
    "Coors Field (COL)":              330,   # CF points NNW
    "Globe Life Field (TEX)":         None,  # dome
    "Tropicana Field (TB)":           None,  # dome
    "Rogers Centre (TOR)":            None,  # dome/retractable
    "T-Mobile Park (SEA)":            None,  # retractable
    "Minute Maid Park (HOU)":         None,  # retractable
    "Petco Park (SD)":                315,   # CF points NW (marine layer)
    "American Family Field (MIL)":    None,  # retractable
    "Wrigley Field (CHC)":            88,    # CF points nearly due east — SW wind blows OUT
    "Camden Yards (BAL)":             345,   # CF points NNW
    "Fenway Park (BOS)":              30,    # CF points NNE
    "Yankee Stadium (NYY)":           285,   # CF points WNW
    "Sutter Health Park (ATH)":       320,   # CF NW
    "loanDepot Park (MIA)":           None,  # retractable
    "Chase Field (AZ)":               None,  # retractable
    "Oracle Park (SF)":               310,   # CF NW — bay wind almost always blows IN
    "Citizens Bank Park (PHI)":       15,    # CF points NNE
    "Dodger Stadium (LAD)":           30,    # CF points NNE
    "Target Field (MIN)":             350,   # CF points near north
    "Truist Park (ATL)":              10,    # CF NNE
    "PNC Park (PIT)":                 285,   # CF WNW
    "Busch Stadium (STL)":            310,   # CF NW
    "Kauffman Stadium (KC)":          350,   # CF north
    "Progressive Field (CLE)":        330,   # CF NNW
    "Guaranteed Rate Field (CWS)":    350,   # CF north
    "Nationals Park (WSH)":           20,    # CF NNE
    "Angel Stadium (LAA)":            310,   # CF NW
    "Citi Field (NYM)":               30,    # CF NNE
    "Oriole Park (BAL)":              345,   # same as Camden
}


def precise_wind_direction(home_team: str, wind_bearing_deg: float | None) -> str:
    """
    Signal 10: Compute true in/out/cross wind direction using stadium CF bearing.
    Returns 'in', 'out', 'cross-L', 'cross-R', or 'calm'.
    """
    if wind_bearing_deg is None:
        return "calm"

    park_key = TEAM_TO_PARK.get(home_team, "")
    cf_bearing = STADIUM_CF_BEARING.get(park_key)

    if cf_bearing is None:
        return "dome"  # dome/retractable — no wind effect

    # Angle difference: how far wind deviates from CF line
    diff = (wind_bearing_deg - cf_bearing + 360) % 360

    # Wind blowing FROM the direction it's headed toward CF = blowing IN
    # Wind blowing toward CF from behind home plate = blowing OUT
    if diff <= 30 or diff >= 330:
        return "in"       # wind blowing straight in from CF
    elif 150 <= diff <= 210:
        return "out"      # wind at back of batter, blowing toward CF
    elif 30 < diff < 150:
        return "cross-R"  # blowing right to left from batter's perspective
    else:
        return "cross-L"  # blowing left to right
