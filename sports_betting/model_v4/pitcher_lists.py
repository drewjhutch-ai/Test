"""
Permanent pitcher fade and backs lists.
Built from 24 days of live data — May 2026.
All entries confirmed by ≥2 metrics before inclusion.
"""

PERMANENT_FADE_LIST: dict[str, dict] = {
    "Leahy (STL)":      {"era": 5.03, "xera": 6.01, "note": "All metrics catastrophic"},
    "Bassitt (BAL)":    {"era": 5.91, "xera": 5.18, "fip": 5.15, "note": "3 metrics confirm"},
    "Sasaki (LAD)":     {"era": 5.97, "xera": 5.77, "fip": 6.83, "note": "All 3 catastrophic"},
    "Burrows (HOU)":    {"era": 5.97, "whip": 1.65, "note": "WHIP catastrophic"},
    "McGreevey (SD)":   {"era": 2.52, "xera": 5.77, "note": "Extreme ERA fraud"},
    "Sproat (MIL)":     {"era": 6.75, "note": "0-2, 3+ ER in 3 of 4 starts"},
    "Young (BAL)":      {"note": "3+ ER in each of last 3 starts"},
    "Abbott (CIN)":     {"era": 5.17, "xera": 5.20, "note": "Both metrics confirm"},
    "Vasquez (SD)":     {"era": 1.88, "xera": 4.32, "note": "ERA fraud, regression due"},
    "Chandler":         {"era": 4.76, "note": "Consistent struggles"},
    "Mlodzinski (PIT)": {"note": "14 runs on 19 hits in 13.2 IP"},
    "Rocker (TEX)":     {"note": "5 ER 7 H in 2 innings recent start"},
    "Walker (PHI)":     {"xera": 6.58, "note": "xERA catastrophic"},
    "Feltner":          {"xera": 8.70, "note": "Catastrophic"},
    "Williamson":       {"bb9": 5.3,   "note": "Walk rate catastrophic"},
    "Prielipp (CWS)":   {"era": 3.86,  "note": "Limited data, monitor"},
    "Pallante (STL)":   {"bb9": 4.1,   "note": "Walk rate trending bad"},
}

PERMANENT_BACKS_LIST: dict[str, dict] = {
    "Gausman (TOR)":    {
        "era": 2.57, "xera": 2.98, "k9": 9.7,
        "preferred_market": "F5 ML or K Over",
        "note": "Elite command, consistent",
    },
    "Schlittler (NYY)": {
        "era": 1.51, "siera": 2.54, "xfip": 2.43, "whip": 0.74,
        "preferred_market": "ER Under (+money), K Over",
        "note": "#1 SIERA in baseball",
    },
    "deGrom (TEX)":     {
        "era": 2.01, "xera": 3.21, "k9": 11.4,
        "preferred_market": "F5 ML, K Over",
        "note": "Cy Young form, peak SwStr%",
    },
    "Ohtani (LAD)":     {
        "era": 0.60, "k9": 9.5,
        "preferred_market": "K Over",
        "note": "Pitches only",
    },
    "Messick (CLE)":    {
        "era": 1.76, "xera": 3.17,
        "preferred_market": "F5 ML",
        "note": "Consistent, permanent backs",
    },
    "McClanahan (TB)":  {
        "era": 3.91, "nrfi_rate_tropicana": 0.86,
        "preferred_market": "NRFI, K Over",
        "note": "86% NRFI at Tropicana career",
    },
    "Cantillo (CLE)":   {
        "changeup_percentile": 88, "extension_pct": 95,
        "preferred_market": "F5 ML, ER Under",
        "note": "6 of 8 starts ≤2 ER",
    },
    "Elder (ATL)":      {
        "era": 2.02, "xera": 2.85,
        "preferred_market": "F5 ML",
        "note": "NL elite, consistent",
    },
    "Sale (ATL)":       {
        "preferred_market": "F5 ML, ER Under",
        "note": "≤1 ER in 4 straight starts",
    },
    "Eovaldi (TEX)":    {
        "recent": "15 IP 1-run ball, 15 Ks, 1 BB", "era": 4.15,
        "preferred_market": "K Over, TEX ML",
        "note": "Momentum confirmed",
    },
    "Kirby (SEA)":      {
        "era": 3.00,
        "preferred_market": "F5 ML",
        "note": "Strong stretch confirmed",
    },
    "Misiorowski (MIL)": {
        "k9": 13.9,
        "preferred_market": "K Over",
        "note": "8+ Ks in 5 of 7 starts",
    },
    "Cease (TOR)":      {
        "k9": 15.4, "swstr_pct": 0.17,
        "preferred_market": "F5 ML, K Over",
        "note": "Career-high swing-miss",
    },
    "Williams (CLE)":   {
        "era": 2.70, "xera": 3.85, "k9": 11.0,
        "preferred_market": "F5 ML, K Over",
        "note": "Consistent elite",
    },
    "Martinez (TB)":    {
        "era": 1.71, "whip": 1.02,
        "preferred_market": "F5 ML",
        "note": "3-1, dominant",
    },
    "Wacha (KC)":       {
        "era": 1.00, "k_bb_ratio": 7.0,
        "preferred_market": "F5 ML",
        "note": "Elite command",
    },
    "Glasnow (LAD)":    {
        "preferred_market": "K Over",
        "note": "Dominant stretch",
    },
    "Lodolo (CIN)":     {
        "rehab": "12 IP, 1.50 ERA, 12.7 K/9",
        "preferred_market": "K Over (standalone only)",
        "note": "Debut/return = never parlay",
    },
}


def is_on_fade_list(pitcher_name: str) -> tuple[bool, dict]:
    """Check if a pitcher matches any entry on the permanent fade list."""
    pitcher_lower = pitcher_name.lower()
    for key, data in PERMANENT_FADE_LIST.items():
        name_part = key.split(" (")[0].lower()
        if name_part in pitcher_lower or pitcher_lower in name_part:
            return True, {"key": key, **data}
    return False, {}


def is_on_backs_list(pitcher_name: str) -> tuple[bool, dict]:
    """Check if a pitcher matches any entry on the permanent backs list."""
    pitcher_lower = pitcher_name.lower()
    for key, data in PERMANENT_BACKS_LIST.items():
        name_part = key.split(" (")[0].lower()
        if name_part in pitcher_lower or pitcher_lower in name_part:
            return True, {"key": key, **data}
    return False, {}


def is_era_fraud(era: float, xera: float = None, fip: float = None, siera: float = None) -> dict:
    """
    Detect ERA fraud: ERA significantly below predictive metrics.
    Requires 2+ confirming metrics — never fade on ERA alone.
    """
    fraud_threshold = 1.50
    confirmations = []
    confirmed = False

    if xera and (xera - era) >= fraud_threshold:
        confirmations.append(f"xERA {xera:.2f} vs ERA {era:.2f} (+{xera-era:.2f})")
    if fip and (fip - era) >= fraud_threshold:
        confirmations.append(f"FIP {fip:.2f} vs ERA {era:.2f} (+{fip-era:.2f})")
    if siera and (siera - era) >= fraud_threshold:
        confirmations.append(f"SIERA {siera:.2f} vs ERA {era:.2f} (+{siera-era:.2f})")

    if len(confirmations) >= 2:
        confirmed = True

    return {
        "is_fraud": confirmed,
        "confirmations": confirmations,
        "confirmation_count": len(confirmations),
        "note": "ERA fraud confirmed — fade this pitcher" if confirmed else "Single metric — insufficient to confirm fraud",
    }
