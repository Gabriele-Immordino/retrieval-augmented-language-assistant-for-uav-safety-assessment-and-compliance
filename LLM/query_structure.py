# --- 1) Replace query_structure with a minimal + conditional template ----------

BASE_QUERIES = {
    "likely_regulatory_pathway": [
        # keep only “routing” anchors; drop generic/legal fluff
        "STS-01", "STS-02",
        "PDRA-S01", "PDRA-S02", "PDRA-G01", "PDRA-G02", "PDRA-G03",
        "standard scenario", "predefined risk assessment", "PDRA",
        "SORA", "AMC1",
    ],
    "initial_ground_risk_orientation": [
        # keep only ground-risk specific anchors
        "intrinsic ground risk class", "GRC", "unmitigated ground risk",
        "population density", "assembly of people",
        "controlled ground area", "ground risk buffer",
        "maximum characteristic dimension", "kinetic energy", "lethal area",
        "SORA Step #2",
    ],
    "initial_air_risk_orientation": [
        # keep only air-risk specific anchors
        "initial air risk class", "ARC",
        "airspace collision risk", "airspace characterisation",
        "decision tree", "aggregated collision risk categories",
        "aerodrome environment", "urban airspace", "rural airspace",
        "SORA Step #4",
    ],
    "expected_assessment_depth": [
        # depth decision anchors (not generic “assessment” words)
        "declaration", "standard scenario", "predefined risk assessment", "PDRA",
        "full risk assessment", "SORA process", "pre-application evaluation",
        "not fully fit within limits",  # triggers full SORA in the doc language
    ],
}

# --- 2) Condition-to-keywords mapping ----------------------------------------

VLOS_MODE_TERMS = {
    "VLOS": ["VLOS", "visual line of sight", "de-confliction scheme"],
    "BVLOS": ["BVLOS", "beyond visual line of sight", "TMPR", "tactical mitigations", "detect and avoid", "DAA"],
}

GROUND_ENV_TERMS = {
    "controlled_area": ["controlled ground area", "no uninvolved persons", "procedures", "containment"],
    "sparsely_populated": ["sparsely populated area", "rural area"],
    "populated": ["populated area", "congested area", "urban area"],
}

AIRSPACE_TYPE_TERMS = {
    "controlled": ["controlled airspace", "ANSP", "prior approval"],
    "uncontrolled": ["uncontrolled airspace"],
}

MASS_TERMS = {
    "lt_25kg": ["maximum take-off mass 25 kg", "up to 25 kg"],
    "gte_25kg": ["greater than 25 kg", "above 25 kg"],
}

ALTITUDE_TERMS = {
    "le_50m": ["up to 50 m AGL", "low altitude"],
    "gt_50m_le_120m": ["above 50 m", "up to 120 m AGL", "120 m"],
    "gt_120m_le_150m": ["above 120 m", "up to 150 m AGL", "150 m"],
    "gt_150m": ["above 150 m AGL", "greater than 150 m"],
}

# # Optional: map ground/air inputs to the concrete STS/PDRA entries to bias retrieval.
# # This makes pathway queries “snap” to the right table rows.
# PATHWAY_HINTS = {
#     # STS
#     ("VLOS", "controlled_area", "lt_25kg", "gt_50m_le_120m"): ["STS-01", "C5 class marking", "120 metres"],
#     ("BVLOS", "controlled_area", "lt_25kg", "gt_50m_le_120m"): ["STS-02", "C6 class marking", "120 metres"],
#     # PDRA (height differs: often 150 m)
#     ("VLOS", "controlled_area", "lt_25kg", "gt_120m_le_150m"): ["PDRA-S01", "150 metres"],
#     ("BVLOS", "controlled_area", "lt_25kg", "gt_120m_le_150m"): ["PDRA-S02", "150 metres"],
#     ("BVLOS", "sparsely_populated", "lt_25kg", "gt_120m_le_150m"): ["PDRA-G01", "uncontrolled airspace", "150 metres"],
# }