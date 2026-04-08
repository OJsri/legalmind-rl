"""LegalMind-RL task definitions — Easy / Medium / Hard."""
from __future__ import annotations
from models import Case, Evidence, Witness

TASK_EASY = Case(
    id="task_shoplifting_v1",
    title="The Riverside Market Theft",
    description=(
        "Defendant Alex Morgan is accused of stealing a $240 jacket from Riverside Market "
        "on March 3rd. Security camera footage captured the incident and a store employee "
        "witnessed the event. The jacket was found in Morgan's bag at the exit."
    ),
    prosecution_goal="Establish theft beyond reasonable doubt using camera footage and witness.",
    defense_goal="Challenge chain of custody and witness reliability.",
    evidence=[
        Evidence(id="E001", description="Security camera footage at 14:32 showing defendant concealing jacket under coat", type="physical", relevance_tags=["direct","timing","concealment"]),
        Evidence(id="E002", description="Store employee Jessica Lin's statement: 'I saw him put the jacket in his bag'", type="testimony", relevance_tags=["eyewitness","direct"]),
        Evidence(id="E003", description="Loss prevention officer found jacket (still tagged) in defendant's bag at exit", type="physical", relevance_tags=["possession","stolen_goods"]),
        Evidence(id="E004", description="Store receipt log: no purchase of jacket by defendant on record", type="document", relevance_tags=["no_payment","intent"]),
    ],
    witnesses=[
        Witness(name="Jessica Lin", testimony="I clearly saw the defendant place the jacket inside his bag near the rack.", credibility=0.9),
        Witness(name="Officer Ben Davis", testimony="Upon exit screening, I found an unscanned jacket with tags intact in the bag.", credibility=0.95),
    ],
    verdict_criteria={"max_reward": 2.0, "expected_contradictions": 0, "key_evidence_ids": ["E001","E003"]},
    difficulty="easy", max_rounds=8,
)

TASK_MEDIUM = Case(
    id="task_fraud_v1",
    title="The Harmon Capital Fraud",
    description=(
        "CFO Diana Harmon is accused of embezzling $180,000 from Harmon Capital LLC via "
        "fraudulent wire transfers to a shell company between January and June 2023. "
        "Two witnesses give conflicting accounts of authorization."
    ),
    prosecution_goal="Prove fraudulent intent through wire records and expose two witness contradictions.",
    defense_goal="Establish transfers were authorized and records are incomplete.",
    evidence=[
        Evidence(id="E001", description="Wire transfer records: 6 transfers totaling $180,000 to 'NexGen Solutions LLC'", type="document", relevance_tags=["financial","transfers"]),
        Evidence(id="E002", description="Secretary of State records: NexGen Solutions LLC registered to Harmon's brother", type="document", relevance_tags=["shell_company","conflict_of_interest"]),
        Evidence(id="E003", description="Email from Harmon to CEO Jan 8: 'Approving vendor NexGen for Q1 services'", type="document", relevance_tags=["authorization","self_dealing"]),
        Evidence(id="E004", description="No service delivery records, invoices, or work product from NexGen on file", type="document", relevance_tags=["no_services_rendered","fraud"]),
        Evidence(id="E005", description="CEO Marcus Webb deposition: 'I never authorized any vendor called NexGen'", type="testimony", relevance_tags=["no_authorization"]),
    ],
    witnesses=[
        Witness(name="Marcus Webb", testimony="I never approved NexGen Solutions as a vendor. I had no knowledge of these transfers.", credibility=0.8, has_contradiction=True, contradiction_hint="Webb signed a Q1 budget approval that included a 'vendor services' line matching transfer amounts exactly."),
        Witness(name="Karen Ortiz", testimony="As accounting manager I processed transfers after verbal sign-off from Diana.", credibility=0.7, has_contradiction=True, contradiction_hint="Ortiz previously stated she 'always required written authorization' — contradicting her claim of verbal sign-off."),
    ],
    verdict_criteria={"max_reward": 3.5, "expected_contradictions": 2, "key_evidence_ids": ["E002","E004","E005"]},
    difficulty="medium", max_rounds=12,
)

TASK_HARD = Case(
    id="task_homicide_v1",
    title="The Vantage Hotel Homicide",
    description=(
        "Defendant Reyes is charged with premeditated murder of businessman Victor Crane at "
        "the Vantage Hotel on November 14th. Prosecution alleges financial motive, opportunity, "
        "and physical evidence. Defense claims alibi backed by hotel records and a third party. "
        "Eight evidence items must be synthesized and three witness contradictions exposed."
    ),
    prosecution_goal="Establish motive, opportunity, physical link; discredit alibi and forensic handling.",
    defense_goal="Destroy prosecution timeline, establish alibi, challenge DNA chain-of-custody.",
    evidence=[
        Evidence(id="E001", description="Hotel key-card log: room 814 accessed at 22:07 using key issued to Reyes", type="physical", relevance_tags=["opportunity","timeline"]),
        Evidence(id="E002", description="Forensic report: defendant's partial fingerprint on whiskey glass near body", type="physical", relevance_tags=["physical_link"]),
        Evidence(id="E003", description="Text messages: Reyes to Crane — 'Pay me back by Friday or you'll regret it' (Nov 12)", type="document", relevance_tags=["motive","threat","premeditation"]),
        Evidence(id="E004", description="Financial records: Crane owed Reyes $340,000 in an undocumented private loan", type="document", relevance_tags=["motive","financial"]),
        Evidence(id="E005", description="Restaurant receipt: Reyes paid at 'La Mer' at 21:45 (1.2 miles from hotel)", type="document", relevance_tags=["alibi","defense"]),
        Evidence(id="E006", description="Toxicology: victim died between 21:30 and 23:00 — exact time disputed ±45 min", type="physical", relevance_tags=["time_of_death","uncertain"]),
        Evidence(id="E007", description="DNA on victim's collar: mixture profile, Reyes cannot be excluded (1-in-850 match)", type="physical", relevance_tags=["dna","probabilistic"]),
        Evidence(id="E008", description="CCTV: hotel lobby shows person matching Reyes's build entering at 21:58 (face obscured)", type="physical", relevance_tags=["opportunity","uncertain_identity"]),
    ],
    witnesses=[
        Witness(name="Dr. Sandra Meeks", testimony="Time of death was approximately 22:15, based on liver temperature and rigor.", credibility=0.85, has_contradiction=True, contradiction_hint="Meeks's written report filed 48h earlier stated window 21:30–23:00; her oral testimony narrows to 22:15 without explanation."),
        Witness(name="Tommy Vega", testimony="I was with Reyes at La Mer from 9 PM until at least 10:30 PM. He never left.", credibility=0.6, has_contradiction=True, contradiction_hint="Vega's credit card shows a 1h23m gap in activity — he cannot account for this gap when pressed."),
        Witness(name="Rachel Park", testimony="I processed the DNA sample according to standard protocol on November 16th.", credibility=0.9, has_contradiction=True, contradiction_hint="Chain-of-custody log shows sample was unrefrigerated for 18 hours before Park received it — not noted in her report."),
    ],
    verdict_criteria={"max_reward": 5.0, "expected_contradictions": 3, "key_evidence_ids": ["E001","E003","E006","E007"]},
    difficulty="hard", max_rounds=16,
)

TASKS: dict[str, Case] = {
    "easy":                TASK_EASY,
    "medium":              TASK_MEDIUM,
    "hard":                TASK_HARD,
    "task_shoplifting_v1": TASK_EASY,
    "task_fraud_v1":       TASK_MEDIUM,
    "task_homicide_v1":    TASK_HARD,
}

def get_task(name: str) -> Case:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]
