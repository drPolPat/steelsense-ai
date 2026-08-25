"""Curated domain knowledge base for SteelSense AI's RAG layer.

Every entry below is an original summary written for this project to give
the agent grounded domain context on magnetic stress sensing, steel
fatigue, and structural health monitoring (SHM) practice. None of the text
is copied verbatim from any source. `source_note` names the general body
of literature or standard the summary is informed by, for attribution --
not as a quotation, and not as a claim of authoritative/verified content.
This is a portfolio project; nothing here should be treated as engineering
or safety guidance for a real structure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeDoc:
    doc_id: str
    title: str
    category: str
    text: str
    source_note: str


KNOWLEDGE_BASE: list[KnowledgeDoc] = [
    KnowledgeDoc(
        doc_id="magnetoelastic-effect",
        title="The Magnetoelastic (Villari) Effect in Ferromagnetic Steel",
        category="sensing-principles",
        text=(
            "Ferromagnetic steel is built from microscopic magnetic domains whose "
            "alignment shifts under mechanical stress -- the inverse of "
            "magnetostriction, commonly called the Villari effect. Compressive or "
            "tensile stress reorients these domains, which changes the material's "
            "magnetic permeability. A Hall-effect sensor placed near the steel "
            "picks up the resulting local field change as a proportional voltage "
            "shift. The relationship between stress and signal is monotonic over a "
            "useful working range, but it is not perfectly linear, and it shows "
            "hysteresis: the reading depends somewhat on the material's recent "
            "stress history, not only its instantaneous load. Sensor systems "
            "typically address this by tracking relative change from a "
            "known-good baseline rather than treating any single reading as an "
            "absolute stress value."
        ),
        source_note=(
            "General synthesis of magnetomechanical-effect literature, e.g. "
            "Jiles, D.C., Introduction to Magnetism and Magnetic Materials, and "
            "the Jiles-Atherton magnetomechanical hysteresis model."
        ),
    ),
    KnowledgeDoc(
        doc_id="temperature-cross-sensitivity",
        title="Temperature Cross-Sensitivity and Compensation",
        category="sensing-principles",
        text=(
            "Both a steel structure's magnetic permeability and a Hall sensor's "
            "own sensitivity vary with temperature, independent of any change in "
            "mechanical stress. A raw sensor reading therefore conflates two "
            "effects: genuine stress-driven change and ambient thermal drift. "
            "Left uncorrected, ordinary day-night or seasonal temperature swings "
            "can look like a stress anomaly. Standard practice is to log ambient "
            "or sensor-local temperature alongside every reading and subtract an "
            "expected thermal contribution -- usually a simple per-sensor "
            "coefficient calibrated during commissioning -- before evaluating a "
            "reading for anomalies. Any anomaly-detection pipeline over magnetic "
            "sensor data should treat a temperature-correlated change very "
            "differently from one that persists after thermal correction."
        ),
        source_note=(
            "General synthesis of magnetic-sensor calibration practice; thermal "
            "cross-sensitivity is a well-known confound in magnetoelastic and "
            "Hall-effect stress sensing literature."
        ),
    ),
    KnowledgeDoc(
        doc_id="fatigue-mechanism-steel",
        title="Cyclic Fatigue Failure Mechanisms in Structural Steel",
        category="fatigue-analysis",
        text=(
            "Fatigue failure in structural steel proceeds in three broad stages: "
            "crack initiation at a stress concentration (a weld toe, connection, "
            "or existing flaw), stable crack growth under repeated load cycles, "
            "and finally rapid fracture once the remaining cross-section can no "
            "longer carry the load. Crack growth rate is commonly described by "
            "the Paris law relating growth per cycle to the stress-intensity "
            "range, and cumulative fatigue life is often estimated with Miner's "
            "rule, which sums damage fractions across cycles at different stress "
            "amplitudes. The critical point for monitoring purposes: fatigue "
            "damage accumulates from repeated cycles well below the material's "
            "yield strength. A structure can show no single alarming overload "
            "event and still be accumulating meaningful fatigue damage from "
            "ordinary, repeated traffic or operational loading."
        ),
        source_note=(
            "General synthesis of standard fatigue-analysis concepts (S-N/"
            "Basquin relations, Paris law, Miner's rule) as covered in fatigue "
            "testing standards such as ASTM E466 and common structural fatigue "
            "references."
        ),
    ),
    KnowledgeDoc(
        doc_id="drift-vs-spike-vs-divergence",
        title="Distinguishing Drift, Spike, and Divergence Signatures",
        category="diagnostics",
        text=(
            "Three anomaly shapes tend to map to different underlying causes. A "
            "gradual, monotonic drift -- especially one where a cyclic-load "
            "oscillation's amplitude is also slowly growing -- is consistent with "
            "cumulative fatigue damage building up under normal repeated loading. "
            "A sudden step or spike, particularly one that partially but not "
            "fully recovers, suggests a discrete overload or impact event that "
            "may have caused localized plastic deformation. Divergence between "
            "two sensors that normally track each other closely (e.g. "
            "symmetric locations on paired beams) points to a localized issue -- "
            "a developing crack, a loosened connection, or load redistribution -- "
            "affecting one location disproportionately rather than the structure "
            "as a whole. These are heuristics, not diagnoses: each pattern "
            "narrows down what to investigate, not a confirmed root cause."
        ),
        source_note=(
            "General synthesis of anomaly-classification reasoning common in "
            "structural health monitoring (SHM) practice."
        ),
    ),
    KnowledgeDoc(
        doc_id="shm-principles",
        title="Structural Health Monitoring: General Principles",
        category="shm-standards",
        text=(
            "Structural health monitoring (SHM) systems work by establishing a "
            "sensor baseline under known-good conditions, then continuously or "
            "periodically comparing new readings against that baseline using "
            "statistical or threshold-based checks. A widely referenced way to "
            "frame SHM's scope is a four-level hierarchy: (1) detecting that "
            "something has changed, (2) localizing where, (3) assessing the "
            "severity of the change, and (4) estimating remaining useful life. "
            "Most automated SHM sensor systems, including sensor-based tools "
            "like this one, are realistically only reliable through levels 1 and "
            "2 -- detection and rough localization. Severity assessment and "
            "remaining-life prediction require engineering judgment and, "
            "typically, physical inspection. SHM data is best understood as "
            "triaging where a qualified inspector should look next, not as a "
            "replacement for that inspection."
        ),
        source_note=(
            "General reference to Rytter's four-level SHM damage-identification "
            "hierarchy, a commonly cited framing in SHM literature."
        ),
    ),
    KnowledgeDoc(
        doc_id="bridge-inspection-standards",
        title="Bridge Inspection & Monitoring Standards (General Overview)",
        category="shm-standards",
        text=(
            "In the US, formal bridge inspection is governed by frameworks such "
            "as the National Bridge Inspection Standards and AASHTO's Manual for "
            "Bridge Evaluation, which mandate periodic visual and physical "
            "inspection cycles and assign standardized condition ratings. "
            "Continuous sensor-based monitoring, including magnetic/Hall-effect "
            "stress sensing, is generally positioned as a supplement that "
            "provides data between formal inspection cycles -- useful for "
            "catching a developing issue earlier -- rather than a replacement "
            "for a certified inspector's sign-off. Standardization specifically "
            "for magnetic-sensor-based SHM is still comparatively immature "
            "relative to visual inspection protocols, which is part of why "
            "sensor flags are best treated as prompts to schedule inspection "
            "rather than as stand-alone condition assessments."
        ),
        source_note=(
            "General reference to the existence and role of the National Bridge "
            "Inspection Standards and AASHTO Manual for Bridge Evaluation as "
            "named frameworks; no text reproduced from either document."
        ),
    ),
    KnowledgeDoc(
        doc_id="sensor-limitations",
        title="Practical Limitations of Magnetic Stress Sensing",
        category="sensing-principles",
        text=(
            "Magnetic stress sensing has several practical limitations worth "
            "accounting for when interpreting data. Hysteresis means a reading "
            "reflects load history, not just the current instant, so two "
            "readings at the same apparent stress level can differ. "
            "Sensor-to-sensor variation in mounting, material microstructure, "
            "and manufacturing tolerance means each sensor typically needs its "
            "own calibrated baseline rather than a single shared threshold. "
            "Nearby ferromagnetic objects -- vehicles, equipment, even rebar in "
            "adjacent concrete -- can introduce noise unrelated to structural "
            "stress. And magnetic response tends to saturate at high stress "
            "levels, which reduces the sensor's ability to discriminate between "
            "\"concerning\" and \"very concerning\" exactly when that "
            "distinction matters most."
        ),
        source_note=(
            "General synthesis of practical caveats commonly noted in magnetic "
            "and magnetoelastic sensing literature."
        ),
    ),
    KnowledgeDoc(
        doc_id="response-guidance",
        title="Interpreting Flags: When to Escalate",
        category="diagnostics",
        text=(
            "Not every flagged deviation warrants the same response. A small "
            "drift that tracks with ordinary ambient temperature swings is "
            "typically not a structural concern once thermally corrected. A "
            "gradual fatigue-consistent drift sustained over multiple weeks "
            "warrants continued monitoring and, if it continues, scheduling a "
            "physical inspection -- it is rarely urgent on its own within a "
            "short window, but the trend matters. A sudden spike well beyond a "
            "sensor's normal noise band, especially one that leaves a lasting "
            "residual offset, is the pattern most worth prompt human follow-up, "
            "since it suggests a discrete event may have caused lasting change. "
            "Divergence from a normally-correlated sensor pair should narrow "
            "attention to that specific location. In all cases, this kind of "
            "sensor reasoning is meant to inform where a qualified engineer "
            "looks next, not to stand in for their assessment."
        ),
        source_note=(
            "General synthesis of triage heuristics common in SHM practice; not "
            "a substitute for engineering judgment or certified inspection."
        ),
    ),
]
