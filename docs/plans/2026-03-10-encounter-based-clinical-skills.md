# Encounter-Based Clinical Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace calculator-specific workflows with generalizable encounter-based clinical skills that the agent selects and loads dynamically, like Claude Code's Skill tool.

**Architecture:** Skills are Markdown files with three layers (clinical reasoning framework, decision triggers, safety invariants). The ReAct loop gains an internal `load_clinical_skill` tool that injects skill content into the conversation. Evaluation splits into two tiers: calculator reliability (existing) and clinical scenario assessment (new).

**Tech Stack:** Python 3.10+, Pydantic, PyYAML, pytest + pytest-asyncio

---

## Task 1: Skill Loader Module

**Files:**
- Create: `src/fangbot/skills/clinical_loader.py`
- Create: `src/fangbot/skills/clinical/registry.yaml`
- Test: `tests/unit/test_clinical_loader.py`

**Step 1: Write the failing test**

Create `tests/unit/__init__.py` and `tests/unit/test_clinical_loader.py`:

```python
# tests/unit/__init__.py
# (empty)
```

```python
# tests/unit/test_clinical_loader.py
"""Tests for the clinical skill loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from fangbot.skills.clinical_loader import ClinicalSkillLoader, SkillNotFoundError


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with registry and one skill."""
    skill_dir = tmp_path / "clinical"
    skill_dir.mkdir()

    registry = skill_dir / "registry.yaml"
    registry.write_text(
        "skills:\n"
        "  - name: initial_consultation\n"
        '    description: "First ambulatory visit"\n'
        "    encounter_types:\n"
        '      - "new_patient"\n'
        '      - "first_visit"\n'
    )

    skill_file = skill_dir / "initial_consultation.md"
    skill_file.write_text("# Initial Consultation\n\nFramework content here.\n")

    return skill_dir


class TestClinicalSkillLoader:
    def test_load_registry(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        registry = loader.registry
        assert len(registry) == 1
        assert registry[0].name == "initial_consultation"
        assert "new_patient" in registry[0].encounter_types

    def test_list_skills_returns_names_and_descriptions(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        listing = loader.list_skills()
        assert len(listing) == 1
        assert listing[0]["name"] == "initial_consultation"
        assert listing[0]["description"] == "First ambulatory visit"

    def test_load_skill_returns_content(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        content = loader.load_skill("initial_consultation")
        assert "# Initial Consultation" in content
        assert "Framework content here." in content

    def test_load_unknown_skill_raises(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        with pytest.raises(SkillNotFoundError, match="unknown_skill"):
            loader.load_skill("unknown_skill")

    def test_get_tool_definition(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        tool_def = loader.get_tool_definition()
        assert tool_def.name == "load_clinical_skill"
        assert "initial_consultation" in tool_def.description
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_clinical_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fangbot.skills.clinical_loader'`

**Step 3: Write minimal implementation**

```python
# src/fangbot/skills/clinical_loader.py
"""Loader for encounter-based clinical reasoning skills."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from fangbot.models import ToolDefinition

logger = logging.getLogger(__name__)

# Default path — resolved relative to this file
_DEFAULT_SKILLS_DIR = Path(__file__).parent / "clinical"


class SkillNotFoundError(Exception):
    """Raised when a requested clinical skill does not exist."""


class SkillEntry(BaseModel):
    """A single entry in the skill registry."""

    name: str
    description: str
    encounter_types: list[str] = Field(default_factory=list)


class ClinicalSkillLoader:
    """Discovers and loads clinical reasoning skill files."""

    def __init__(self, skills_dir: Path | None = None):
        self._skills_dir = skills_dir or _DEFAULT_SKILLS_DIR
        self._registry: list[SkillEntry] | None = None

    @property
    def registry(self) -> list[SkillEntry]:
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> list[SkillEntry]:
        registry_path = self._skills_dir / "registry.yaml"
        if not registry_path.exists():
            logger.warning(f"No skill registry found at {registry_path}")
            return []
        data = yaml.safe_load(registry_path.read_text())
        return [SkillEntry(**entry) for entry in data.get("skills", [])]

    def list_skills(self) -> list[dict[str, str]]:
        """Return a list of available skills with name and description."""
        return [
            {"name": entry.name, "description": entry.description}
            for entry in self.registry
        ]

    def load_skill(self, skill_name: str) -> str:
        """Load the content of a clinical skill by name."""
        valid_names = {entry.name for entry in self.registry}
        if skill_name not in valid_names:
            raise SkillNotFoundError(
                f"Skill '{skill_name}' not found. Available: {sorted(valid_names)}"
            )
        skill_path = self._skills_dir / f"{skill_name}.md"
        if not skill_path.exists():
            raise SkillNotFoundError(
                f"Skill file not found: {skill_path}"
            )
        return skill_path.read_text()

    def get_tool_definition(self) -> ToolDefinition:
        """Return the ToolDefinition for load_clinical_skill."""
        skill_names = [entry.name for entry in self.registry]
        return ToolDefinition(
            name="load_clinical_skill",
            description=(
                "Load a clinical reasoning framework for the current encounter. "
                f"Available skills: {', '.join(skill_names)}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": skill_names,
                        "description": "The encounter type skill to load",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this skill is relevant to the current encounter",
                    },
                },
                "required": ["skill_name"],
            },
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_clinical_loader.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/fangbot/skills/clinical_loader.py tests/unit/__init__.py tests/unit/test_clinical_loader.py
git commit -m "feat: add clinical skill loader with registry and tool definition"
```

---

## Task 2: Skill Registry and Placeholder Skills

**Files:**
- Create: `src/fangbot/skills/clinical/registry.yaml`
- Create: `src/fangbot/skills/clinical/initial_consultation.md`
- Create: `src/fangbot/skills/clinical/follow_up.md`
- Create: `src/fangbot/skills/clinical/preoperative_evaluation.md`
- Create: `src/fangbot/skills/clinical/preventive_checkup.md`

**Step 1: Create the registry**

```yaml
# src/fangbot/skills/clinical/registry.yaml
skills:
  - name: initial_consultation
    description: "First ambulatory visit — full HPI, PMH, exam, assessment/plan"
    encounter_types:
      - "new_patient"
      - "first_visit"
      - "initial_evaluation"

  - name: follow_up
    description: "Return ambulatory visit — interval history, focused exam, plan review"
    encounter_types:
      - "follow_up"
      - "return_visit"
      - "recheck"

  - name: preoperative_evaluation
    description: "Surgical clearance — risk stratification, medication holds"
    encounter_types:
      - "preop"
      - "surgical_clearance"
      - "pre_surgical"

  - name: preventive_checkup
    description: "Annual preventive visit — screening, immunizations, chronic disease management"
    encounter_types:
      - "annual_exam"
      - "wellness_visit"
      - "screening"
      - "checkup"
```

**Step 2: Create placeholder skills with three-layer structure**

Each skill file gets the correct skeleton. The clinical content will be authored by the user (domain expert) — we write the structure and key sections, marking where detailed clinical content goes.

```markdown
<!-- src/fangbot/skills/clinical/initial_consultation.md -->
# Initial Consultation

## Clinical Reasoning Framework

You are conducting an initial ambulatory consultation. Follow this systematic approach, adapting depth and order to the clinical context:

1. **Chief Complaint & Context** — Establish why the patient is here. Clarify urgency, duration, and what prompted the visit.
2. **Problem Representation** — Build a concise problem representation: age, sex, relevant comorbidities, acute vs. chronic, key features.
3. **History of Present Illness** — Gather targeted HPI using the clinical context. Focus on pertinent positives and negatives for the emerging differential.
4. **Past Medical/Surgical History** — Document active diagnoses, prior surgeries, and relevant hospitalizations.
5. **Medications & Allergies** — Perform medication reconciliation. Identify adherence issues, duplications, and interactions.
6. **Review of Systems** — Targeted ROS based on the chief complaint and emerging differential.
7. **Physical Examination** — Describe relevant exam findings. Focus on systems pertinent to the differential.
8. **Assessment** — Synthesize findings into a problem list. For each problem, provide a working diagnosis or differential.
9. **Plan** — For each problem: investigations, medications, lifestyle modifications, referrals, follow-up timeline.

Adapt this framework to the patient. A straightforward hypertension follow-up needs less depth than a complex multi-system presentation. Let the clinical picture guide which steps deserve more attention.

## Clinical Decision Triggers

### Hardcoded (Must-Not-Miss)

When you identify any of these conditions, you MUST use the corresponding clinical calculator via MCP tools:

| Condition Identified | Required Action | Guideline Source |
|---|---|---|
| Atrial fibrillation | Calculate CHA2DS2-VASc for stroke risk | AHA/ACC/HRS 2023 AFib Guidelines |
| Atrial fibrillation + anticoagulation decision | Calculate HAS-BLED for bleeding risk | 2018 ESC Guidelines on AF |
| Chronic kidney disease or nephrotoxic medications | Calculate CKD-EPI for GFR estimation | KDIGO 2024 CKD Guidelines |
| Suspected heart failure | Assess NYHA class, consider BNP/NT-proBNP | AHA/ACC 2022 HF Guidelines |
| Type 2 diabetes | Assess HbA1c target, cardiovascular risk | ADA Standards of Care 2025 |
| Cardiovascular risk assessment needed | Calculate 10-year ASCVD risk | AHA/ACC 2019 Primary Prevention |
| Altered mental status or trauma | Calculate Glasgow Coma Scale | Teasdale & Jennett, Lancet 1974 |

### Dynamic Discovery

For conditions not listed above, use `search_clinical_calculators` and `search_guidelines` to discover relevant tools and evidence. Always prefer validated calculators over parametric knowledge.

## Safety Invariants

These MUST be addressed in every initial consultation, regardless of chief complaint:

- [ ] **Medication reconciliation** — Document current medications, verify doses, check for interactions
- [ ] **Allergy verification** — Confirm and document drug allergies vs. intolerances
- [ ] **Red flag screening** — Screen for red flags relevant to the chief complaint (e.g., chest pain: ACS features; headache: thunderclap onset, neurological deficits)
- [ ] **Contraindication check** — Before any new medication recommendation, verify no contraindications
- [ ] **Preventive care gaps** — Note any overdue screenings or immunizations relevant to the patient's age/sex/risk factors

### Escalation Criteria

Recommend immediate escalation (ED referral / urgent specialist consult) if:
- Hemodynamic instability signs in the history
- Red flag symptoms suggesting life-threatening diagnosis
- Findings requiring time-sensitive intervention (e.g., STEMI criteria, stroke symptoms)

## Decision Guidance

When making clinical decisions, structure your reasoning as:

1. **What does the evidence say?** — Cite the specific guideline or calculator result
2. **How does it apply to THIS patient?** — Consider comorbidities, contraindications, patient factors
3. **What are the options?** — Present acceptable alternatives with trade-offs
4. **What is the recommendation?** — State your recommendation with reasoning
5. **What needs follow-up?** — Timeline for reassessment, monitoring parameters

For disposition decisions: default to outpatient management unless red flags or instability criteria are met.
For medication decisions: always state the indication, verify no contraindications, and specify follow-up monitoring.
For referral decisions: state the clinical question for the specialist.
```

```markdown
<!-- src/fangbot/skills/clinical/follow_up.md -->
# Follow-Up Visit

## Clinical Reasoning Framework

You are conducting an ambulatory follow-up visit. The patient has an established relationship and known problem list. Focus on:

1. **Interval History** — What has changed since the last visit? New symptoms, medication changes, hospitalizations, ED visits.
2. **Medication Review** — Adherence, side effects, refills needed. Reconcile with current list.
3. **Problem-Oriented Review** — For each active problem:
   - Current status (improved, stable, worsening)
   - Relevant interval data (labs, imaging, specialist notes)
   - Response to current treatment
4. **Focused Examination** — Targeted to active problems and any new complaints.
5. **Assessment Update** — Update problem list. Reclassify diagnoses if needed.
6. **Plan Adjustment** — For each problem: continue, modify, or escalate treatment. Set next follow-up.

The depth of each step depends on the complexity of the visit. A stable chronic disease check needs less than a post-hospitalization follow-up.

## Clinical Decision Triggers

### Hardcoded (Must-Not-Miss)

| Condition Identified | Required Action | Guideline Source |
|---|---|---|
| Atrial fibrillation (on anticoagulation) | Reassess CHA2DS2-VASc and HAS-BLED if risk factors changed | AHA/ACC/HRS 2023 AFib Guidelines |
| CKD on nephrotoxic medications | Recalculate CKD-EPI GFR with latest creatinine | KDIGO 2024 CKD Guidelines |
| Diabetes follow-up | Assess HbA1c trend, adjust targets | ADA Standards of Care 2025 |
| Heart failure follow-up | Reassess NYHA class, volume status, medication optimization | AHA/ACC 2022 HF Guidelines |
| Hypertension follow-up | Assess BP trend, medication adherence, target achievement | AHA/ACC 2017 HTN Guidelines |

### Dynamic Discovery

For conditions not listed above, use `search_clinical_calculators` and `search_guidelines` to discover relevant tools and evidence.

## Safety Invariants

- [ ] **Medication reconciliation** — Compare current medications to what was prescribed at last visit
- [ ] **Lab/imaging review** — Address all pending results since last visit
- [ ] **Overdue screenings** — Flag any overdue preventive care
- [ ] **Contraindication recheck** — If adjusting medications, re-verify contraindications
- [ ] **Worsening detection** — Explicitly assess for signs of clinical deterioration

### Escalation Criteria

Recommend escalation if:
- Significant clinical deterioration since last visit
- Treatment failure despite appropriate therapy
- New red flag symptoms
- Need for urgent specialist intervention

## Decision Guidance

For follow-up visits, frame decisions around:

1. **Is the current plan working?** — Compare to goals set at last visit
2. **What needs to change?** — Only modify what isn't working; don't change stable regimens without reason
3. **Are there new problems?** — Triage new complaints: address now vs. schedule dedicated visit
4. **When is the next visit?** — Based on clinical stability and pending actions
```

```markdown
<!-- src/fangbot/skills/clinical/preoperative_evaluation.md -->
# Preoperative Evaluation

## Clinical Reasoning Framework

You are conducting a preoperative evaluation for surgical clearance. The goal is to assess perioperative risk and optimize the patient for surgery.

1. **Surgical Context** — What procedure? Urgency (elective, urgent, emergent)? Expected duration, blood loss, physiological stress?
2. **Functional Capacity** — Assess METs (metabolic equivalents). Can the patient climb 2 flights of stairs? Walk 2 blocks?
3. **Cardiac Risk Assessment** — Stratify using validated tools based on surgery type and patient factors.
4. **Pulmonary Risk** — Assess for obstructive/restrictive disease, OSA, smoking history.
5. **Medication Management** — Which medications to continue, hold, or bridge perioperatively.
6. **Anesthesia-Relevant History** — Prior anesthesia complications, airway concerns, family history of malignant hyperthermia.
7. **Lab/Testing Requirements** — Based on patient comorbidities and surgical risk, not routine for all.
8. **Risk Communication** — Communicate perioperative risk clearly. Document shared decision-making.
9. **Clearance Decision** — Clear for surgery, clear with conditions/optimization, or defer pending further workup.

## Clinical Decision Triggers

### Hardcoded (Must-Not-Miss)

| Condition Identified | Required Action | Guideline Source |
|---|---|---|
| Non-cardiac surgery risk assessment | Calculate Revised Cardiac Risk Index (RCRI/Lee Index) | AHA/ACC 2014 Perioperative Guidelines |
| Elevated cardiac risk | Consider NSQIP surgical risk calculator | ACS NSQIP |
| Patient on anticoagulation | Assess thrombotic vs. bleeding risk for bridging decision | ACC 2017 Expert Consensus |
| Diabetes (insulin-dependent) | Perioperative glucose management plan | ADA Perioperative Guidelines |
| CKD or renal-risk medications | Calculate CKD-EPI GFR, assess nephrotoxic risk | KDIGO 2024 CKD Guidelines |
| Suspected difficult airway | Document Mallampati score, neck mobility, thyromental distance | ASA Practice Guidelines |

### Dynamic Discovery

Use `search_clinical_calculators` for additional perioperative risk tools relevant to the patient's specific comorbidities.

## Safety Invariants

- [ ] **Medication hold list** — Explicitly document which medications to continue and which to hold, with timing (e.g., "hold metformin 48h before surgery")
- [ ] **Anticoagulation plan** — If on anticoagulants, explicit bridging/holding instructions with resume timing
- [ ] **NPO instructions** — Confirm fasting requirements communicated
- [ ] **Allergy verification** — Confirm allergies documented, especially antibiotics and latex
- [ ] **Blood type/crossmatch** — If significant blood loss expected
- [ ] **VTE prophylaxis plan** — Assess and document VTE risk and prophylaxis strategy

### Escalation Criteria

Defer surgery / request cardiology or specialist consult if:
- Active cardiac conditions (unstable angina, decompensated HF, significant arrhythmia, severe valvular disease)
- Functional capacity < 4 METs with elevated surgical risk
- Uncontrolled comorbidities requiring optimization before elective surgery
- Abnormal preoperative testing requiring further workup

## Decision Guidance

Preoperative decisions center on:

1. **Risk vs. benefit** — Is the surgical risk acceptable given the expected benefit?
2. **Optimization opportunities** — Can modifiable risks be reduced before surgery (BP control, glucose optimization, smoking cessation)?
3. **Medication management** — Clear, specific instructions: drug name, when to stop, when to resume
4. **Communication** — Ensure the surgeon, anesthesiologist, and patient all have the same risk understanding
```

```markdown
<!-- src/fangbot/skills/clinical/preventive_checkup.md -->
# Preventive Checkup

## Clinical Reasoning Framework

You are conducting a preventive health visit (annual exam, wellness visit, screening). Focus on health maintenance, screening, and chronic disease prevention.

1. **Health Maintenance Review** — Review last preventive services: when were screenings last done? What's overdue?
2. **Risk Factor Assessment** — Identify modifiable risk factors: smoking, alcohol, diet, exercise, obesity, family history.
3. **Age/Sex-Appropriate Screening** — Apply USPSTF and specialty guidelines for recommended screenings.
4. **Immunization Review** — Check immunization status against current CDC schedule.
5. **Chronic Disease Monitoring** — For existing chronic conditions, ensure monitoring is up to date.
6. **Mental Health Screening** — PHQ-2/PHQ-9 for depression, assess anxiety, substance use.
7. **Counseling** — Lifestyle modifications, risk reduction, advance directives discussion if appropriate.
8. **Plan** — Order screenings, update immunizations, address gaps, schedule follow-up.

## Clinical Decision Triggers

### Hardcoded (Must-Not-Miss)

| Condition / Age Group | Required Action | Guideline Source |
|---|---|---|
| Age 40-75 + cardiovascular risk factors | Calculate 10-year ASCVD risk score | AHA/ACC 2019 Primary Prevention |
| Age 50-75 | Ensure colorectal cancer screening is current | USPSTF 2021 |
| Women 50-74 | Ensure breast cancer screening is current | USPSTF 2024 |
| Women 21-65 | Ensure cervical cancer screening is current | USPSTF 2018 |
| Smoker age 50-80 with 20+ pack-year history | Low-dose CT lung cancer screening | USPSTF 2021 |
| Diabetes risk factors | Screen with fasting glucose or HbA1c | ADA Standards of Care 2025 |
| Age >= 65 or risk factors | Osteoporosis screening (DEXA) | USPSTF 2018 |

### Dynamic Discovery

Use `search_guidelines` to check for additional age/sex/risk-specific screening recommendations.

## Safety Invariants

- [ ] **Immunization status** — Review and update per CDC schedule
- [ ] **Medication reconciliation** — Even in preventive visits, reconcile the medication list
- [ ] **Allergy list update** — Verify current allergy documentation
- [ ] **Cancer screening status** — Document which screenings are current, overdue, or not applicable
- [ ] **Advance directives** — For appropriate patients, assess if advance directives are documented

### Escalation Criteria

Schedule urgent follow-up or specialist referral if:
- Screening reveals abnormal findings requiring workup
- Previously undiagnosed significant condition identified
- Mental health screening positive for active suicidality

## Decision Guidance

For preventive visits:

1. **Evidence-based screening only** — Follow USPSTF grades A and B recommendations. Do not order tests without evidence-based indication.
2. **Risk-stratified approach** — Not every patient needs every screening. Tailor to individual risk profile.
3. **Shared decision-making** — For screenings with trade-offs (e.g., PSA, lung CT), discuss benefits and harms with the patient.
4. **Actionable plan** — Every identified gap should have a concrete next step: order placed, referral made, or scheduled for follow-up.
```

**Step 3: Verify the loader works with real skills**

Run: `uv run python -c "from fangbot.skills.clinical_loader import ClinicalSkillLoader; l = ClinicalSkillLoader(); print([s['name'] for s in l.list_skills()])"`
Expected: `['follow_up', 'initial_consultation', 'preoperative_evaluation', 'preventive_checkup']`

**Step 4: Commit**

```bash
git add src/fangbot/skills/clinical/
git commit -m "feat: add clinical skill registry and 4 ambulatory encounter skills"
```

---

## Task 3: Tool Routing in the ReAct Loop

**Files:**
- Modify: `src/fangbot/brain/react.py`
- Test: `tests/unit/test_tool_routing.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_tool_routing.py
"""Tests for internal tool routing in the ReAct loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from fangbot.brain.react import ReActLoop
from fangbot.memory.audit import AuditLogger
from fangbot.memory.session import SessionContext
from fangbot.models import ProviderResponse, ToolCall

# Import from tests/ conftest — need to adjust import path
from tests.conftest import MockMCPClient, MockProvider


@pytest.fixture
def audit_logger(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    logger.start_session()
    return logger


@pytest.fixture
def session():
    return SessionContext(system_prompt="Test system prompt")


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "clinical"
    skill_dir.mkdir()
    registry = skill_dir / "registry.yaml"
    registry.write_text(
        "skills:\n"
        "  - name: initial_consultation\n"
        '    description: "First ambulatory visit"\n'
        "    encounter_types:\n"
        '      - "new_patient"\n'
    )
    skill_file = skill_dir / "initial_consultation.md"
    skill_file.write_text("# Initial Consultation\n\nDo systematic assessment.\n")
    return skill_dir


class TestToolRouting:
    @pytest.mark.asyncio
    async def test_internal_tool_handled_locally(
        self, audit_logger, session, skills_dir
    ):
        """load_clinical_skill should be intercepted, not sent to MCP."""
        provider = MockProvider(
            responses=[
                # Agent requests skill load
                ProviderResponse(
                    content="Let me load the clinical framework.",
                    tool_calls=[
                        ToolCall(
                            id="tc_skill",
                            name="load_clinical_skill",
                            arguments={
                                "skill_name": "initial_consultation",
                                "reason": "New patient visit",
                            },
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Agent then calls MCP tool
                ProviderResponse(
                    content="Now calculating CHA2DS2-VASc.",
                    tool_calls=[
                        ToolCall(
                            id="tc_calc",
                            name="search_clinical_calculators",
                            arguments={"query": "CHA2DS2-VASc"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Final synthesis
                ProviderResponse(
                    content="Assessment complete.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()

        from fangbot.skills.clinical_loader import ClinicalSkillLoader

        loader = ClinicalSkillLoader(skills_dir)

        loop = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit_logger,
            clinical_skill_loader=loader,
        )

        from fangbot.models import ToolDefinition

        tools = [
            loader.get_tool_definition(),
            ToolDefinition(
                name="search_clinical_calculators",
                description="Search for clinical calculators",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ]

        result = await loop.run("New patient with AFib", session, tools)

        # Skill load should NOT have been sent to MCP
        mcp_tool_names = [c["name"] for c in mcp.calls]
        assert "load_clinical_skill" not in mcp_tool_names
        # MCP tool should have been called
        assert "search_clinical_calculators" in mcp_tool_names
        # Skill content should be in the conversation
        assert "load_clinical_skill" in result.tool_calls_made

    @pytest.mark.asyncio
    async def test_skill_load_returns_content_to_agent(
        self, audit_logger, session, skills_dir
    ):
        """The skill content should be returned as a tool result."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Loading skill.",
                    tool_calls=[
                        ToolCall(
                            id="tc_skill",
                            name="load_clinical_skill",
                            arguments={"skill_name": "initial_consultation"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Following the framework now.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()

        from fangbot.skills.clinical_loader import ClinicalSkillLoader

        loader = ClinicalSkillLoader(skills_dir)

        loop = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit_logger,
            clinical_skill_loader=loader,
        )

        tools = [loader.get_tool_definition()]
        result = await loop.run("Test skill loading", session, tools)

        # Check that the skill content ended up in the session messages
        tool_messages = [m for m in session.messages if m.role.value == "tool"]
        assert any("Initial Consultation" in m.content for m in tool_messages)

    @pytest.mark.asyncio
    async def test_invalid_skill_returns_error(
        self, audit_logger, session, skills_dir
    ):
        """Loading a non-existent skill should return an error, not crash."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Loading skill.",
                    tool_calls=[
                        ToolCall(
                            id="tc_bad",
                            name="load_clinical_skill",
                            arguments={"skill_name": "nonexistent"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Skill not found, proceeding without framework.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()

        from fangbot.skills.clinical_loader import ClinicalSkillLoader

        loader = ClinicalSkillLoader(skills_dir)
        loop = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit_logger,
            clinical_skill_loader=loader,
        )

        tools = [loader.get_tool_definition()]
        result = await loop.run("Test bad skill", session, tools)

        # Should complete without crashing
        assert result.synthesis is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tool_routing.py -v`
Expected: FAIL — `TypeError: ReActLoop.__init__() got an unexpected keyword argument 'clinical_skill_loader'`

**Step 3: Modify the ReAct loop**

Modify `src/fangbot/brain/react.py` — add `clinical_skill_loader` parameter and internal tool routing.

Changes:
1. Add optional `clinical_skill_loader` parameter to `__init__`
2. Add `SKILL_LOADED` to `EventType` in audit.py
3. In `_execute_tool_calls`, route `load_clinical_skill` to the loader instead of MCP

In `src/fangbot/memory/audit.py`, add to `EventType` enum:

```python
SKILL_LOADED = "skill_loaded"
```

In `src/fangbot/brain/react.py`:

```python
"""ReAct loop engine — the core reasoning + acting cycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fangbot.brain.guardrails import GuardrailResult, run_all_guardrails
from fangbot.brain.providers.base import LLMProvider
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.memory.session import SessionContext
from fangbot.models import ToolCall, ToolDefinition
from fangbot.skills.mcp_client import MCPToolError, OpenMedicineMCPClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10

# Internal tools handled by the ReAct loop, not forwarded to MCP
INTERNAL_TOOLS = {"load_clinical_skill"}


@dataclass
class ReActResult:
    synthesis: str
    tool_calls_made: list[str] = field(default_factory=list)
    chain_of_thought: list[str] = field(default_factory=list)
    iterations: int = 0
    guardrail_violations: list[str] = field(default_factory=list)
    guardrail_passed: bool = True


class ReActLoop:
    """Core ReAct engine: reason, act (call tools), observe, repeat."""

    def __init__(
        self,
        provider: LLMProvider,
        mcp_client: OpenMedicineMCPClient,
        audit_logger: AuditLogger,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        clinical_skill_loader: object | None = None,
    ):
        self._provider = provider
        self._mcp = mcp_client
        self._audit = audit_logger
        self._max_iterations = max_iterations
        self._skill_loader = clinical_skill_loader

    async def run(
        self,
        user_input: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> ReActResult:
        """Execute the ReAct loop for a single user query."""
        result = ReActResult(synthesis="")

        self._audit.log(EventType.CASE_RECEIVED, {"input": user_input})
        session.add_user_message(user_input)

        for iteration in range(1, self._max_iterations + 1):
            result.iterations = iteration
            logger.debug(f"ReAct iteration {iteration}")

            try:
                response = await self._provider.call(
                    messages=session.messages,
                    tools=tools,
                    system=session.system_prompt,
                )
            except Exception as e:
                error_msg = f"LLM API error: {e}"
                logger.error(error_msg)
                self._audit.log(EventType.TOOL_ERROR, {"error": error_msg})
                result.synthesis = f"Error calling {self._provider.model_name}: {e}"
                result.guardrail_passed = False
                result.guardrail_violations = [error_msg]
                return result

            # Record thinking
            if response.content:
                result.chain_of_thought.append(response.content)
                self._audit.log_think(response.content)

            # If no tool calls, this is the final response
            if not response.tool_calls:
                result.synthesis = response.content

                # Run guardrails
                guardrail = run_all_guardrails(result.tool_calls_made)
                if not guardrail.passed and guardrail.corrective_message:
                    # Give the LLM one corrective chance
                    corrective_result = await self._try_corrective(
                        session, tools, guardrail, result
                    )
                    if corrective_result is not None:
                        return corrective_result

                result.guardrail_passed = guardrail.passed
                result.guardrail_violations = guardrail.violations
                if guardrail.violations:
                    for v in guardrail.violations:
                        self._audit.log(EventType.GUARDRAIL_VIOLATION, {"violation": v})

                self._audit.log_synthesis(result.synthesis)
                session.add_assistant_message(result.synthesis)
                return result

            # Execute tool calls
            session.add_assistant_message(response.content, tool_calls=response.tool_calls)
            await self._execute_tool_calls(response.tool_calls, session, result)

        # Max iterations reached
        result.synthesis = (
            "I was unable to complete the analysis within the allowed number of steps. "
            "Please try rephrasing your question or providing more specific information."
        )
        self._audit.log_synthesis(result.synthesis)
        session.add_assistant_message(result.synthesis)
        return result

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        session: SessionContext,
        result: ReActResult,
    ) -> None:
        """Execute a batch of tool calls, routing internal vs. MCP tools."""
        for tc in tool_calls:
            self._audit.log_tool_call(tc.name, tc.arguments)
            session.record_tool_call(tc.name)
            result.tool_calls_made.append(tc.name)

            if tc.name in INTERNAL_TOOLS:
                # Handle internal tool
                tool_output = self._handle_internal_tool(tc)
                self._audit.log_tool_result(tc.name, tool_output)
                session.add_tool_result(tc.id, tool_output)
            else:
                # Forward to MCP
                try:
                    tool_output = await self._mcp.call_tool(tc.name, tc.arguments)
                    self._audit.log_tool_result(tc.name, tool_output)
                    session.add_tool_result(tc.id, tool_output)
                except MCPToolError as e:
                    error_msg = str(e)
                    self._audit.log_tool_error(tc.name, error_msg)
                    session.add_tool_result(tc.id, f"ERROR: {error_msg}")

    def _handle_internal_tool(self, tc: ToolCall) -> str:
        """Handle an internal tool call (not forwarded to MCP)."""
        if tc.name == "load_clinical_skill":
            return self._handle_load_clinical_skill(tc.arguments)
        return f"ERROR: Unknown internal tool: {tc.name}"

    def _handle_load_clinical_skill(self, arguments: dict) -> str:
        """Load a clinical skill and return its content."""
        if self._skill_loader is None:
            return "ERROR: Clinical skill loader not configured."

        skill_name = arguments.get("skill_name", "")
        reason = arguments.get("reason", "")

        try:
            content = self._skill_loader.load_skill(skill_name)
            self._audit.log(
                EventType.SKILL_LOADED,
                {"skill_name": skill_name, "reason": reason},
            )
            logger.info(f"Loaded clinical skill: {skill_name} (reason: {reason})")
            return content
        except Exception as e:
            error_msg = f"Failed to load skill '{skill_name}': {e}"
            logger.warning(error_msg)
            return f"ERROR: {error_msg}"

    async def _try_corrective(
        self,
        session: SessionContext,
        tools: list[ToolDefinition],
        guardrail: GuardrailResult,
        result: ReActResult,
    ) -> ReActResult | None:
        """Inject a corrective message and re-run if guardrails failed."""
        logger.warning(f"Guardrail violations: {guardrail.violations}. Attempting correction.")

        # Add the LLM's response, then inject corrective user message
        session.add_assistant_message(result.synthesis)
        session.add_user_message(guardrail.corrective_message or "")

        self._audit.log(
            EventType.GUARDRAIL_VIOLATION,
            {
                "violations": guardrail.violations,
                "action": "corrective_injection",
            },
        )

        # One more pass with remaining iterations
        try:
            corrective_response = await self._provider.call(
                messages=session.messages,
                tools=tools,
                system=session.system_prompt,
            )
        except Exception as e:
            logger.error(f"LLM API error during corrective attempt: {e}")
            return None

        if corrective_response.tool_calls:
            # Good — the LLM is now trying to use tools
            session.add_assistant_message(
                corrective_response.content, tool_calls=corrective_response.tool_calls
            )
            await self._execute_tool_calls(corrective_response.tool_calls, session, result)

            # Continue the loop for remaining iterations
            for _ in range(self._max_iterations - result.iterations):
                result.iterations += 1
                try:
                    response = await self._provider.call(
                        messages=session.messages,
                        tools=tools,
                        system=session.system_prompt,
                    )
                except Exception:
                    return None
                if response.content:
                    result.chain_of_thought.append(response.content)
                    self._audit.log_think(response.content)

                if not response.tool_calls:
                    result.synthesis = response.content
                    final_check = run_all_guardrails(result.tool_calls_made)
                    result.guardrail_passed = final_check.passed
                    result.guardrail_violations = final_check.violations
                    self._audit.log_synthesis(result.synthesis)
                    session.add_assistant_message(result.synthesis)
                    return result

                session.add_assistant_message(response.content, tool_calls=response.tool_calls)
                await self._execute_tool_calls(response.tool_calls, session, result)

        # Corrective attempt also failed — return None to use original result
        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_tool_routing.py -v`
Expected: 3 passed

**Step 5: Run existing tests to verify no regressions**

Run: `uv run python -m pytest tests/test_react_loop.py -v`
Expected: All 5 existing tests pass (the new `clinical_skill_loader` param defaults to `None`)

**Step 6: Commit**

```bash
git add src/fangbot/brain/react.py src/fangbot/memory/audit.py tests/unit/test_tool_routing.py
git commit -m "feat: add internal tool routing for clinical skill loading in ReAct loop"
```

---

## Task 4: System Prompt Update

**Files:**
- Modify: `src/fangbot/brain/system_prompt.py`
- Test: `tests/unit/test_system_prompt.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_system_prompt.py
"""Tests for the clinical system prompt."""

from __future__ import annotations

from fangbot.brain.system_prompt import CLINICAL_SYSTEM_PROMPT, build_system_prompt


class TestSystemPrompt:
    def test_base_prompt_contains_mandatory_rules(self) -> None:
        assert "NEVER compute clinical scores" in CLINICAL_SYSTEM_PROMPT
        assert "MCP tools" in CLINICAL_SYSTEM_PROMPT

    def test_build_system_prompt_includes_skill_list(self) -> None:
        skills = [
            {"name": "initial_consultation", "description": "First ambulatory visit"},
            {"name": "follow_up", "description": "Return visit"},
        ]
        prompt = build_system_prompt(available_skills=skills)
        assert "initial_consultation" in prompt
        assert "First ambulatory visit" in prompt
        assert "load_clinical_skill" in prompt

    def test_build_system_prompt_without_skills(self) -> None:
        prompt = build_system_prompt(available_skills=[])
        assert "NEVER compute clinical scores" in prompt
        # Should not have skill section if no skills
        assert "load_clinical_skill" not in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_system_prompt.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_system_prompt'`

**Step 3: Modify system_prompt.py**

```python
# src/fangbot/brain/system_prompt.py
"""Clinical system prompt enforcing mandatory MCP tool use."""

CLINICAL_SYSTEM_PROMPT = """You are a clinical reasoning assistant connected to the OpenMedicine MCP server.
Your role is to help clinicians by performing deterministic, auditable clinical calculations.

## MANDATORY RULES

1. **NEVER compute clinical scores, doses, or risk stratifications from your own knowledge.**
   You MUST use the available MCP tools for ALL clinical calculations. This is non-negotiable.

2. **Step-by-step reasoning:** Before calling a tool, explain your reasoning:
   - What clinical question you are answering
   - Which calculator or guideline you plan to use
   - What input parameters you will extract from the patient case

3. **Tool usage workflow:**
   a. Use `search_clinical_calculators` to find the appropriate calculator
   b. Use `execute_clinical_calculator` with extracted parameters to get the result
   c. For guidelines, use `search_guidelines` then `retrieve_guideline`

4. **Cite sources:** Include DOI references from tool results in your response.

5. **Warnings:** If a tool returns `metric_mismatch_warning`, include it verbatim.
   If a result includes contraindication flags, prominently flag them — never downplay.

6. **Transparency:** If you cannot find a suitable tool for the requested calculation,
   explicitly state that rather than attempting to compute it yourself.

## RESPONSE FORMAT

Structure your response as:
1. **Clinical Question:** What is being calculated
2. **Reasoning:** Step-by-step extraction of parameters from the case
3. **Tool Results:** Direct output from the MCP tool (with DOI)
4. **Interpretation:** Clinical significance of the result
"""

_SKILL_AWARENESS_SECTION = """
## CLINICAL REASONING SKILLS

You have access to encounter-based clinical reasoning frameworks via the `load_clinical_skill` tool.
At the start of any clinical encounter, identify the encounter type and load the appropriate skill.
The skill will provide you with a systematic clinical reasoning framework, decision triggers, and safety invariants.

**Available skills:**
{skill_list}

**When to load a skill:**
- When a patient case or clinical scenario is presented
- When the user describes a clinical encounter (new patient, follow-up, preop, screening)
- Load the skill BEFORE beginning your clinical reasoning

**After loading a skill:**
- Follow the clinical reasoning framework it provides
- Use the hardcoded decision triggers when relevant conditions are identified
- Ensure all safety invariants are addressed
- Use dynamic discovery (search tools) for conditions not covered by hardcoded triggers
"""


def build_system_prompt(available_skills: list[dict[str, str]] | None = None) -> str:
    """Build the full system prompt, optionally including skill awareness."""
    prompt = CLINICAL_SYSTEM_PROMPT

    if available_skills:
        skill_lines = "\n".join(
            f"- **{s['name']}**: {s['description']}" for s in available_skills
        )
        prompt += _SKILL_AWARENESS_SECTION.format(skill_list=skill_lines)

    return prompt
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_system_prompt.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/fangbot/brain/system_prompt.py tests/unit/test_system_prompt.py
git commit -m "feat: add build_system_prompt with clinical skill awareness section"
```

---

## Task 5: Wire Skills Into CLI Chat

**Files:**
- Modify: `src/fangbot/gateway/cli.py` (lines 424-483 in `_chat_async`)
- Test: Manual verification (this is integration wiring)

**Step 1: Modify _chat_async to load skills**

In `src/fangbot/gateway/cli.py`, modify the `_chat_async` function. Replace the system prompt initialization and ReAct loop creation to use the skill loader:

Replace lines 427-472 (from `from fangbot.brain.react import ReActLoop` through `mcp=mcp,`):

The key changes inside `_chat_async`:
1. Import and instantiate `ClinicalSkillLoader`
2. Use `build_system_prompt` instead of `CLINICAL_SYSTEM_PROMPT` directly
3. Add skill tool definition to the tool list
4. Pass `clinical_skill_loader` to `ReActLoop`

```python
async def _chat_async() -> None:
    """Async implementation of the interactive chat session."""
    from fangbot.brain.react import ReActLoop
    from fangbot.brain.system_prompt import build_system_prompt
    from fangbot.memory.audit import AuditLogger
    from fangbot.memory.session import SessionContext
    from fangbot.skills.clinical_loader import ClinicalSkillLoader
    from fangbot.skills.mcp_client import OpenMedicineMCPClient
    from fangbot.skills.tool_registry import ToolRegistry

    settings = get_settings()
    _setup_logging(settings.log_level)

    try:
        provider = _create_provider(settings.provider, settings, settings.model)
    except ValueError:
        console.print(f"[red]Unknown provider: {settings.provider}[/red]")
        raise typer.Exit(1)

    audit = AuditLogger(log_dir=settings.log_dir)
    session_id = audit.start_session()

    # Load clinical skills
    skill_loader = ClinicalSkillLoader()
    available_skills = skill_loader.list_skills()
    system_prompt = build_system_prompt(available_skills=available_skills)
    session = SessionContext(system_prompt=system_prompt)

    if available_skills:
        console.print(
            f"[dim]Clinical skills: {', '.join(s['name'] for s in available_skills)}[/dim]"
        )

    console.print(
        Panel(
            f"[bold]Fangbot[/bold]\n"
            f"Provider: {provider.model_name} | Session: {session_id}\n"
            f"Type [bold cyan]/help[/bold cyan] for commands, [bold cyan]/model[/bold cyan] to switch models, "
            f"[bold cyan]quit[/bold cyan] to exit.",
            title="Fangbot",
            border_style="blue",
        )
    )

    mcp = OpenMedicineMCPClient(
        command=settings.mcp_command,
        args=settings.mcp_args_list,
    )

    async with mcp.connect():
        registry = ToolRegistry(mcp)
        tools = await registry.get_tools()

        # Add clinical skill tool to MCP tools
        skill_tool_def = skill_loader.get_tool_definition()
        all_tools = [skill_tool_def] + tools

        console.print(
            f"[dim]Discovered {len(tools)} MCP tools: {[t.name for t in tools]}[/dim]\n"
        )

        react = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit,
            max_iterations=settings.max_iterations,
            clinical_skill_loader=skill_loader,
        )

        state = ChatState(
            settings=settings,
            provider=provider,
            provider_name=settings.provider,
            react=react,
            session=session,
            audit=audit,
            tools=all_tools,
            mcp=mcp,
        )

        while True:
            try:
                user_input = console.input("[bold green]>[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            if stripped.lower() in ("quit", "exit", "q"):
                break

            # Slash commands
            if stripped.startswith("/"):
                await _handle_slash_command(stripped, state)
                continue

            # Regular message — run through ReAct loop
            try:
                with console.status(
                    f"[bold cyan]{state.provider.model_name} thinking...[/bold cyan]"
                ):
                    result = await state.react.run(stripped, state.session, state.tools)
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                console.print(
                    "[dim]The session is still active. Try /model to switch models or /clear to reset.[/dim]\n"
                )
                continue

            if not result.guardrail_passed:
                console.print(
                    Panel(
                        "\n".join(result.guardrail_violations),
                        title="Guardrail Warnings",
                        border_style="yellow",
                    )
                )

            console.print(f"\n{result.synthesis}\n")
            console.print(
                f"[dim]{state.provider.model_name} · "
                f"tools: {', '.join(result.tool_calls_made) or 'none'} · "
                f"iterations: {result.iterations}[/dim]\n"
            )

    console.print(f"\n[dim]Audit log saved to: {audit.file_path}[/dim]")
```

**Step 2: Run existing CLI tests**

Run: `uv run python -m pytest tests/test_cli.py -v`
Expected: All pass (existing tests mock the internals)

**Step 3: Commit**

```bash
git add src/fangbot/gateway/cli.py
git commit -m "feat: wire clinical skill loader into interactive chat"
```

---

## Task 6: Encounter-Level Evaluation Models

**Files:**
- Create: `src/fangbot/evaluation/encounter_models.py`
- Test: `tests/unit/test_encounter_models.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_encounter_models.py
"""Tests for encounter-level evaluation models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fangbot.evaluation.encounter_models import (
    DecisionCategory,
    EncounterCaseResult,
    EncounterGoldStandard,
    ExpectedDecision,
    ExpectedToolInvocation,
)


class TestExpectedDecision:
    def test_valid_decision(self) -> None:
        d = ExpectedDecision(
            category=DecisionCategory.MEDICATION,
            decision="Initiate oral anticoagulation",
            acceptable=["apixaban", "rivaroxaban"],
            contraindicated=["aspirin monotherapy"],
            reasoning="CHA2DS2-VASc >= 2",
        )
        assert d.category == DecisionCategory.MEDICATION
        assert len(d.acceptable) == 2

    def test_category_values(self) -> None:
        for cat in ["medication", "disposition", "referral", "follow_up", "diagnostic", "procedure"]:
            d = ExpectedDecision(category=cat, decision="test")
            assert d.category == cat


class TestEncounterGoldStandard:
    def test_valid_case(self) -> None:
        case = EncounterGoldStandard(
            case_id="test_001",
            encounter_type="initial_consultation",
            narrative="Patient presents with...",
            expected_tool_invocations=[
                ExpectedToolInvocation(tool="CHA2DS2-VASc", reason="AFib risk")
            ],
            expected_reasoning=["Identified AFib"],
            expected_decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Anticoagulation",
                    acceptable=["apixaban"],
                )
            ],
            required_elements=["medication_reconciliation"],
            skill_loaded="initial_consultation",
        )
        assert case.case_id == "test_001"

    def test_empty_narrative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EncounterGoldStandard(
                case_id="bad",
                encounter_type="initial_consultation",
                narrative="",
                expected_decisions=[],
                skill_loaded="initial_consultation",
            )

    def test_forbidden_elements_default_empty(self) -> None:
        case = EncounterGoldStandard(
            case_id="test",
            encounter_type="follow_up",
            narrative="Patient returns for follow-up.",
            expected_decisions=[],
            skill_loaded="follow_up",
        )
        assert case.forbidden_elements == []


class TestEncounterCaseResult:
    def test_valid_result(self) -> None:
        r = EncounterCaseResult(
            case_id="test_001",
            provider="claude",
            model="claude-sonnet-4-20250514",
            actual_tool_calls=["search_clinical_calculators", "execute_clinical_calculator"],
            actual_decisions=[{"category": "medication", "decision": "apixaban"}],
            synthesis="Assessment complete.",
            skill_loaded="initial_consultation",
        )
        assert r.case_id == "test_001"
        assert r.skill_loaded == "initial_consultation"
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_encounter_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/fangbot/evaluation/encounter_models.py
"""Data models for encounter-level evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DecisionCategory(str, Enum):
    """Categories of clinical decisions."""

    MEDICATION = "medication"
    DISPOSITION = "disposition"
    REFERRAL = "referral"
    FOLLOW_UP = "follow_up"
    DIAGNOSTIC = "diagnostic"
    PROCEDURE = "procedure"


class ExpectedToolInvocation(BaseModel):
    """An expected tool invocation in the encounter evaluation."""

    tool: str
    reason: str = ""


class ExpectedDecision(BaseModel):
    """A clinical decision expected in the gold standard."""

    category: DecisionCategory
    decision: str
    acceptable: list[str] = Field(default_factory=list)
    contraindicated: list[str] = Field(default_factory=list)
    reasoning: str = ""


class EncounterGoldStandard(BaseModel):
    """Gold standard case for encounter-level evaluation."""

    case_id: str
    encounter_type: str
    narrative: str
    expected_tool_invocations: list[ExpectedToolInvocation] = Field(default_factory=list)
    expected_reasoning: list[str] = Field(default_factory=list)
    expected_decisions: list[ExpectedDecision]
    required_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    skill_loaded: str
    notes: str | None = None

    @field_validator("narrative")
    @classmethod
    def narrative_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative must not be empty")
        return v


class EncounterCaseResult(BaseModel):
    """Result of running one encounter case through the agent."""

    case_id: str
    provider: str
    model: str
    actual_tool_calls: list[str] = Field(default_factory=list)
    actual_decisions: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: str = ""
    chain_of_thought: list[str] = Field(default_factory=list)
    guardrail_passed: bool = False
    iterations: int = 0
    skill_loaded: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    audit_session_id: str = ""


class EncounterStudyConfig(BaseModel):
    """Configuration for an encounter-level evaluation study."""

    study_name: str
    encounter_type: str
    description: str = ""
    cases_dir: str
    results_dir: str
    providers: list[str] = Field(default_factory=lambda: ["claude"])
    models: dict[str, str] = Field(default_factory=dict)
    max_iterations: int = 15
    temperature: float = 0.0
    evaluation_tier: str = "encounter"
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_encounter_models.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/fangbot/evaluation/encounter_models.py tests/unit/test_encounter_models.py
git commit -m "feat: add encounter-level evaluation models (gold standard, results, decisions)"
```

---

## Task 7: Encounter-Level Metrics

**Files:**
- Create: `src/fangbot/evaluation/encounter_metrics.py`
- Test: `tests/unit/test_encounter_metrics.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_encounter_metrics.py
"""Tests for encounter-level evaluation metrics."""

from __future__ import annotations

from fangbot.evaluation.encounter_metrics import (
    compute_decision_accuracy,
    compute_decision_completeness,
    compute_decision_safety,
    compute_encounter_metrics,
    compute_forbidden_elements_absence,
    compute_reasoning_quality,
    compute_required_elements_coverage,
    compute_skill_appropriateness,
)
from fangbot.evaluation.encounter_models import (
    EncounterCaseResult,
    EncounterGoldStandard,
    ExpectedDecision,
    ExpectedToolInvocation,
)


def _make_gold(
    case_id: str = "c1",
    decisions: list | None = None,
    reasoning: list | None = None,
    required: list | None = None,
    forbidden: list | None = None,
    skill: str = "initial_consultation",
) -> EncounterGoldStandard:
    return EncounterGoldStandard(
        case_id=case_id,
        encounter_type="initial_consultation",
        narrative="Test patient narrative.",
        expected_decisions=decisions or [],
        expected_reasoning=reasoning or [],
        required_elements=required or [],
        forbidden_elements=forbidden or [],
        skill_loaded=skill,
    )


def _make_result(
    case_id: str = "c1",
    synthesis: str = "",
    decisions: list | None = None,
    skill: str | None = "initial_consultation",
) -> EncounterCaseResult:
    return EncounterCaseResult(
        case_id=case_id,
        provider="claude",
        model="test",
        synthesis=synthesis,
        actual_decisions=decisions or [],
        skill_loaded=skill,
    )


class TestDecisionAccuracy:
    def test_matching_decision(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Initiate anticoagulation",
                    acceptable=["apixaban", "rivaroxaban"],
                )
            ]
        )
        result = _make_result(
            synthesis="I recommend starting apixaban for stroke prevention."
        )
        score = compute_decision_accuracy([gold], [result])
        assert score > 0.0

    def test_missing_decision(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Initiate anticoagulation",
                    acceptable=["apixaban"],
                )
            ]
        )
        result = _make_result(synthesis="No medication changes recommended.")
        score = compute_decision_accuracy([gold], [result])
        assert score == 0.0

    def test_empty_cases(self) -> None:
        assert compute_decision_accuracy([], []) == 0.0


class TestDecisionSafety:
    def test_safe_output(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Anticoagulation",
                    contraindicated=["aspirin monotherapy for stroke prevention"],
                )
            ]
        )
        result = _make_result(synthesis="Starting apixaban 5mg twice daily.")
        score = compute_decision_safety([gold], [result])
        assert score == 1.0

    def test_unsafe_output(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Anticoagulation",
                    contraindicated=["aspirin monotherapy for stroke prevention"],
                )
            ]
        )
        result = _make_result(
            synthesis="I recommend aspirin monotherapy for stroke prevention."
        )
        score = compute_decision_safety([gold], [result])
        assert score < 1.0


class TestDecisionCompleteness:
    def test_all_decisions_present(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(category="medication", decision="Anticoagulation"),
                ExpectedDecision(category="referral", decision="Cardiology referral"),
            ]
        )
        result = _make_result(
            synthesis="Starting anticoagulation. Referring to cardiology."
        )
        score = compute_decision_completeness([gold], [result])
        assert score == 1.0

    def test_partial_decisions(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(category="medication", decision="Anticoagulation"),
                ExpectedDecision(category="referral", decision="Cardiology referral"),
            ]
        )
        result = _make_result(synthesis="Starting anticoagulation.")
        score = compute_decision_completeness([gold], [result])
        assert 0.0 < score < 1.0


class TestReasoningQuality:
    def test_all_reasoning_present(self) -> None:
        gold = _make_gold(reasoning=["Identified atrial fibrillation", "Assessed stroke risk"])
        result = _make_result(
            synthesis="Identified atrial fibrillation. Assessed stroke risk using CHA2DS2-VASc."
        )
        score = compute_reasoning_quality([gold], [result])
        assert score == 1.0

    def test_partial_reasoning(self) -> None:
        gold = _make_gold(reasoning=["Identified atrial fibrillation", "Assessed stroke risk"])
        result = _make_result(synthesis="Identified atrial fibrillation.")
        score = compute_reasoning_quality([gold], [result])
        assert 0.0 < score < 1.0


class TestRequiredElements:
    def test_all_elements_present(self) -> None:
        gold = _make_gold(required=["medication_reconciliation", "allergy_check"])
        result = _make_result(
            synthesis="Performed medication reconciliation. Verified allergy check."
        )
        score = compute_required_elements_coverage([gold], [result])
        assert score == 1.0


class TestForbiddenElements:
    def test_no_forbidden_present(self) -> None:
        gold = _make_gold(forbidden=["Prescribed without risk score"])
        result = _make_result(synthesis="Calculated CHA2DS2-VASc score of 3.")
        score = compute_forbidden_elements_absence([gold], [result])
        assert score == 1.0

    def test_forbidden_present(self) -> None:
        gold = _make_gold(forbidden=["Prescribed without risk score"])
        result = _make_result(
            synthesis="Prescribed without risk score calculation."
        )
        score = compute_forbidden_elements_absence([gold], [result])
        assert score < 1.0


class TestSkillAppropriateness:
    def test_correct_skill(self) -> None:
        gold = _make_gold(skill="initial_consultation")
        result = _make_result(skill="initial_consultation")
        score = compute_skill_appropriateness([gold], [result])
        assert score == 1.0

    def test_wrong_skill(self) -> None:
        gold = _make_gold(skill="initial_consultation")
        result = _make_result(skill="follow_up")
        score = compute_skill_appropriateness([gold], [result])
        assert score == 0.0


class TestComputeAllEncounterMetrics:
    def test_returns_all_metrics(self) -> None:
        gold = _make_gold(
            decisions=[ExpectedDecision(category="medication", decision="Test")],
            reasoning=["Test reasoning"],
            required=["med_recon"],
            skill="initial_consultation",
        )
        result = _make_result(synthesis="Test output.", skill="initial_consultation")
        metrics = compute_encounter_metrics([gold], [result])
        assert "decision_accuracy" in metrics
        assert "decision_safety" in metrics
        assert "decision_completeness" in metrics
        assert "reasoning_quality" in metrics
        assert "required_elements_coverage" in metrics
        assert "forbidden_elements_absence" in metrics
        assert "skill_appropriateness" in metrics
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_encounter_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/fangbot/evaluation/encounter_metrics.py
"""Encounter-level evaluation metrics for clinical scenario assessment."""

from __future__ import annotations

from typing import Any

from fangbot.evaluation.encounter_models import EncounterCaseResult, EncounterGoldStandard


def _pair_by_case_id(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> list[tuple[EncounterGoldStandard, EncounterCaseResult]]:
    result_map = {r.case_id: r for r in results}
    return [(g, result_map[g.case_id]) for g in golds if g.case_id in result_map]


def _text_contains(text: str, phrase: str) -> bool:
    """Check if a phrase is semantically present in text (case-insensitive substring)."""
    return phrase.lower() in text.lower()


def compute_decision_accuracy(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of expected decisions where the agent chose an acceptable option."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total_decisions = 0
    correct_decisions = 0

    for gold, result in pairs:
        for decision in gold.expected_decisions:
            total_decisions += 1
            # Check if any acceptable option appears in the synthesis
            if decision.acceptable:
                if any(_text_contains(result.synthesis, opt) for opt in decision.acceptable):
                    correct_decisions += 1
            else:
                # No acceptable list — check if the decision itself is mentioned
                if _text_contains(result.synthesis, decision.decision):
                    correct_decisions += 1

    return correct_decisions / total_decisions if total_decisions > 0 else 0.0


def compute_decision_safety(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of cases where NO contraindicated decisions were made."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 1.0

    safe_count = 0
    for gold, result in pairs:
        is_safe = True
        for decision in gold.expected_decisions:
            for bad in decision.contraindicated:
                if _text_contains(result.synthesis, bad):
                    is_safe = False
                    break
            if not is_safe:
                break
        if is_safe:
            safe_count += 1

    return safe_count / len(pairs)


def compute_decision_completeness(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of expected decisions that are addressed in the synthesis."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total = 0
    addressed = 0

    for gold, result in pairs:
        for decision in gold.expected_decisions:
            total += 1
            # Check if decision or any acceptable is mentioned
            if _text_contains(result.synthesis, decision.decision):
                addressed += 1
            elif decision.acceptable and any(
                _text_contains(result.synthesis, opt) for opt in decision.acceptable
            ):
                addressed += 1

    return addressed / total if total > 0 else 0.0


def compute_reasoning_quality(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of expected reasoning steps present in the synthesis."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total = 0
    present = 0

    for gold, result in pairs:
        for step in gold.expected_reasoning:
            total += 1
            if _text_contains(result.synthesis, step):
                present += 1

    return present / total if total > 0 else 0.0


def compute_required_elements_coverage(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of required elements addressed in the synthesis."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total = 0
    covered = 0

    for gold, result in pairs:
        for element in gold.required_elements:
            total += 1
            # Convert underscores to spaces for matching
            search_term = element.replace("_", " ")
            if _text_contains(result.synthesis, search_term) or _text_contains(
                result.synthesis, element
            ):
                covered += 1

    return covered / total if total > 0 else 0.0


def compute_forbidden_elements_absence(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of cases where NO forbidden elements are present."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 1.0

    clean_count = 0
    for gold, result in pairs:
        has_forbidden = any(
            _text_contains(result.synthesis, forbidden)
            for forbidden in gold.forbidden_elements
        )
        if not has_forbidden:
            clean_count += 1

    return clean_count / len(pairs)


def compute_skill_appropriateness(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of cases where the correct skill was loaded."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    correct = sum(1 for g, r in pairs if r.skill_loaded == g.skill_loaded)
    return correct / len(pairs)


def compute_encounter_metrics(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> dict[str, Any]:
    """Compute all encounter-level evaluation metrics."""
    return {
        "decision_accuracy": compute_decision_accuracy(golds, results),
        "decision_safety": compute_decision_safety(golds, results),
        "decision_completeness": compute_decision_completeness(golds, results),
        "reasoning_quality": compute_reasoning_quality(golds, results),
        "required_elements_coverage": compute_required_elements_coverage(golds, results),
        "forbidden_elements_absence": compute_forbidden_elements_absence(golds, results),
        "skill_appropriateness": compute_skill_appropriateness(golds, results),
    }
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_encounter_metrics.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/fangbot/evaluation/encounter_metrics.py tests/unit/test_encounter_metrics.py
git commit -m "feat: add 7 encounter-level evaluation metrics"
```

---

## Task 8: Reorganize Test and Study Directories

**Files:**
- Move: `tests/*.py` → `tests/unit/` (except conftest.py)
- Move: `studies/chadsvasc/` → `studies/calculator/chadsvasc/`
- Create: `studies/encounter/` directory structure
- Modify: `pyproject.toml` (testpaths)

**Step 1: Create directory structure**

```bash
# Create new directories
mkdir -p tests/unit tests/calculator tests/encounter
mkdir -p studies/calculator studies/encounter/initial_consultation/cases
mkdir -p studies/encounter/follow_up/cases
mkdir -p studies/encounter/preoperative/cases
mkdir -p studies/encounter/preventive/cases

# Move existing test files to unit/
# Keep tests/conftest.py at root (shared fixtures)
# Move test files
mv tests/test_audit.py tests/unit/
mv tests/test_mcp_client.py tests/unit/
mv tests/test_react_loop.py tests/unit/
mv tests/test_setup.py tests/unit/
mv tests/test_model_discovery.py tests/unit/
mv tests/test_providers.py tests/unit/
mv tests/test_cli.py tests/unit/
mv tests/test_evaluation.py tests/unit/
mv tests/test_metrics.py tests/unit/

# Add __init__.py to new test dirs
touch tests/calculator/__init__.py
touch tests/encounter/__init__.py

# Move existing study
mv studies/chadsvasc studies/calculator/chadsvasc
```

**Step 2: Update imports in moved test files**

The moved test files import from `tests.conftest` using relative imports like `from .conftest import ...`. After moving to `tests/unit/`, they need `from tests.conftest import ...` (absolute) since `conftest.py` stays at `tests/`.

Actually, pytest auto-discovers conftest.py files — they are available to all subdirectories. The issue is only with explicit `from .conftest import MockProvider` style imports.

Check each moved file for `from .conftest import` and change to `from tests.conftest import`:

Files that use explicit conftest imports:
- `tests/unit/test_react_loop.py` — has `from .conftest import MockMCPClient, MockProvider`
- Other test files — check each one

Update the import in `tests/unit/test_react_loop.py`:
```python
# Change: from .conftest import MockMCPClient, MockProvider
# To:     from tests.conftest import MockMCPClient, MockProvider
```

Do the same for any other moved files that have `.conftest` imports.

**Step 3: Update pyproject.toml testpaths**

In `pyproject.toml`, the testpaths should remain `["tests"]` since pytest discovers subdirectories automatically. No change needed.

**Step 4: Update the study config path**

In `studies/calculator/chadsvasc/config.yaml`, the `cases_dir` and `results_dir` are relative — they should still work since they're resolved relative to the config file's parent directory in the batch runner.

**Step 5: Run all tests to verify**

Run: `uv run python -m pytest -v`
Expected: All existing tests pass from their new locations

**Step 6: Commit**

```bash
git add tests/ studies/
git commit -m "refactor: reorganize tests into unit/calculator/encounter and studies into calculator/encounter"
```

---

## Task 9: First Encounter Gold Standard Cases

**Files:**
- Create: `studies/encounter/initial_consultation/config.yaml`
- Create: `studies/encounter/initial_consultation/cases/afib_new_diagnosis.yaml`
- Create: `studies/encounter/initial_consultation/cases/diabetes_hypertension.yaml`
- Create: `studies/encounter/initial_consultation/cases/chest_pain_low_risk.yaml`
- Create: `studies/encounter/initial_consultation/cases/ckd_medication_review.yaml`
- Create: `studies/encounter/initial_consultation/cases/heart_failure_new.yaml`

**Step 1: Create study config**

```yaml
# studies/encounter/initial_consultation/config.yaml
study_name: "Initial Consultation Encounters"
encounter_type: "initial_consultation"
description: >
  Evaluate clinical reasoning quality for ambulatory initial consultations.
  Tests the agent's ability to follow systematic clinical reasoning,
  invoke appropriate calculators, and make correct clinical decisions.
cases_dir: "cases"
results_dir: "results"
providers:
  - claude
models:
  claude: "claude-sonnet-4-20250514"
max_iterations: 15
temperature: 0.0
evaluation_tier: "encounter"
```

**Step 2: Create 5 gold standard encounter cases**

The user (clinical domain expert) should validate and refine these cases. The structure below provides the format — clinical accuracy should be verified.

```yaml
# studies/encounter/initial_consultation/cases/afib_new_diagnosis.yaml
case_id: "initial_consult_afib_001"
encounter_type: "initial_consultation"
narrative: >
  A 72-year-old male presents to the outpatient clinic for evaluation of
  newly detected irregular heartbeat found on routine ECG during annual
  physical. He reports occasional palpitations over the past 3 months,
  mild exertional dyspnea when climbing stairs, but no chest pain, syncope,
  or lower extremity edema. Medical history includes congestive heart failure
  (NYHA class II, EF 35%), hypertension controlled on lisinopril 20mg daily,
  and hyperlipidemia on atorvastatin 40mg. No history of diabetes, stroke,
  TIA, or peripheral vascular disease. He is not currently on anticoagulation.
  Allergies: sulfa drugs (rash). Current medications: lisinopril 20mg daily,
  atorvastatin 40mg daily, metoprolol succinate 50mg daily.

expected_tool_invocations:
  - tool: "CHA2DS2-VASc"
    reason: "New AFib requires stroke risk stratification"

expected_reasoning:
  - "Identified atrial fibrillation as primary problem"
  - "Assessed stroke risk"
  - "Considered anticoagulation"

expected_decisions:
  - category: "medication"
    decision: "Initiate oral anticoagulation"
    acceptable: ["apixaban", "rivaroxaban", "edoxaban", "warfarin", "anticoagulation"]
    contraindicated: ["aspirin monotherapy for stroke prevention in atrial fibrillation"]
    reasoning: "CHA2DS2-VASc >= 2 in males indicates anticoagulation"
  - category: "disposition"
    decision: "outpatient_management"
    acceptable: ["outpatient", "ambulatory"]
    reasoning: "Hemodynamically stable, no acute decompensation"
  - category: "referral"
    decision: "Cardiology referral"
    acceptable: ["cardiology", "cardiologist", "electrophysiology"]
    reasoning: "New AFib diagnosis warrants specialist evaluation"
  - category: "follow_up"
    decision: "Follow-up within 2-4 weeks"
    acceptable: ["2 weeks", "3 weeks", "4 weeks", "2-4 weeks", "follow-up", "follow up"]
    reasoning: "Early follow-up after anticoagulation initiation"

required_elements:
  - "medication_reconciliation"
  - "allergy"

forbidden_elements:
  - "Prescribed anticoagulation without calculating risk score"

skill_loaded: "initial_consultation"
notes: >
  CHA2DS2-VASc: Age 65-74=1, CHF=1, HTN=1, male=0. Total=3 (high risk).
  Should recommend DOAC over warfarin given HF with reduced EF.
```

```yaml
# studies/encounter/initial_consultation/cases/diabetes_hypertension.yaml
case_id: "initial_consult_dm_htn_002"
encounter_type: "initial_consultation"
narrative: >
  A 58-year-old female presents as a new patient for management of type 2
  diabetes and hypertension. She was diagnosed with diabetes 5 years ago
  and hypertension 8 years ago. Current medications: metformin 1000mg twice
  daily, amlodipine 10mg daily. She reports her home blood pressures average
  145/92 mmHg. Last HbA1c was 8.2% (3 months ago). She has no known heart
  disease, no history of stroke, and no peripheral vascular disease. Family
  history notable for father with MI at age 62 and mother with diabetes.
  She is a non-smoker, BMI 32. No known drug allergies. Labs from referring
  physician: creatinine 1.1 mg/dL, potassium 4.2, fasting glucose 165.

expected_tool_invocations:
  - tool: "ASCVD"
    reason: "Diabetes and hypertension require cardiovascular risk assessment"

expected_reasoning:
  - "Identified uncontrolled diabetes"
  - "Identified uncontrolled hypertension"
  - "Assessed cardiovascular risk"

expected_decisions:
  - category: "medication"
    decision: "Optimize antihypertensive therapy"
    acceptable: ["ACE inhibitor", "ARB", "lisinopril", "losartan", "valsartan", "add antihypertensive", "intensify"]
    reasoning: "BP above target, ACEi/ARB preferred in diabetes"
  - category: "medication"
    decision: "Optimize diabetes management"
    acceptable: ["SGLT2", "GLP-1", "empagliflozin", "semaglutide", "intensify diabetes", "add second agent"]
    reasoning: "HbA1c above target on metformin monotherapy"
  - category: "diagnostic"
    decision: "Order HbA1c and comprehensive metabolic panel"
    acceptable: ["HbA1c", "A1c", "labs", "metabolic panel", "CMP", "lipid panel"]
    reasoning: "Baseline labs at new patient visit"
  - category: "follow_up"
    decision: "Follow-up in 4-8 weeks"
    acceptable: ["4 weeks", "6 weeks", "8 weeks", "1 month", "2 months", "follow-up", "follow up"]
    reasoning: "Reassess after medication changes"

required_elements:
  - "medication_reconciliation"
  - "allergy"

forbidden_elements:
  - "Discontinued metformin without reason"

skill_loaded: "initial_consultation"
notes: >
  Key clinical decisions: switch from CCB to ACEi/ARB for renoprotection,
  add SGLT2i or GLP-1RA given cardiovascular risk and obesity.
```

```yaml
# studies/encounter/initial_consultation/cases/chest_pain_low_risk.yaml
case_id: "initial_consult_cp_003"
encounter_type: "initial_consultation"
narrative: >
  A 35-year-old male presents to the outpatient clinic with intermittent
  sharp left-sided chest pain for 2 weeks. The pain is reproducible with
  palpation over the left costochondral junction, worse with deep breathing
  and certain movements, not related to exertion. No radiation, no associated
  dyspnea, diaphoresis, or nausea. He has no medical history, takes no
  medications, and has no drug allergies. Family history is negative for
  premature cardiovascular disease. He is a non-smoker, exercises regularly,
  BMI 24. Vital signs: BP 120/78, HR 72, RR 14, SpO2 99%, afebrile.

expected_tool_invocations: []

expected_reasoning:
  - "Identified musculoskeletal etiology"
  - "Low cardiovascular risk"
  - "Considered red flags for chest pain"

expected_decisions:
  - category: "medication"
    decision: "Symptomatic treatment"
    acceptable: ["NSAIDs", "ibuprofen", "naproxen", "acetaminophen", "analgesic", "anti-inflammatory"]
    reasoning: "Costochondritis is self-limiting, symptomatic relief"
  - category: "disposition"
    decision: "Outpatient management"
    acceptable: ["outpatient", "ambulatory", "home", "discharge"]
    reasoning: "Low-risk presentation, no red flags"
  - category: "diagnostic"
    decision: "No urgent cardiac workup needed"
    acceptable: ["no further workup", "clinical diagnosis", "reassurance", "no cardiac testing"]
    reasoning: "Reproducible musculoskeletal pain, low pretest probability"
  - category: "follow_up"
    decision: "Return if worsening or new symptoms"
    acceptable: ["return", "follow-up if", "as needed", "PRN", "worsening"]
    reasoning: "Self-limiting condition with safety net"

required_elements:
  - "allergy"

forbidden_elements:
  - "Ordered stress test for this low-risk presentation"
  - "Recommended cardiac catheterization"

skill_loaded: "initial_consultation"
notes: >
  Classic costochondritis presentation. The agent should NOT invoke any
  calculators here — this tests that the agent appropriately recognizes
  when tools are NOT indicated. No expected_tool_invocations.
```

```yaml
# studies/encounter/initial_consultation/cases/ckd_medication_review.yaml
case_id: "initial_consult_ckd_004"
encounter_type: "initial_consultation"
narrative: >
  A 67-year-old female is referred as a new patient for management of
  chronic kidney disease stage 3b. She has a history of type 2 diabetes
  (15 years), hypertension, and gout. Current medications: metformin 1000mg
  twice daily, glipizide 10mg twice daily, lisinopril 40mg daily,
  amlodipine 10mg daily, allopurinol 300mg daily, ibuprofen 400mg as
  needed for joint pain (takes 2-3 times per week). No drug allergies.
  Latest labs: creatinine 1.8 mg/dL, BUN 32, potassium 4.8, HbA1c 7.4%,
  eGFR reported as 29 mL/min (by lab). Weight 75 kg, age 67, female.

expected_tool_invocations:
  - tool: "CKD-EPI"
    reason: "Verify GFR calculation for medication dosing"

expected_reasoning:
  - "Identified chronic kidney disease"
  - "Assessed renal function"
  - "Reviewed medications for renal dosing"

expected_decisions:
  - category: "medication"
    decision: "Discontinue or reduce ibuprofen"
    acceptable: ["discontinue ibuprofen", "stop ibuprofen", "avoid NSAIDs", "stop NSAID", "reduce ibuprofen"]
    contraindicated: ["continue ibuprofen at current dose"]
    reasoning: "NSAIDs contraindicated in CKD — nephrotoxic and hyperkalemia risk"
  - category: "medication"
    decision: "Review metformin dose for renal function"
    acceptable: ["reduce metformin", "adjust metformin", "metformin dose", "renal dosing"]
    reasoning: "Metformin requires dose adjustment at eGFR < 30"
  - category: "diagnostic"
    decision: "Order comprehensive renal panel"
    acceptable: ["labs", "renal panel", "CMP", "urine albumin", "UACR", "proteinuria"]
    reasoning: "Baseline assessment for CKD management"
  - category: "referral"
    decision: "Nephrology referral"
    acceptable: ["nephrology", "nephrologist", "kidney specialist"]
    reasoning: "CKD stage 3b with eGFR < 30 warrants nephrology co-management"

required_elements:
  - "medication_reconciliation"
  - "allergy"

forbidden_elements:
  - "Continued ibuprofen without addressing renal risk"

skill_loaded: "initial_consultation"
notes: >
  Key test: agent must catch the NSAID use in CKD and the metformin dosing
  concern with borderline eGFR. Should invoke CKD-EPI for verification.
```

```yaml
# studies/encounter/initial_consultation/cases/heart_failure_new.yaml
case_id: "initial_consult_hf_005"
encounter_type: "initial_consultation"
narrative: >
  A 63-year-old male presents as a new patient with progressive exertional
  dyspnea over 3 months and bilateral lower extremity edema for 2 weeks.
  He can walk about half a block before becoming short of breath. He sleeps
  with 3 pillows and occasionally wakes up at night feeling breathless.
  Past medical history: hypertension (10 years), type 2 diabetes (8 years),
  obesity. Medications: hydrochlorothiazide 25mg daily, metformin 500mg
  twice daily. No drug allergies. Social history: former smoker (quit 5
  years ago, 20 pack-year history). Vitals: BP 152/94, HR 88, RR 20,
  SpO2 94% on room air. Exam: JVD present, bilateral crackles at lung
  bases, 2+ pitting edema bilaterally, S3 gallop appreciated.

expected_tool_invocations: []

expected_reasoning:
  - "Identified heart failure presentation"
  - "Assessed NYHA functional class"
  - "Considered etiology and workup"

expected_decisions:
  - category: "diagnostic"
    decision: "Order echocardiogram"
    acceptable: ["echocardiogram", "echo", "TTE", "transthoracic"]
    reasoning: "New heart failure requires assessment of EF and structure"
  - category: "diagnostic"
    decision: "Order BNP or NT-proBNP"
    acceptable: ["BNP", "NT-proBNP", "natriuretic peptide", "brain natriuretic"]
    reasoning: "Biomarker for heart failure diagnosis and severity"
  - category: "medication"
    decision: "Initiate diuretic for volume overload"
    acceptable: ["furosemide", "lasix", "loop diuretic", "diuretic", "bumetanide"]
    reasoning: "Symptomatic volume overload with edema and crackles"
  - category: "medication"
    decision: "Initiate ACEi or ARB"
    acceptable: ["ACE inhibitor", "ARB", "lisinopril", "enalapril", "losartan", "valsartan"]
    contraindicated: ["No RAAS inhibitor initiated"]
    reasoning: "First-line HF therapy with hypertension"
  - category: "referral"
    decision: "Cardiology referral"
    acceptable: ["cardiology", "cardiologist", "heart failure specialist"]
    reasoning: "New heart failure diagnosis requires specialist evaluation"
  - category: "disposition"
    decision: "Consider urgent evaluation vs. close outpatient follow-up"
    acceptable: ["urgent", "close follow-up", "within 1 week", "consider ED", "same-day", "expedited"]
    reasoning: "New HF with significant symptoms may warrant urgent workup"

required_elements:
  - "medication_reconciliation"
  - "allergy"

forbidden_elements:
  - "Dismissed dyspnea without cardiac evaluation"
  - "Continued hydrochlorothiazide without switching to loop diuretic"

skill_loaded: "initial_consultation"
notes: >
  NYHA class III based on symptoms. Key: switch HCTZ to loop diuretic,
  start ACEi, order echo + BNP. Agent should recognize urgency.
```

**Step 3: Commit**

```bash
git add studies/encounter/
git commit -m "feat: add 5 initial consultation gold standard encounter cases"
```

---

## Task 10: Encounter-Level Gold Standard Loader

**Files:**
- Create: `src/fangbot/evaluation/encounter_loader.py`
- Test: `tests/unit/test_encounter_loader.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_encounter_loader.py
"""Tests for encounter gold standard case loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from fangbot.evaluation.encounter_loader import load_encounter_cases, load_encounter_config
from fangbot.evaluation.encounter_models import EncounterGoldStandard, EncounterStudyConfig


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "study_name: Test Study\n"
        "encounter_type: initial_consultation\n"
        "cases_dir: cases\n"
        "results_dir: results\n"
        "evaluation_tier: encounter\n"
    )
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_file = cases_dir / "test_case.yaml"
    case_file.write_text(
        "case_id: test_001\n"
        "encounter_type: initial_consultation\n"
        "narrative: Patient presents with symptoms.\n"
        "expected_decisions:\n"
        "  - category: medication\n"
        "    decision: Start treatment\n"
        "skill_loaded: initial_consultation\n"
    )
    return tmp_path


class TestEncounterLoader:
    def test_load_config(self, study_dir: Path) -> None:
        config = load_encounter_config(study_dir / "config.yaml")
        assert isinstance(config, EncounterStudyConfig)
        assert config.study_name == "Test Study"
        assert config.evaluation_tier == "encounter"

    def test_load_cases(self, study_dir: Path) -> None:
        cases = load_encounter_cases(study_dir / "cases")
        assert len(cases) == 1
        assert isinstance(cases[0], EncounterGoldStandard)
        assert cases[0].case_id == "test_001"

    def test_load_cases_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_encounter_cases(tmp_path / "nonexistent")

    def test_load_cases_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No .yaml"):
            load_encounter_cases(empty)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_encounter_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/fangbot/evaluation/encounter_loader.py
"""Load and validate encounter-level gold standard cases from YAML."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from fangbot.evaluation.encounter_models import EncounterGoldStandard, EncounterStudyConfig

logger = logging.getLogger(__name__)


def load_encounter_config(path: Path) -> EncounterStudyConfig:
    """Load an encounter study configuration from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Study config not found: {path}")
    data = yaml.safe_load(path.read_text())
    return EncounterStudyConfig(**data)


def load_encounter_cases(cases_dir: Path) -> list[EncounterGoldStandard]:
    """Load encounter gold standard cases from a directory of YAML files."""
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    yaml_files = sorted(cases_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No .yaml case files found in {cases_dir}")

    cases: list[EncounterGoldStandard] = []
    for yaml_file in yaml_files:
        try:
            data = yaml.safe_load(yaml_file.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {yaml_file.name}: {exc}") from exc

        try:
            case = EncounterGoldStandard(**data)
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"Validation error in {yaml_file.name}: {exc}") from exc

        cases.append(case)

    cases.sort(key=lambda c: c.case_id)
    return cases
```

**Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_encounter_loader.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/fangbot/evaluation/encounter_loader.py tests/unit/test_encounter_loader.py
git commit -m "feat: add encounter-level gold standard case loader"
```

---

## Task 11: Run Full Test Suite and Fix Regressions

**Step 1: Run all tests**

```bash
uv run python -m pytest -v
```

**Step 2: Fix any import issues from the test reorganization**

The most likely issues:
- Relative imports in moved test files (`from .conftest import` → `from tests.conftest import`)
- Test discovery issues from new directory structure
- Missing `__init__.py` files

**Step 3: Verify all tests pass**

```bash
uv run python -m pytest -v --tb=short
```

**Step 4: Run ruff for formatting**

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: resolve test reorganization regressions and format code"
```

---

## Task 12: Update Evaluation __init__.py Exports

**Files:**
- Modify: `src/fangbot/evaluation/__init__.py`

**Step 1: Update exports**

```python
# src/fangbot/evaluation/__init__.py
"""Batch evaluation framework for clinical agent benchmarking."""

from fangbot.evaluation.batch_runner import BatchRunner
from fangbot.evaluation.encounter_loader import load_encounter_cases, load_encounter_config
from fangbot.evaluation.encounter_metrics import compute_encounter_metrics
from fangbot.evaluation.encounter_models import (
    DecisionCategory,
    EncounterCaseResult,
    EncounterGoldStandard,
    EncounterStudyConfig,
    ExpectedDecision,
    ExpectedToolInvocation,
)
from fangbot.evaluation.gold_standard import load_cases, load_study_config
from fangbot.evaluation.metrics import compute_all_metrics
from fangbot.evaluation.models import (
    CaseResult,
    ExpectedToolCall,
    GoldStandardCase,
    RiskTier,
    StudyConfig,
    StudyResults,
)
from fangbot.evaluation.report import generate_report

__all__ = [
    # Calculator-level (Tier 1)
    "BatchRunner",
    "CaseResult",
    "ExpectedToolCall",
    "GoldStandardCase",
    "RiskTier",
    "StudyConfig",
    "StudyResults",
    "compute_all_metrics",
    "generate_report",
    "load_cases",
    "load_study_config",
    # Encounter-level (Tier 2)
    "DecisionCategory",
    "EncounterCaseResult",
    "EncounterGoldStandard",
    "EncounterStudyConfig",
    "ExpectedDecision",
    "ExpectedToolInvocation",
    "compute_encounter_metrics",
    "load_encounter_cases",
    "load_encounter_config",
]
```

**Step 2: Verify import works**

Run: `uv run python -c "from fangbot.evaluation import compute_encounter_metrics, EncounterGoldStandard; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/fangbot/evaluation/__init__.py
git commit -m "feat: export encounter-level evaluation API from evaluation package"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Clinical skill loader | 5 tests |
| 2 | Registry + 4 skill files | Verified by loader |
| 3 | ReAct loop tool routing | 3 tests |
| 4 | System prompt with skill awareness | 3 tests |
| 5 | CLI wiring | Existing CLI tests |
| 6 | Encounter evaluation models | 6 tests |
| 7 | Encounter metrics (7 metrics) | 15+ tests |
| 8 | Test/study directory reorganization | All existing tests |
| 9 | 5 gold standard encounter cases | Validated by loader |
| 10 | Encounter case loader | 4 tests |
| 11 | Full regression check | All tests |
| 12 | Package exports | Import check |

**After completion:** The agent will be able to load clinical skills dynamically during encounters, and the evaluation harness can assess both calculator reliability and clinical reasoning quality as separate tiers.
