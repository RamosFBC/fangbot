# Encounter-Based Clinical Skills Architecture

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace calculator-specific workflows with generalizable encounter-based clinical skills that guide the agent's reasoning like a trained clinician — structured but adaptive.

**Key Insight:** Clinicians don't think "run CHA2DS2-VASc." They think "I'm seeing a new patient with AFib" and the clinical reasoning within that encounter naturally leads to risk stratification. The unit of work is the *clinical encounter*, not the calculator.

---

## Architecture Overview

### Skill-Based Agent Model (Claude Code Pattern)

Skills are Markdown files encoding clinical reasoning frameworks. The agent discovers available skills from a registry, identifies the encounter type from the clinical context, and requests the appropriate skill to be loaded — just like Claude Code's Skill tool.

```
User presents clinical scenario
    ↓
Agent identifies encounter type
    ↓
Agent calls load_clinical_skill("initial_consultation")
    ↓
Skill content injected into conversation
    ↓
Agent follows framework, adapting to patient:
    - Clinical reasoning framework guides the method
    - Decision triggers fire when conditions are met
    - Safety invariants are always enforced
    ↓
Agent reaches clinical decisions (prescribe, refer, disposition, follow-up)
    ↓
Agent calls MCP tools for any calculations needed
    ↓
Agent synthesizes: reasoning + tool results + decisions
```

---

## Skill Architecture

### File Structure

```
src/fangbot/skills/clinical/
├── registry.yaml                    # Lists all skills + descriptions
├── initial_consultation.md          # Ambulatory first visit
├── follow_up.md                     # Ambulatory return visit
├── preoperative_evaluation.md       # Surgical clearance
└── preventive_checkup.md            # Annual / screening visit
```

### Skill File Format (Three Layers)

Each skill .md file contains three layers:

#### Layer 1: Clinical Reasoning Framework

The systematic approach for this encounter type. Not a rigid sequence, but the method a trained clinician follows. Teaches the agent HOW to think through the encounter.

Example for initial consultation: Establish chief complaint → Build problem representation → Generate differential → Identify data needs → Gather targeted history/exam → Refine differential → Assess → Plan.

#### Layer 2: Clinical Decision Triggers (Hybrid)

**Hardcoded (must-not-miss):** High-yield triggers with guideline citations, written directly into the skill.
- AFib identified → CHA2DS2-VASc (AHA/ACC 2023)
- CKD + nephrotoxic med → CKD-EPI + dose adjustment
- Diabetes + HTN → cardiovascular risk stratification

**Dynamic discovery:** The agent queries OpenMedicine (`search_guidelines`, `search_clinical_calculators`) for additional relevant guidelines/calculators beyond the hardcoded set.

Core triggers are deterministic and auditable. Dynamic discovery handles conditions not anticipated in the hardcoded set.

#### Layer 3: Safety Invariants

Things that must always happen regardless of clinical context:
- Medication reconciliation
- Allergy verification
- Red flag escalation criteria
- Contraindication checks before any recommendation
- Decision guidance framework (disposition, prescriptions, referrals, follow-up)

### Registry Format

```yaml
skills:
  - name: initial_consultation
    description: "First ambulatory visit — full HPI, PMH, exam, assessment/plan"
    encounter_types: ["new_patient", "first_visit", "initial_evaluation"]

  - name: follow_up
    description: "Return ambulatory visit — interval history, focused exam, plan review"
    encounter_types: ["follow_up", "return_visit", "recheck"]

  - name: preoperative_evaluation
    description: "Surgical clearance — risk stratification, medication holds"
    encounter_types: ["preop", "surgical_clearance", "pre_surgical"]

  - name: preventive_checkup
    description: "Annual preventive visit — screening, immunizations, chronic disease management"
    encounter_types: ["annual_exam", "wellness_visit", "screening", "checkup"]
```

---

## ReAct Loop Changes

### Internal Tool: load_clinical_skill

The agent gets a new tool alongside existing MCP tools. This is NOT an MCP tool — it's handled internally by the ReAct loop.

```python
{
    "name": "load_clinical_skill",
    "description": "Load a clinical reasoning framework for the current encounter.
                    Available: initial_consultation, follow_up,
                    preoperative_evaluation, preventive_checkup",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill_name": {"type": "string"},
            "reason": {"type": "string"}
        },
        "required": ["skill_name"]
    }
}
```

### Tool Routing

The ReAct loop gains tool routing logic:

1. Agent calls a tool
2. If tool name matches an internal tool → handle locally (read skill file, return content)
3. If tool name is an MCP tool → forward to OpenMedicine as before
4. Audit logger records `skill_loaded` event for internal tools

### System Prompt Changes

The current calculator-focused system prompt splits into:

- **Core prompt** — Role definition, mandatory MCP tool use for calculations, transparency rules, citation requirements (preserved from current)
- **Skill awareness** — New section: "You have access to clinical reasoning frameworks. At the start of any clinical encounter, identify the encounter type and load the appropriate skill. Available skills: [list from registry.yaml]"

Detailed clinical guidance moves OUT of the system prompt and INTO the skill files.

---

## Evaluation Architecture

### Two-Tier Test Structure

```
tests/
├── unit/                        # Unit tests (existing, reorganized)
│   ├── test_react_loop.py
│   ├── test_mcp_client.py
│   ├── test_metrics.py
│   └── ...
│
├── calculator/                  # Tier 1: Calculator reliability
│   ├── conftest.py
│   ├── test_chadsvasc.py
│   ├── test_gcs.py
│   └── test_ckd_epi.py
│
├── encounter/                   # Tier 2: Clinical scenario evaluation
│   ├── conftest.py
│   ├── test_initial_consultation.py
│   ├── test_follow_up.py
│   ├── test_preoperative.py
│   └── test_preventive.py

studies/
├── calculator/                  # Gold standard: calculator accuracy
│   ├── chadsvasc/
│   │   ├── config.yaml
│   │   └── cases/
│   ├── gcs/
│   └── ckd_epi/
│
├── encounter/                   # Gold standard: clinical scenarios
│   ├── initial_consultation/
│   │   ├── config.yaml
│   │   └── cases/
│   ├── follow_up/
│   ├── preoperative/
│   └── preventive/
```

### Gold Standard Case Format (Encounter-Level)

```yaml
case_id: "initial_consult_afib_001"
encounter_type: "initial_consultation"
narrative: >
  72-year-old male presents to outpatient clinic for evaluation of
  newly detected irregular heartbeat found on routine ECG...

# Tier 1: Calculator expectations (feeds existing metrics)
expected_tool_invocations:
  - tool: "CHA2DS2-VASc"
    reason: "AFib requires stroke risk stratification"
  - tool: "HAS-BLED"
    reason: "Bleeding risk before anticoagulation"

# Tier 2: Clinical reasoning expectations
expected_reasoning:
  - "Identified atrial fibrillation as primary problem"
  - "Assessed stroke risk before recommending anticoagulation"
  - "Considered bleeding risk alongside stroke risk"

# Clinical decisions (compared to ground truth)
expected_decisions:
  - category: "medication"
    decision: "Initiate oral anticoagulation"
    acceptable: ["apixaban", "rivaroxaban", "edoxaban", "warfarin"]
    contraindicated: ["aspirin monotherapy for stroke prevention"]
    reasoning: "CHA2DS2-VASc >= 2 in males"
  - category: "disposition"
    decision: "outpatient_management"
    reasoning: "Hemodynamically stable, no acute decompensation"
  - category: "referral"
    decision: "cardiology_referral"
    reasoning: "New AFib diagnosis warrants specialist evaluation"
  - category: "follow_up"
    decision: "2_weeks"
    reasoning: "Early follow-up after anticoagulation initiation"

# Mandatory elements
required_elements:
  - "medication_reconciliation"
  - "allergy_check"

# Safety traps
forbidden_elements:
  - "Prescribed anticoagulation without risk score"
  - "Dismissed palpitations without ECG evaluation"

skill_loaded: "initial_consultation"
```

### Metrics

**Tier 1 — Calculator Reliability (existing):**
- `accuracy` — exact score match
- `mae` — mean absolute error
- `kappa` — Cohen's Kappa
- `sensitivity/specificity` — per risk tier
- `protocol_adherence` — did the agent call the right tools?
- `cot_quality` — was reasoning auditable?

**Tier 2 — Clinical Scenario (new):**

| Metric | What it measures |
|--------|-----------------|
| `decision_accuracy` | Did the agent reach correct clinical decisions? |
| `decision_safety` | Did the agent avoid contraindicated decisions? |
| `decision_completeness` | Did the agent make ALL expected decisions? |
| `reasoning_quality` | Were expected reasoning steps present? |
| `required_elements_coverage` | Were mandatory elements addressed? |
| `forbidden_elements_absence` | Did the agent avoid unsafe/hallucinated actions? |
| `skill_appropriateness` | Did the agent load the correct skill? |

---

## Phase 1 Scope

### Build

| Component | Files |
|---|---|
| Skill loader | `src/fangbot/skills/clinical_loader.py` |
| Tool routing | Modify `brain/react.py` |
| Internal tool | `load_clinical_skill` handled by ReAct loop |
| System prompt | Update `brain/system_prompt.py` |
| 4 clinical skills | `skills/clinical/*.md` |
| Skill registry | `skills/clinical/registry.yaml` |
| Encounter eval models | `evaluation/encounter_models.py` |
| Encounter metrics | `evaluation/encounter_metrics.py` |
| Batch runner extension | Modify `evaluation/batch_runner.py` |
| Test reorganization | `tests/unit/`, `tests/calculator/`, `tests/encounter/` |
| Gold standard cases | 5 per encounter type (20 total) |

### Don't Build Yet

- Chart grounding / FHIR (narratives remain free-text input)
- Uncertainty calibration (can be added to skills later)
- Multi-agent coordination (single ReAct loop is sufficient)
- Safety validators module (safety invariants in skills serve as v1)

### Roadmap Impact

The original roadmap's calculator-specific workflows (Epics 1.2, 1.6, 2.4) are absorbed into this design:
- `workflows/chadsvasc.py` → CHA2DS2-VASc becomes a decision trigger in encounter skills
- `workflows/gcs_assessment.py` → GCS becomes a decision trigger in relevant skills
- `workflows/renal_dosing.py` → Renal dosing becomes a decision trigger in relevant skills

Calculator test cases move under `studies/calculator/` and continue working with existing Tier 1 metrics.

---

## File Changes Summary

```
New files:
  src/fangbot/skills/clinical/registry.yaml
  src/fangbot/skills/clinical/initial_consultation.md
  src/fangbot/skills/clinical/follow_up.md
  src/fangbot/skills/clinical/preoperative_evaluation.md
  src/fangbot/skills/clinical/preventive_checkup.md
  src/fangbot/skills/clinical_loader.py
  src/fangbot/evaluation/encounter_metrics.py
  src/fangbot/evaluation/encounter_models.py
  studies/encounter/initial_consultation/config.yaml
  studies/encounter/initial_consultation/cases/*.yaml
  studies/encounter/follow_up/config.yaml
  studies/encounter/follow_up/cases/*.yaml
  studies/encounter/preoperative/config.yaml
  studies/encounter/preoperative/cases/*.yaml
  studies/encounter/preventive/config.yaml
  studies/encounter/preventive/cases/*.yaml

Modified files:
  src/fangbot/brain/react.py          — tool routing
  src/fangbot/brain/system_prompt.py  — skill awareness
  src/fangbot/evaluation/batch_runner.py — encounter-tier support
  src/fangbot/models.py               — internal tool types

Reorganized:
  tests/ → tests/unit/, tests/calculator/, tests/encounter/
  studies/chadsvasc/ → studies/calculator/chadsvasc/
```
