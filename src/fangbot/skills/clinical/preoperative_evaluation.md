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
