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
