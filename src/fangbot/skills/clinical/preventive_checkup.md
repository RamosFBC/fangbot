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
