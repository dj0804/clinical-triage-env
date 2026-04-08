# Clinical Triage AI — OpenEnv Environment

A fully OpenEnv-compliant multi-step clinical triage simulation environment for evaluating AI agents on **diagnostic reasoning, risk assessment, safety awareness, and decision efficiency** under uncertainty.

## Overview

The agent receives partial patient data and must:
1. Gather information via questions and diagnostic tests
2. Propose a diagnosis with a confidence score
3. Assign a risk level (Critical / Monitor / Routine)
4. Route the patient to the appropriate department
5. Optionally escalate to a human clinician when uncertain
6. Finalize the triage

All transitions and rewards are **fully deterministic** — identical seeds and action sequences always produce identical results.

---

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then visit `http://localhost:7860/docs` to explore the API.

### Docker

```bash
docker build -t clinical-triage .
docker run -p 7860:7860 clinical-triage
```

### Run Inference

```bash
export API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Initialize episode |
| `POST` | `/step` | Execute action |
| `GET` | `/state` | Internal debug state |
| `GET` | `/schema` | Action/Observation schemas |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | WebSocket real-time API |

### POST /reset

```json
{
  "task_name": "easy_triage",
  "seed": 0,
  "episode_id": null
}
```

### POST /step

```json
{
  "action": {
    "action_type": "ask_question",
    "question": "Does the pain radiate to the arm?"
  }
}
```

---

## Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `ask_question` | `question: str` | Ask patient/records a question |
| `request_test` | `test_name: str` | Order a diagnostic test |
| `make_diagnosis` | `diagnosis: str`, `confidence: float [0.0,1.0]` | Propose diagnosis |
| `assign_risk` | `risk_level: Critical\|Monitor\|Routine` | Assign triage risk |
| `route_patient` | `department: str` | Route to department |
| `escalate_to_human` | `reason: str` | Escalate (terminates episode) |
| `finalize` | — | Finalize triage (terminates episode) |

---

## Observation Space

```json
{
  "symptoms": ["list", "of", "symptoms"],
  "patient_history": {"age": 62, "sex": "male", "comorbidities": [...]},
  "revealed_tests": {"troponin": "Troponin I: 4.2 ng/mL (elevated)"},
  "previous_actions": ["[step 1] request_test: ..."],
  "available_actions": ["ask_question", "request_test", "make_diagnosis", ...],
  "step_number": 3,
  "done": false,
  "reward": 0.05,
  "info": {"event": "useful_test"},
  "metadata": {"episode_id": "...", "has_diagnosis": false, "max_steps": 15}
}
```

---

## Tasks

### easy_triage
- **Data**: Complete, clean, unambiguous clinical presentation
- **Goal**: Direct diagnosis from given data
- **Scoring**: 40% diagnosis, 30% risk, 20% routing, 10% efficiency

### medium_triage
- **Data**: Missing key information requiring tests and questions
- **Goal**: Gather evidence, then diagnose
- **Scoring**: 30% diagnosis, 20% risk, 15% routing, 25% info quality, 10% efficiency

### hard_triage
- **Data**: Adversarial/misleading presentation, escalation may be required
- **Goal**: Avoid overconfident errors, recognize when to escalate
- **Scoring**: 25% diagnosis, 20% risk, 15% routing, 20% safety, 20% uncertainty handling

---

## Reward Design

### Per-step Rewards

| Event | Reward |
|-------|--------|
| Correct diagnosis (conf ≥ 0.7) | +0.30 |
| Correct diagnosis (conf < 0.7) | +0.20 |
| Wrong diagnosis | -0.25 |
| Overconfident wrong dx (≥0.8) at early step | -0.35 to -0.43 |
| Critical risk classified as Routine | -0.40 |
| Correct risk | +0.20 |
| Correct routing | +0.15 |
| Useful test | +0.05 |
| Appropriate escalation | +0.15 |
| Each step | -0.01 |
| Timeout (max steps exceeded) | -0.20 |

### Final Score
Per-component weighted average, each component clamped to [0.0, 1.0]. Always in [0.0, 1.0].

---

## Case Bank (15 Total)

### Easy (5)
1. STEMI — classic presentation
2. Appendicitis — RLQ pain + fever + leukocytosis
3. Hypertensive Crisis — BP 228/134 + end-organ damage
4. UTI — dysuria + positive UA
5. Asthma Exacerbation — wheezing + low O2 + known history

### Medium (5)
1. Pulmonary Embolism — dyspnea + risk factors (leg swelling hidden)
2. Diabetic Ketoacidosis — altered consciousness + Kussmaul breathing
3. Ectopic Pregnancy — abdominal pain + missed period (hCG hidden)
4. Bacterial Meningitis — fever + headache (neck stiffness subtle)
5. Acute Pancreatitis — epigastric pain + lipase hidden

### Hard / Adversarial (5)
1. Aortic Dissection — mimics GERD, antacids work, widened mediastinum
2. NSTEMI — normal first troponin, atypical in elderly diabetic woman
3. Panic Attack — convincingly mimics ACS, all cardiac workup normal
4. CO Poisoning — flu-like without fever, whole family affected
5. Subarachnoid Hemorrhage — thunderclap headache with migraine history

---

## Anti-Exploit Guarantees

- Repeated identical actions incur diminishing returns (×0.5 per repeat)
- Escalation without uncertainty/ambiguity markers → penalty
- Diagnosis without evidence gathering → safety penalty
- Overconfident wrong diagnosis at early steps → scaled penalty
- Final decision contradicting prior decisions → -0.10 score deduction
- Max steps = 15 enforced, timeout penalty applied at limit

---

## Baseline Scores

| Task | Random Agent | Rule-Following Agent | Strong Agent |
|------|-------------|---------------------|--------------|
| easy_triage | ~0.05–0.10 | ~0.60–0.75 | 0.85–1.00 |
| medium_triage | ~0.03–0.08 | ~0.45–0.60 | 0.70–0.90 |
| hard_triage | ~0.02–0.05 | ~0.30–0.50 | 0.60–0.85 |

---

## Environment Constraints

- Max steps: 15 per episode
- Runtime: < 6 min per task (20 min total)
- No ML weights — pure rule-based deterministic logic
- Docker image: < 200MB
- Memory: < 512MB RAM
- CPU: 1 vCPU sufficient

---

## Validation

```bash
# Health check
curl http://localhost:7860/health

# Reset easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_triage", "seed": 0}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "request_test", "test_name": "ECG"}}'

# Get state
curl http://localhost:7860/state
```
