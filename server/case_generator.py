"""
Clinical Triage AI — Case Generator

Pre-built curated case bank: 5 easy, 5 medium, 5 hard cases.
Each case is a deterministic dict — no randomness in case data itself.
Case selection is seeded by reset(seed=N) → cases[seed % len(cases)].
"""

from typing import Any, Dict, List, Literal


# ---------------------------------------------------------------------------
# Case type alias
# ---------------------------------------------------------------------------

CaseDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# EASY CASES — complete data, direct diagnosis, minimal info-gathering needed
# ---------------------------------------------------------------------------

EASY_CASES: List[CaseDict] = [
    {
        "case_id": "easy_001",
        "difficulty": "easy",
        "initial_observation": {
            "symptoms": [
                "severe crushing chest pain radiating to left arm",
                "diaphoresis",
                "shortness of breath",
                "nausea",
            ],
            "patient_history": {
                "age": 62,
                "sex": "male",
                "comorbidities": ["hypertension", "hyperlipidemia", "smoking"],
                "medications": ["amlodipine", "atorvastatin"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "ST-elevation myocardial infarction",
            "diagnosis_aliases": ["STEMI", "heart attack", "myocardial infarction", "MI"],
            "risk_level": "Critical",
            "department": "Cardiology",
            "requires_escalation": False,
        },
        "available_tests": {
            "troponin": {
                "result": "Troponin I: 4.2 ng/mL (markedly elevated, normal <0.04)",
                "reveals_key_info": True,
            },
            "ECG": {
                "result": "ST elevation in leads II, III, aVF and V4-V6. Reciprocal changes in I, aVL.",
                "reveals_key_info": True,
            },
            "chest_xray": {
                "result": "Mild cardiomegaly. No pneumothorax.",
                "reveals_key_info": False,
            },
            "CBC": {
                "result": "WBC 11.2, Hgb 13.8, Plt 210 — within normal limits",
                "reveals_key_info": False,
            },
            "BMP": {
                "result": "Na 139, K 4.1, Creatinine 1.0, Glucose 118",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "how long have you had chest pain": "Started about 45 minutes ago, came on suddenly.",
            "do you have any cardiac history": "No prior heart attacks, but I have high blood pressure.",
            "does the pain radiate anywhere": "Yes, it goes to my left arm and jaw.",
            "are you short of breath": "Yes, I am very short of breath.",
            "do you feel dizzy or lightheaded": "A little dizzy, yes.",
            "any recent illness": "No recent illness.",
            "what is your pain score": "9 out of 10.",
        },
        "question_relevance": {
            "how long have you had chest pain": "high",
            "do you have any cardiac history": "high",
            "does the pain radiate anywhere": "high",
            "are you short of breath": "medium",
            "do you feel dizzy or lightheaded": "medium",
            "any recent illness": "low",
            "what is your pain score": "medium",
        },
    },
    {
        "case_id": "easy_002",
        "difficulty": "easy",
        "initial_observation": {
            "symptoms": [
                "right lower quadrant abdominal pain",
                "fever (38.7°C)",
                "nausea",
                "loss of appetite",
                "rebound tenderness on palpation",
            ],
            "patient_history": {
                "age": 24,
                "sex": "female",
                "comorbidities": [],
                "medications": [],
                "allergies": ["penicillin"],
            },
        },
        "hidden_truth": {
            "diagnosis": "acute appendicitis",
            "diagnosis_aliases": ["appendicitis"],
            "risk_level": "Monitor",
            "department": "Surgery",
            "requires_escalation": False,
        },
        "available_tests": {
            "CBC": {
                "result": "WBC 16.8 (elevated, normal 4.5–11.0). Neutrophilia 88%.",
                "reveals_key_info": True,
            },
            "CT_abdomen": {
                "result": "Appendix measures 9mm in diameter with periappendiceal fat stranding. Consistent with acute appendicitis.",
                "reveals_key_info": True,
            },
            "urinalysis": {
                "result": "Normal. No signs of UTI.",
                "reveals_key_info": False,
            },
            "beta_hCG": {
                "result": "Negative. Not pregnant.",
                "reveals_key_info": True,
            },
            "lipase": {
                "result": "Normal (32 U/L).",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "where exactly is the pain": "Lower right side, gets worse when I move.",
            "when did the pain start": "About 18 hours ago, started around the belly button then moved down.",
            "do you have any fever": "Yes, I felt feverish earlier.",
            "have you vomited": "I feel like I might, but haven't yet.",
            "last menstrual period": "About 3 weeks ago, on time.",
            "any previous abdominal surgeries": "No.",
            "do you have diarrhea or constipation": "I haven't had a bowel movement today.",
        },
        "question_relevance": {
            "where exactly is the pain": "high",
            "when did the pain start": "high",
            "do you have any fever": "high",
            "have you vomited": "medium",
            "last menstrual period": "high",
            "any previous abdominal surgeries": "medium",
            "do you have diarrhea or constipation": "medium",
        },
    },
    {
        "case_id": "easy_003",
        "difficulty": "easy",
        "initial_observation": {
            "symptoms": [
                "severe headache",
                "blurred vision",
                "blood pressure 228/134 mmHg",
                "confusion",
            ],
            "patient_history": {
                "age": 55,
                "sex": "male",
                "comorbidities": ["hypertension", "diabetes type 2"],
                "medications": ["metformin"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "hypertensive crisis",
            "diagnosis_aliases": ["hypertensive emergency", "malignant hypertension"],
            "risk_level": "Critical",
            "department": "ICU",
            "requires_escalation": False,
        },
        "available_tests": {
            "ECG": {
                "result": "Left ventricular hypertrophy pattern. No acute ST changes.",
                "reveals_key_info": True,
            },
            "BMP": {
                "result": "Creatinine 2.8 (elevated, baseline 1.1). K 4.2. Na 138.",
                "reveals_key_info": True,
            },
            "urinalysis": {
                "result": "Proteinuria 3+. Microscopic hematuria.",
                "reveals_key_info": True,
            },
            "CT_head": {
                "result": "No intracranial hemorrhage. Mild cerebral edema.",
                "reveals_key_info": True,
            },
            "chest_xray": {
                "result": "Mild pulmonary vascular congestion.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "are you taking your blood pressure medication": "I ran out two weeks ago and didn't refill.",
            "how long have you had this headache": "Since this morning, about 6 hours.",
            "have you had any vision changes before": "Not like this.",
            "do you have chest pain": "A little tightness.",
            "any recent stress or illness": "Very stressful week at work.",
        },
        "question_relevance": {
            "are you taking your blood pressure medication": "high",
            "how long have you had this headache": "high",
            "have you had any vision changes before": "medium",
            "do you have chest pain": "medium",
            "any recent stress or illness": "low",
        },
    },
    {
        "case_id": "easy_004",
        "difficulty": "easy",
        "initial_observation": {
            "symptoms": [
                "burning sensation during urination",
                "frequent urge to urinate",
                "cloudy urine",
                "mild pelvic discomfort",
            ],
            "patient_history": {
                "age": 28,
                "sex": "female",
                "comorbidities": [],
                "medications": ["oral contraceptive"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "urinary tract infection",
            "diagnosis_aliases": ["UTI", "cystitis", "bladder infection"],
            "risk_level": "Routine",
            "department": "Urology",
            "requires_escalation": False,
        },
        "available_tests": {
            "urinalysis": {
                "result": "Nitrites positive. Leukocyte esterase 3+. WBC >20/hpf. Bacteria present.",
                "reveals_key_info": True,
            },
            "urine_culture": {
                "result": "E. coli growth, >100,000 CFU/mL. Sensitive to trimethoprim-sulfamethoxazole.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "WBC 9.2 (normal). No systemic infection signs.",
                "reveals_key_info": False,
            },
            "beta_hCG": {
                "result": "Negative.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "do you have fever or chills": "No fever.",
            "do you have back or flank pain": "No back pain.",
            "how long have these symptoms been present": "Since yesterday.",
            "any recent sexual activity": "Yes, recently.",
            "any previous UTIs": "Had one about a year ago.",
        },
        "question_relevance": {
            "do you have fever or chills": "high",
            "do you have back or flank pain": "high",
            "how long have these symptoms been present": "medium",
            "any recent sexual activity": "medium",
            "any previous UTIs": "medium",
        },
    },
    {
        "case_id": "easy_005",
        "difficulty": "easy",
        "initial_observation": {
            "symptoms": [
                "wheezing",
                "shortness of breath",
                "chest tightness",
                "oxygen saturation 91%",
            ],
            "patient_history": {
                "age": 17,
                "sex": "male",
                "comorbidities": ["asthma", "allergic rhinitis"],
                "medications": ["albuterol inhaler (PRN)", "fluticasone inhaler"],
                "allergies": ["cat dander"],
            },
        },
        "hidden_truth": {
            "diagnosis": "acute asthma exacerbation",
            "diagnosis_aliases": ["asthma attack", "asthma exacerbation", "bronchospasm"],
            "risk_level": "Monitor",
            "department": "Respiratory",
            "requires_escalation": False,
        },
        "available_tests": {
            "spirometry": {
                "result": "FEV1 58% predicted. Significant obstruction.",
                "reveals_key_info": True,
            },
            "chest_xray": {
                "result": "Hyperinflation. No pneumonia or pneumothorax.",
                "reveals_key_info": False,
            },
            "ABG": {
                "result": "pH 7.44, pO2 68, pCO2 38. Mild hypoxemia.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "WBC 8.4 (normal). Eosinophils slightly elevated.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "what triggered this": "I was near a cat earlier.",
            "have you used your inhaler": "Yes, used it twice but it barely helped.",
            "how long has this been going on": "About 2 hours.",
            "do you have fever": "No fever.",
            "any recent illness or infections": "Had a cold last week.",
        },
        "question_relevance": {
            "what triggered this": "high",
            "have you used your inhaler": "high",
            "how long has this been going on": "high",
            "do you have fever": "medium",
            "any recent illness or infections": "medium",
        },
    },
]


# ---------------------------------------------------------------------------
# MEDIUM CASES — incomplete data, requires info gathering / test ordering
# ---------------------------------------------------------------------------

MEDIUM_CASES: List[CaseDict] = [
    {
        "case_id": "medium_001",
        "difficulty": "medium",
        "initial_observation": {
            "symptoms": [
                "sudden onset shortness of breath",
                "tachycardia (HR 118 bpm)",
                "pleuritic chest pain (worsens with breathing)",
            ],
            "patient_history": {
                "age": 45,
                "sex": "female",
                "comorbidities": ["obesity"],
                "medications": ["oral contraceptive"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "pulmonary embolism",
            "diagnosis_aliases": ["PE", "pulmonary thromboembolism"],
            "risk_level": "Critical",
            "department": "Cardiology",
            "requires_escalation": False,
        },
        "available_tests": {
            "D_dimer": {
                "result": "D-dimer: 3.8 mg/L (markedly elevated, normal <0.5). High probability PE.",
                "reveals_key_info": True,
            },
            "CT_pulmonary_angiography": {
                "result": "Bilateral pulmonary emboli. Saddle embolus at bifurcation. Right heart strain.",
                "reveals_key_info": True,
            },
            "ECG": {
                "result": "Sinus tachycardia. S1Q3T3 pattern. Right bundle branch block.",
                "reveals_key_info": True,
            },
            "troponin": {
                "result": "Troponin I: 0.18 ng/mL (mildly elevated — right heart strain).",
                "reveals_key_info": True,
            },
            "chest_xray": {
                "result": "Subtle Hampton's hump (wedge-shaped opacity). Otherwise unremarkable.",
                "reveals_key_info": False,
            },
            "ABG": {
                "result": "pH 7.48, pO2 61, pCO2 32. Hypoxemia with respiratory alkalosis.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "Normal CBC.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "do you have leg swelling or pain": "Now that you mention it, my left leg has been a bit swollen for a few days.",
            "have you traveled recently": "Long flight from Asia 10 days ago, 14 hours.",
            "are you on birth control": "Yes, oral contraceptive for 2 years.",
            "do you have family history of clotting": "My mother had a DVT once.",
            "any recent surgery or prolonged immobility": "No surgery, but I was mostly sitting on that long flight.",
            "how fast did the shortness of breath come on": "Very suddenly, within minutes.",
        },
        "question_relevance": {
            "do you have leg swelling or pain": "high",
            "have you traveled recently": "high",
            "are you on birth control": "high",
            "do you have family history of clotting": "medium",
            "any recent surgery or prolonged immobility": "high",
            "how fast did the shortness of breath come on": "medium",
        },
    },
    {
        "case_id": "medium_002",
        "difficulty": "medium",
        "initial_observation": {
            "symptoms": [
                "altered level of consciousness",
                "fruity breath odor",
                "rapid deep breathing (Kussmaul respirations)",
                "abdominal pain",
            ],
            "patient_history": {
                "age": 19,
                "sex": "female",
                "comorbidities": ["type 1 diabetes (newly diagnosed, non-compliant)"],
                "medications": ["insulin (reported non-compliant)"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "diabetic ketoacidosis",
            "diagnosis_aliases": ["DKA"],
            "risk_level": "Critical",
            "department": "ICU",
            "requires_escalation": False,
        },
        "available_tests": {
            "blood_glucose": {
                "result": "Blood glucose: 612 mg/dL (markedly elevated).",
                "reveals_key_info": True,
            },
            "ABG": {
                "result": "pH 7.14, pCO2 18, HCO3 8. Severe metabolic acidosis with respiratory compensation.",
                "reveals_key_info": True,
            },
            "BMP": {
                "result": "Na 131, K 5.8, HCO3 8, anion gap 28 (elevated). Creatinine 1.6.",
                "reveals_key_info": True,
            },
            "urinalysis": {
                "result": "Glucose 4+. Ketones 4+. No infection markers.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "WBC 14.2 (mildly elevated, stress response). Otherwise normal.",
                "reveals_key_info": False,
            },
            "HbA1c": {
                "result": "HbA1c: 14.2% (severely uncontrolled diabetes).",
                "reveals_key_info": True,
            },
        },
        "valid_questions": {
            "when did you last take insulin": "I haven't taken it in 3 days.",
            "have you been eating or drinking normally": "No appetite, very thirsty.",
            "any recent illness or infection": "Had a stomach bug last week.",
            "how long have you been feeling this way": "Getting worse over 2 days.",
            "do you feel nauseous or have you vomited": "Yes, I vomited twice.",
            "any headache or vision changes": "Mild headache.",
        },
        "question_relevance": {
            "when did you last take insulin": "high",
            "have you been eating or drinking normally": "high",
            "any recent illness or infection": "high",
            "how long have you been feeling this way": "medium",
            "do you feel nauseous or have you vomited": "medium",
            "any headache or vision changes": "medium",
        },
    },
    {
        "case_id": "medium_003",
        "difficulty": "medium",
        "initial_observation": {
            "symptoms": [
                "severe lower abdominal pain",
                "vaginal spotting",
            ],
            "patient_history": {
                "age": 26,
                "sex": "female",
                "comorbidities": [],
                "medications": [],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "ectopic pregnancy",
            "diagnosis_aliases": ["tubal pregnancy", "ectopic"],
            "risk_level": "Critical",
            "department": "Surgery",
            "requires_escalation": False,
        },
        "available_tests": {
            "beta_hCG": {
                "result": "beta-hCG: 3200 mIU/mL (elevated). Discriminatory zone: no IUP on US at this level.",
                "reveals_key_info": True,
            },
            "pelvic_ultrasound": {
                "result": "No intrauterine pregnancy. 2.5 cm adnexal mass on right side. Free fluid in pelvis (hemoperitoneum).",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "Hgb 9.2 (low, normal 12-16). Active hemorrhage suspected.",
                "reveals_key_info": True,
            },
            "BMP": {
                "result": "Normal",
                "reveals_key_info": False,
            },
            "urinalysis": {
                "result": "Normal.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "what is your last menstrual period": "About 7 weeks ago.",
            "could you be pregnant": "I thought maybe, but I haven't tested.",
            "do you have shoulder pain": "Yes, actually, right shoulder — I thought it was nothing.",
            "have you had ectopic before": "No, first time.",
            "are you using contraception": "No.",
            "do you feel dizzy or lightheaded": "Yes, I almost fainted coming in.",
        },
        "question_relevance": {
            "what is your last menstrual period": "high",
            "could you be pregnant": "high",
            "do you have shoulder pain": "high",
            "have you had ectopic before": "medium",
            "are you using contraception": "medium",
            "do you feel dizzy or lightheaded": "high",
        },
    },
    {
        "case_id": "medium_004",
        "difficulty": "medium",
        "initial_observation": {
            "symptoms": [
                "fever (39.1°C)",
                "severe headache",
                "photophobia",
                "vomiting",
            ],
            "patient_history": {
                "age": 20,
                "sex": "male",
                "comorbidities": [],
                "medications": [],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "bacterial meningitis",
            "diagnosis_aliases": ["meningitis", "meningococcal meningitis"],
            "risk_level": "Critical",
            "department": "ICU",
            "requires_escalation": False,
        },
        "available_tests": {
            "lumbar_puncture": {
                "result": "CSF: WBC 2800 (neutrophilic pleocytosis), protein 180 mg/dL (elevated), glucose 30 mg/dL (low). Gram stain: gram-positive diplococci.",
                "reveals_key_info": True,
            },
            "blood_cultures": {
                "result": "Positive. Streptococcus pneumoniae.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "WBC 22.4 (markedly elevated). Left shift.",
                "reveals_key_info": True,
            },
            "CT_head": {
                "result": "No mass effect. No contraindication to LP.",
                "reveals_key_info": False,
            },
            "BMP": {
                "result": "Na 128 (hyponatremia — SIADH). Otherwise normal.",
                "reveals_key_info": True,
            },
        },
        "valid_questions": {
            "do you have neck stiffness": "Yes, I can barely move my neck.",
            "have you been around anyone sick recently": "Lots of people at a university dorm.",
            "how fast did this come on": "Very fast, within hours.",
            "any rash on your skin": "There's a small purplish rash on my torso.",
            "have you been vaccinated against meningococcus": "I'm not sure.",
            "any recent ear or sinus infection": "Had a cold 2 weeks ago.",
        },
        "question_relevance": {
            "do you have neck stiffness": "high",
            "have you been around anyone sick recently": "medium",
            "how fast did this come on": "high",
            "any rash on your skin": "high",
            "have you been vaccinated against meningococcus": "medium",
            "any recent ear or sinus infection": "medium",
        },
    },
    {
        "case_id": "medium_005",
        "difficulty": "medium",
        "initial_observation": {
            "symptoms": [
                "severe epigastric pain radiating to the back",
                "nausea and vomiting",
                "low-grade fever (37.9°C)",
            ],
            "patient_history": {
                "age": 48,
                "sex": "male",
                "comorbidities": ["alcohol use disorder", "hypertriglyceridemia"],
                "medications": ["none currently"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "acute pancreatitis",
            "diagnosis_aliases": ["pancreatitis"],
            "risk_level": "Monitor",
            "department": "GI",
            "requires_escalation": False,
        },
        "available_tests": {
            "lipase": {
                "result": "Lipase: 1840 U/L (markedly elevated, normal <60). Diagnostic for pancreatitis.",
                "reveals_key_info": True,
            },
            "amylase": {
                "result": "Amylase: 620 U/L (elevated).",
                "reveals_key_info": True,
            },
            "CT_abdomen": {
                "result": "Pancreatic edema and peripancreatic fat stranding. No necrosis.",
                "reveals_key_info": True,
            },
            "BMP": {
                "result": "Ca 7.8 (low), Creatinine 1.3, Glucose 210.",
                "reveals_key_info": True,
            },
            "LFTs": {
                "result": "ALT 88, AST 102 (mildly elevated). Bili 1.4. No biliary obstruction.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "WBC 13.8 (elevated). Hct 47 (hemoconcentration).",
                "reveals_key_info": True,
            },
        },
        "valid_questions": {
            "how much alcohol do you drink": "About a 6-pack nightly for years.",
            "when did the pain start": "After a large meal and heavy drinking last night.",
            "does the pain radiate": "Yes, straight through to my back.",
            "any gallstones or prior pancreatitis": "Had one episode 2 years ago.",
            "any recent trauma": "No trauma.",
            "are you on any lipid-lowering medications": "Was on gemfibrozil but stopped.",
        },
        "question_relevance": {
            "how much alcohol do you drink": "high",
            "when did the pain start": "high",
            "does the pain radiate": "high",
            "any gallstones or prior pancreatitis": "high",
            "any recent trauma": "low",
            "are you on any lipid-lowering medications": "medium",
        },
    },
]


# ---------------------------------------------------------------------------
# HARD CASES — adversarial / misleading / escalation-required
# ---------------------------------------------------------------------------

HARD_CASES: List[CaseDict] = [
    {
        "case_id": "hard_001",
        "difficulty": "hard",
        "initial_observation": {
            "symptoms": [
                "chest pain (described as burning/gnawing)",
                "pain partially relieved by antacids",
                "hypertension (BP 178/102)",
            ],
            "patient_history": {
                "age": 67,
                "sex": "male",
                "comorbidities": ["hypertension", "GERD history"],
                "medications": ["omeprazole", "losartan"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "aortic dissection",
            "diagnosis_aliases": ["type A aortic dissection", "thoracic aortic dissection"],
            "risk_level": "Critical",
            "department": "Cardiology",
            "requires_escalation": True,
        },
        "adversarial_notes": "Antacid response and GERD history are misleading. Must not diagnose GERD.",
        "mimics": ["GERD", "peptic ulcer disease", "STEMI"],
        "available_tests": {
            "troponin": {
                "result": "Troponin I: 0.06 ng/mL (borderline — can occur in dissection).",
                "reveals_key_info": False,
            },
            "chest_xray": {
                "result": "Widened mediastinum (>8cm). Subtle aortic knob enlargement.",
                "reveals_key_info": True,
            },
            "CT_aorta": {
                "result": "Type A aortic dissection. Intimal flap from aortic root extending to descending aorta.",
                "reveals_key_info": True,
            },
            "ECG": {
                "result": "Normal sinus rhythm. No ST changes.",
                "reveals_key_info": False,
            },
            "BMP": {
                "result": "Normal.",
                "reveals_key_info": False,
            },
            "D_dimer": {
                "result": "D-dimer: 2.1 mg/L (elevated — can indicate dissection).",
                "reveals_key_info": True,
            },
        },
        "valid_questions": {
            "describe the pain quality": "It's tearing, ripping — different from my usual heartburn.",
            "does the pain radiate": "It moves to my back and between my shoulder blades.",
            "are both arms equal blood pressure": "Right arm 178/102, left arm 140/85.",
            "did the pain come on suddenly": "Yes, like a sudden explosion in my chest.",
            "any history of Marfan syndrome or aortic disease": "No known history.",
            "do you feel any neurological symptoms": "I had brief numbness in my right hand.",
        },
        "question_relevance": {
            "describe the pain quality": "high",
            "does the pain radiate": "high",
            "are both arms equal blood pressure": "high",
            "did the pain come on suddenly": "high",
            "any history of Marfan syndrome or aortic disease": "medium",
            "do you feel any neurological symptoms": "high",
        },
    },
    {
        "case_id": "hard_002",
        "difficulty": "hard",
        "initial_observation": {
            "symptoms": [
                "atypical chest discomfort (pressure, not pain)",
                "exertional fatigue",
                "mild shortness of breath",
            ],
            "patient_history": {
                "age": 71,
                "sex": "female",
                "comorbidities": ["type 2 diabetes", "hypertension"],
                "medications": ["metformin", "enalapril"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "non-ST-elevation myocardial infarction",
            "diagnosis_aliases": ["NSTEMI", "demand ischemia", "myocardial infarction without ST elevation"],
            "risk_level": "Critical",
            "department": "Cardiology",
            "requires_escalation": True,
        },
        "adversarial_notes": "Initial troponin may be normal. Serial troponins required. Atypical presentation common in elderly diabetic women.",
        "mimics": ["musculoskeletal pain", "GERD", "fatigue", "deconditioning"],
        "available_tests": {
            "troponin": {
                "result": "Troponin I (initial): 0.03 ng/mL (within normal limits). Serial troponin needed.",
                "reveals_key_info": False,
            },
            "serial_troponin": {
                "result": "Troponin I (3-hour): 0.22 ng/mL (rising — confirms myocardial injury). Troponin I (6-hour): 0.89 ng/mL.",
                "reveals_key_info": True,
            },
            "ECG": {
                "result": "T-wave inversions in V3-V6. No ST elevation.",
                "reveals_key_info": True,
            },
            "echocardiogram": {
                "result": "Wall motion abnormality in LAD territory. EF 45%.",
                "reveals_key_info": True,
            },
            "BMP": {
                "result": "Glucose 220. Creatinine 1.2.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "how long have you had this fatigue": "About 3 days, getting progressively worse.",
            "does exertion make symptoms worse": "Yes, walking to the bathroom makes it worse.",
            "any prior cardiac history": "Had a stress test 5 years ago that was normal.",
            "do you experience sweating": "Yes, night sweats recently.",
            "any jaw, arm or epigastric discomfort": "Some jaw aching, I thought it was dental.",
            "do you feel worse than usual": "Much worse than my baseline.",
        },
        "question_relevance": {
            "how long have you had this fatigue": "high",
            "does exertion make symptoms worse": "high",
            "any prior cardiac history": "medium",
            "do you experience sweating": "medium",
            "any jaw, arm or epigastric discomfort": "high",
            "do you feel worse than usual": "high",
        },
    },
    {
        "case_id": "hard_003",
        "difficulty": "hard",
        "initial_observation": {
            "symptoms": [
                "palpitations",
                "chest tightness",
                "breathing difficulty",
                "numbness in hands and lips",
                "feeling of impending doom",
            ],
            "patient_history": {
                "age": 23,
                "sex": "female",
                "comorbidities": ["generalized anxiety disorder"],
                "medications": ["sertraline"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "panic attack",
            "diagnosis_aliases": ["panic disorder episode", "acute anxiety attack"],
            "risk_level": "Routine",
            "department": "Psychiatry",
            "requires_escalation": False,
        },
        "adversarial_notes": "Symptoms convincingly mimic ACS. DO NOT diagnose MI or route to Cardiology. Overconfident Critical assignment is penalized heavily.",
        "mimics": ["ACS", "pulmonary embolism", "arrhythmia"],
        "available_tests": {
            "ECG": {
                "result": "Normal sinus rhythm, rate 110 bpm. No ST changes. No arrhythmia.",
                "reveals_key_info": True,
            },
            "troponin": {
                "result": "Troponin I: <0.01 ng/mL (undetectable — normal).",
                "reveals_key_info": True,
            },
            "D_dimer": {
                "result": "D-dimer: 0.3 mg/L (normal).",
                "reveals_key_info": True,
            },
            "CBG": {
                "result": "Blood glucose: 94 mg/dL (normal).",
                "reveals_key_info": False,
            },
            "ABG": {
                "result": "pH 7.52, pCO2 28, pO2 98. Respiratory alkalosis (hyperventilation).",
                "reveals_key_info": True,
            },
        },
        "valid_questions": {
            "have you had episodes like this before": "Yes, maybe 4-5 times in the past year.",
            "were you under stress before this started": "I had a big exam today.",
            "any family history of heart disease": "None.",
            "did the symptoms peak within minutes": "Yes, about 10 minutes then started improving.",
            "are you seeing a therapist or psychiatrist": "I see a therapist weekly for anxiety.",
            "any history of cardiac problems": "None, all my checkups are fine.",
        },
        "question_relevance": {
            "have you had episodes like this before": "high",
            "were you under stress before this started": "high",
            "any family history of heart disease": "medium",
            "did the symptoms peak within minutes": "high",
            "are you seeing a therapist or psychiatrist": "high",
            "any history of cardiac problems": "medium",
        },
    },
    {
        "case_id": "hard_004",
        "difficulty": "hard",
        "initial_observation": {
            "symptoms": [
                "headache",
                "nausea",
                "fatigue",
                "multiple family members feel unwell simultaneously",
            ],
            "patient_history": {
                "age": 35,
                "sex": "male",
                "comorbidities": [],
                "medications": [],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "carbon monoxide poisoning",
            "diagnosis_aliases": ["CO poisoning", "carbon monoxide toxicity"],
            "risk_level": "Critical",
            "department": "ED",
            "requires_escalation": True,
        },
        "adversarial_notes": "No fever. Flu-like without fever in winter is CO until proven otherwise. Initial workup must include CO level. Escalation is appropriate due to public safety concerns.",
        "mimics": ["viral illness", "food poisoning", "migraine"],
        "available_tests": {
            "COHb": {
                "result": "Carboxyhemoglobin: 28% (severely elevated, normal <5%). CO poisoning confirmed.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "Normal CBC.",
                "reveals_key_info": False,
            },
            "BMP": {
                "result": "Lactate 4.2 (elevated — tissue hypoxia).",
                "reveals_key_info": True,
            },
            "ECG": {
                "result": "Sinus tachycardia. No ischemic changes.",
                "reveals_key_info": False,
            },
            "blood_cultures": {
                "result": "Pending (no growth in 24h).",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "do others in your home feel the same": "Yes, my wife and two children all have headaches.",
            "do symptoms improve when you leave home": "Yes, actually, much better since arriving here.",
            "what season is it and do you use gas heating": "It's winter, we use a gas furnace.",
            "any recent furnace or fireplace use": "We just started using the furnace this week after months off.",
            "any cherry-red flushing to skin": "My wife mentioned my face looked very red this morning.",
            "do you have a carbon monoxide detector at home": "I don't think so.",
        },
        "question_relevance": {
            "do others in your home feel the same": "high",
            "do symptoms improve when you leave home": "high",
            "what season is it and do you use gas heating": "high",
            "any recent furnace or fireplace use": "high",
            "any cherry-red flushing to skin": "medium",
            "do you have a carbon monoxide detector at home": "medium",
        },
    },
    {
        "case_id": "hard_005",
        "difficulty": "hard",
        "initial_observation": {
            "symptoms": [
                "thunderclap headache (worst headache of life)",
                "brief loss of consciousness",
                "neck stiffness",
            ],
            "patient_history": {
                "age": 49,
                "sex": "female",
                "comorbidities": ["migraines (history)"],
                "medications": ["sumatriptan (PRN)"],
                "allergies": [],
            },
        },
        "hidden_truth": {
            "diagnosis": "subarachnoid hemorrhage",
            "diagnosis_aliases": ["SAH", "subarachnoid bleed", "ruptured aneurysm"],
            "risk_level": "Critical",
            "department": "Neurology",
            "requires_escalation": True,
        },
        "adversarial_notes": "Migraine history creates anchoring bias toward misdiagnosis. Thunderclap onset and LOC distinguish SAH. CT may be negative in first hours — LP then required.",
        "mimics": ["migraine", "cluster headache", "tension headache"],
        "available_tests": {
            "CT_head": {
                "result": "CT head: Hyperdensity in basal cisterns and Sylvian fissures. Consistent with subarachnoid hemorrhage. Fisher Grade III.",
                "reveals_key_info": True,
            },
            "lumbar_puncture": {
                "result": "Xanthochromia present. RBC >100,000/mm3 in all tubes (no clearing — true bleed not traumatic tap).",
                "reveals_key_info": True,
            },
            "MRI_brain": {
                "result": "FLAIR hyperintensity in CSF spaces confirming SAH.",
                "reveals_key_info": True,
            },
            "CBC": {
                "result": "Normal CBC.",
                "reveals_key_info": False,
            },
            "BMP": {
                "result": "Normal.",
                "reveals_key_info": False,
            },
        },
        "valid_questions": {
            "was this headache different from your usual migraines": "Yes. This hit like a thunderbolt — nothing like my migraines.",
            "did you lose consciousness": "Yes, for a few seconds.",
            "when did the headache reach max intensity": "Instantaneously — within 1-2 seconds.",
            "do you have neck stiffness": "Yes, very stiff and painful.",
            "any visual changes or neurological symptoms": "Double vision briefly.",
            "any prior similar headaches": "Never had one this bad.",
        },
        "question_relevance": {
            "was this headache different from your usual migraines": "high",
            "did you lose consciousness": "high",
            "when did the headache reach max intensity": "high",
            "do you have neck stiffness": "high",
            "any visual changes or neurological symptoms": "high",
            "any prior similar headaches": "high",
        },
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALL_CASES: List[CaseDict] = EASY_CASES + MEDIUM_CASES + HARD_CASES

CASES_BY_DIFFICULTY: Dict[str, List[CaseDict]] = {
    "easy": EASY_CASES,
    "medium": MEDIUM_CASES,
    "hard": HARD_CASES,
}

TASK_TO_DIFFICULTY: Dict[str, str] = {
    "easy_triage": "easy",
    "medium_triage": "medium",
    "hard_triage": "hard",
}


def get_case(task_name: str, seed: int = 0) -> CaseDict:
    """
    Deterministically select a case for the given task.

    Args:
        task_name: One of "easy_triage", "medium_triage", "hard_triage"
        seed: Integer seed for reproducible selection

    Returns:
        CaseDict — the selected patient case

    Raises:
        ValueError: If task_name is not recognized
    """
    difficulty = TASK_TO_DIFFICULTY.get(task_name)
    if difficulty is None:
        raise ValueError(
            f"Unknown task_name '{task_name}'. "
            f"Valid options: {list(TASK_TO_DIFFICULTY.keys())}"
        )
    cases = CASES_BY_DIFFICULTY[difficulty]
    return cases[seed % len(cases)]


def list_all_cases() -> List[str]:
    """Return all case IDs."""
    return [c["case_id"] for c in ALL_CASES]
