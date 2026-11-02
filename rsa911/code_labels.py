from __future__ import annotations

from typing import Any


MISSING_LABEL = "(blank / missing)"

YES_NO_UNKNOWN_COLUMNS = {
    "amerindian",
    "asian",
    "basicskillsdeficient",
    "black",
    "culturalbarriers",
    "dishomemaker",
    "disability",
    "englishlearner",
    "exoffenderstatus",
    "hawaiian",
    "hispanic",
    "homelessorrunaway",
    "longtermunemp",
    "lowincomestatus",
    "migrantfarmworker",
    "singleparent",
    "student",
    "veteran",
    "white",
}

YES_NO_UNKNOWN_LABELS = {
    "0": "No",
    "1": "Yes",
    "9": "Unknown / not reported",
    "": MISSING_LABEL,
}

DISABILITY_SIGNIFICANCE_LABELS = {
    "0": "No significant disability indicated",
    "1": "Significant disability",
    "2": "Most significant disability",
    "": MISSING_LABEL,
}

IPE_EMP_STATUS_LABELS = {
    "1": "Employed: Competitive Integrated Employment",
    "2": "Employed: Self-Employment",
    "3": "Employed: Randolph-Sheppard BEP",
    "4": "Employed: State Agency Managed BEP",
    "5": "Extended Employment",
    "6": "Received notice of termination of employment or military separation is pending",
    "7": "Not Employed: Student in Secondary Education",
    "8": "Not Employed: All other Students",
    "9": "Not Employed: Trainee, Intern, or Volunteer",
    "10": "Not employed: Other Reason",
    "": MISSING_LABEL,
}

VOC_REHAB_LABELS = {
    "1": "Received services from the vocational rehabilitation program",
    "2": "Received services from Department of Veterans Affairs VR&E program",
    "3": "Received services from both VR and Department of Veterans Affairs VR&E programs",
    "0": "Did not receive any services",
    "9": "Participant did not self-identify",
    "": MISSING_LABEL,
}

EXIT_EMP_OUTCOME_LABELS = {
    "1": "Competitive integrated employment",
    "2": "Self-employment",
    "3": "Randolph-Sheppard BEP",
    "4": "State Agency Managed BEP",
    "5": "Supported employment in competitive integrated employment",
    "7": "Homemaker",
    "": MISSING_LABEL,
}

DISABILITY_TYPE_LABELS = {
    "00": "No Disability",
    "01": "Blindness",
    "02": "Other Visual Disabilities",
    "03": "Deafness, Primary Communication Visual",
    "04": "Deafness, Primary Communication Auditory",
    "05": "Hearing Loss, Primary Communication Visual",
    "06": "Hearing Loss, Primary Communication Auditory",
    "07": "Other Hearing Disabilities",
    "08": "Deaf-Blindness",
    "09": "Communicative Disabilities",
    "10": "Mobility Orthopedic/Neurological Disabilities",
    "11": "Manipulation/Dexterity Orthopedic/Neurological Disabilities",
    "12": "Both Mobility and Manipulation/Dexterity Orthopedic/Neurological Disabilities",
    "13": "Other Orthopedic Disabilities",
    "14": "Respiratory Disabilities",
    "15": "General Physical Debilitation",
    "16": "Other Physical Disabilities",
    "17": "Cognitive Disabilities",
    "18": "Psychosocial Disabilities",
    "19": "Other Mental Disabilities",
}

DISABILITY_SOURCE_LABELS = {
    "00": "Cause Unknown",
    "01": "Accident/Injury (other than TBI or SCI)",
    "02": "Alcohol Abuse or Dependence",
    "03": "Amputations",
    "04": "Anxiety Disorders",
    "05": "Arthritis and Rheumatism",
    "06": "Asthma and Other Allergies",
    "07": "Attention-Deficit Hyperactivity Disorder (ADHD)",
    "08": "Autism",
    "09": "Blood Disorders",
    "10": "Cancer",
    "11": "Cardiac and Other Conditions of the Circulatory System",
    "12": "Cerebral Palsy",
    "13": "Congenital Condition or Birth Injury",
    "14": "Cystic Fibrosis",
    "15": "Depressive and Other Mood Disorders",
    "16": "Diabetes Mellitus",
    "17": "Digestive",
    "18": "Drug Abuse or Dependence (other than alcohol)",
    "19": "Eating Disorders",
    "20": "End-Stage Renal Disease and Other Genitourinary System Disorders",
    "21": "Epilepsy",
    "22": "HIV or AIDS",
    "23": "Immune Deficiencies Excluding HIV or AIDS",
    "24": "Mental Illness (not listed elsewhere)",
    "25": "Intellectual Disability",
    "26": "Multiple Sclerosis",
    "27": "Muscular Dystrophy",
    "28": "Parkinson's Disease and Other Neurological Disorders",
    "29": "Personality Disorders",
    "30": "Physical Disorders/Conditions (not listed elsewhere)",
    "31": "Polio",
    "32": "Respiratory Disorders Other than Cystic Fibrosis or Asthma",
    "33": "Schizophrenia and Other Psychotic Disorders",
    "34": "Specific Learning Disabilities",
    "35": "Spinal Cord Injury (SCI)",
    "36": "Stroke",
    "37": "Traumatic Brain Injury (TBI)",
}


def _normalize_two_digit_code(code: str) -> str:
    if code.isdigit() and len(code) == 1:
        return f"0{code}"
    return code


def _label_primary_or_secondary_disability(value: Any) -> str:
    code = clean_code(value)
    if code == "":
        return MISSING_LABEL
    if ";" in code:
        disability_type, source = [part.strip() for part in code.split(";", 1)]
        disability_type = _normalize_two_digit_code(disability_type)
        source = _normalize_two_digit_code(source)
        type_label = DISABILITY_TYPE_LABELS.get(disability_type, f"Type {disability_type} label not verified")
        source_label = DISABILITY_SOURCE_LABELS.get(source, f"Source {source} label not verified")
        return f"{type_label}; source: {source_label}"
    normalized = _normalize_two_digit_code(code)
    return DISABILITY_TYPE_LABELS.get(normalized, f"Code {code} (label not verified)")


def clean_code(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def label_code(column: str, value: Any) -> str:
    code = clean_code(value)
    column = column.lower()
    if column in YES_NO_UNKNOWN_COLUMNS:
        return YES_NO_UNKNOWN_LABELS.get(code, f"Code {code} (label not verified)")
    if column in {"primdisability", "seconddisability"}:
        return _label_primary_or_secondary_disability(code)
    if column in {"primdisabilitycause", "seconddisabilitycause"}:
        normalized = _normalize_two_digit_code(code)
        return DISABILITY_SOURCE_LABELS.get(normalized, f"Code {code} (label not verified)")
    if column == "disabilitysigcode":
        return DISABILITY_SIGNIFICANCE_LABELS.get(code, f"Code {code} (label not verified)")
    if column == "ipeempstatus":
        return IPE_EMP_STATUS_LABELS.get(code, f"Code {code} (label not verified)")
    if column == "vocrehab":
        return VOC_REHAB_LABELS.get(code, f"Code {code} (label not verified)")
    if column == "exitempoutcome":
        return EXIT_EMP_OUTCOME_LABELS.get(code, f"Code {code} (label not verified)")
    if code == "":
        return MISSING_LABEL
    return f"Code {code} (label not verified)"


def codebook_note() -> str:
    return (
        "Code labels are based on the RSA-PD-19-03 RSA-911 codebook for mapped fields. "
        "Unmapped fields are marked as label not verified and should be checked against "
        "the official codebook before publication."
    )
