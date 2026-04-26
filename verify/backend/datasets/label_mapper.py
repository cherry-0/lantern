"""
Dataset-specific input label mappers for the Input-Output Comparison page.

Each mapper converts dataset-native labels/fields into a binary vector over the
unified attribute list (attribute_list_unified.txt).

Convention:
    get_input_labels(item, unified_attrs) -> dict[str, int]
        item         — item dict from iter_dataset()
        unified_attrs — ordered list of unified attribute names
        returns       — {attr: 0 or 1} for every attr in unified_attrs

Rules:
    - Be conservative: only mark 1 if clearly and directly supported.
    - Attributes not applicable for the item's modality are 0.
    - Cross-modality: text inputs have 0 for image-only attributes; vice versa.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── Attribute category membership ────────────────────────────────────────────
# Used to zero-fill non-applicable attributes in cross-modality scenarios.

_TEXT_ONLY_ATTRS = {"location", "identity", "marital status"}
_IMAGE_ONLY_ATTRS = {
    "face", "race", "nudity", "height", "weight", "disability",
    "ethnic_clothing", "religion", "medical", "sports", "formal",
    "uniforms", "casual", "color", "haircolor", "troupe",
}
_SHARED_ATTRS = {"age", "gender"}


# ── HR-VISPR mapper ───────────────────────────────────────────────────────────

def hrvispr_to_unified(privacy_labels: List[str], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Map HR-VISPR privacy_labels (list of positive attribute names from PKL) to
    the unified attribute list.  Direct 1:1 name match — no inference needed.

    Text-only attributes (location, identity, marital status) are 0 for
    HR-VISPR items since those attributes are not annotated in the dataset.
    """
    label_set = set(privacy_labels)
    return {attr: (1 if attr in label_set else 0) for attr in unified_attrs}


# ── SROIE2019 mapper ──────────────────────────────────────────────────────────

def sroie_to_unified(entities: Dict[str, str], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Map SROIE2019 entity fields to the unified attribute list.

    Mapping rules (conservative):
        company  → identity  (merchant/business name is a form of identity)
                 → formal    (presence of a company name on a receipt implies a
                              formal/business transaction context)
        address  → location  (store address directly reveals a geographic location)
        date     → (no mapping — date alone is not in the unified attribute list)
        total    → (no mapping — financial amount is not in the unified list)

    All image-only biometric attributes (face, race, nudity, …) are 0 because
    SROIE items are evaluated at the receipt-extraction level, not image-content
    level.
    """
    result = {attr: 0 for attr in unified_attrs}

    company = entities.get("company", "").strip()
    if company:
        if "identity" in result:
            result["identity"] = 1
        if "formal" in result:
            result["formal"] = 1

    address = entities.get("address", "").strip()
    if address:
        if "location" in result:
            result["location"] = 1

    return result


# ── PrivacyLens mapper ────────────────────────────────────────────────────────

# Conservative keyword → unified attribute mapping.
# Each entry: attribute name → list of lowercase substrings that clearly indicate it.
# A substring must appear in any of the relevant source fields to mark the attribute.
_PRIVACYLENS_KW: Dict[str, List[str]] = {
    "identity": [
        "identity", "name", "email", "username", "account", "user id",
        "personal info", "personal detail", "contact info", "phone number",
        "social security", "passport", "credential",
    ],
    "location": [
        "location", "address", "home address", "gps", "coordinates",
        "geographic", "whereabouts", "residence", "home",
    ],
    "age": [
        "age", "date of birth", "birthday", "birth date", "born in",
    ],
    "gender": [
        "gender", "sex ",      # note trailing space to avoid "sexual" false-positives
        "male", "female", "woman", "man ",
    ],
    "marital status": [
        "marital", "married", "spouse", "divorce", "widowed", "single",
        "partner", "relationship status",
    ],
    "medical": [
        "medical", "health", "diagnosis", "disease", "illness", "hospital",
        "prescription", "treatment", "patient", "therapy", "mental health",
    ],
    "religion": [
        "religion", "religious", "faith", "church", "mosque", "temple",
        "belief", "worship",
    ],
    "race": [
        "race", "ethnicity", "ethnic", "racial",
    ],
    # Image-side attributes are generally not inferable from PrivacyLens text
    # seeds/trajectories; only include if a seed explicitly names them.
    "face": ["biometric", "facial"],
    "nudity": ["nudity", "intimate image", "explicit image"],
    "disability": ["disability", "disabled", "handicap", "impairment"],
}


def privacylens_to_unified(item: Dict[str, Any], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Conservative mapping from a PrivacyLens item to the unified attribute list.

    Sources inspected:
        seed.data_type           — primary data category
        seed.data_subject        — who the data concerns
        trajectory.sensitive_info_items — explicit list of sensitive info strings

    Only attributes whose keywords clearly appear in one of these fields are
    marked 1.  All image-only biometric/clothing attributes default to 0 unless
    a keyword is found (very rare for text-only datasets).
    """
    result = {attr: 0 for attr in unified_attrs}

    seed = item.get("seed") or {}
    trajectory = item.get("trajectory") or {}

    # Collect all source strings, lowercased
    sources: List[str] = []
    for key in ("data_type", "data_subject", "data_sender", "transmission_principle"):
        val = seed.get(key, "") or ""
        if val.strip():
            sources.append(val.lower())

    sensitive_items: List[str] = trajectory.get("sensitive_info_items") or []
    for s in sensitive_items:
        if isinstance(s, str) and s.strip():
            sources.append(s.lower())

    # If no structured fields, also try item-level data_type_attributes (pre-built PKL mapping)
    prebuilt: List[str] = item.get("data_type_attributes") or []
    for attr in prebuilt:
        if attr in result:
            result[attr] = 1  # trust the pre-built mapping

    # Keyword scan
    for attr, keywords in _PRIVACYLENS_KW.items():
        if attr not in result:
            continue
        if result[attr] == 1:
            continue  # already set by prebuilt
        for src in sources:
            if any(kw in src for kw in keywords):
                result[attr] = 1
                break

    return result


# ── SynthPAI mapper ──────────────────────────────────────────────────────────

# Maps SynthPAI reviews.human field name → unified attribute name.
_SYNTHPAI_REVIEW_MAP: Dict[str, str] = {
    "age":                 "age",
    "sex":                 "gender",
    "city_country":        "location",
    "relationship_status": "marital status",
    "occupation":          "identity",
}


def synthpai_to_unified(item: Dict[str, Any], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Map SynthPAI per-post human annotations to the unified attribute list.

    Ground truth: reviews.human[field].certainty > 0  (human annotator was able
    to infer that attribute from the specific Reddit post).

    This is content-revealed labelling — an attribute is 1 only when a human
    reviewer read the post and could estimate the value with non-zero certainty.
    It is NOT the same as profile completeness (every SynthPAI profile has all
    fields filled in, so profile-based labels are always 1, which is misleading).

    Image-only biometric attributes (face, race, nudity, …) are always 0:
    SynthPAI items are text posts, not images.
    """
    result = {attr: 0 for attr in unified_attrs}

    raw    = item.get("raw") or {}
    human  = (raw.get("reviews") or {}).get("human") or {}

    for review_field, attr in _SYNTHPAI_REVIEW_MAP.items():
        if attr not in result:
            continue
        review    = human.get(review_field) or {}
        estimate  = str(review.get("estimate", "") or "").strip()
        certainty = float(review.get("certainty", 0) or 0)
        if estimate and estimate not in ("None", "null") and certainty > 0:
            result[attr] = 1

    return result


# ── OpenPII mapper ────────────────────────────────────────────────────────────

_OPENPII_LABEL_MAP: Dict[str, str] = {
    # identity — names, contact details, credentials
    "GIVENNAME": "identity", "SURNAME": "identity", "NAME": "identity",
    "NICKNAME": "identity", "USERNAME": "identity", "PREFIX": "identity",
    "EMAIL": "identity", "PHONE": "identity", "PHONENUMBER": "identity",
    "TELEFONNUMBER": "identity", "IDCARD": "identity", "PASS": "identity",
    "PASSPORT": "identity", "DRIVERSLICENSE": "identity", "SOCIALNUM": "identity",
    "SSN": "identity", "TAXNUM": "identity", "ACCOUNTNUM": "identity",
    "IBAN": "identity", "CREDITCARDNUMBER": "identity", "USERAGENT": "identity",
    "SECONDARY": "identity",
    # location — geographic identifiers
    "CITY": "location", "STREET": "location", "COUNTY": "location",
    "STATE": "location", "ZIPCODE": "location", "COUNTRY": "location",
    "POSTCODE": "location", "BUILDINGNUM": "location",
    # age
    "AGE": "age", "DATEOFBIRTH": "age",
    # gender
    "SEX": "gender",
}


def openpii_to_unified(item: Dict[str, Any], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Map OpenPII privacy_mask span labels to the unified attribute list.

    Each span in openpii_spans has: {label, start, end, value, label_index}.
    Labels are upper-cased PII type names (e.g. GIVENNAME, CITY, EMAIL).
    Only positive spans from _OPENPII_LABEL_MAP are mapped; everything else is 0.
    """
    result = {attr: 0 for attr in unified_attrs}
    for span in item.get("openpii_spans") or []:
        if not isinstance(span, dict):
            continue
        label = str(span.get("label", "")).upper()
        attr = _OPENPII_LABEL_MAP.get(label)
        if attr and attr in result:
            result[attr] = 1
    return result


# ── GretelSyntheticPII mapper ──────────────────────────────────────────────────

_GRETEL_PII_LABEL_MAP: Dict[str, str] = {
    # identity — names, contact, credentials
    "person": "identity", "person_name": "identity", "full_name": "identity",
    "first_name": "identity", "last_name": "identity", "name": "identity",
    "email_address": "identity", "email": "identity",
    "phone_number": "identity", "phone": "identity",
    "ssn": "identity", "social_security_number": "identity",
    "credit_card_number": "identity", "id_number": "identity",
    "passport_number": "identity", "drivers_license": "identity",
    "username": "identity", "url": "identity",
    # location
    "street_address": "location", "address": "location",
    "city": "location", "country": "location", "zip_code": "location",
    "state": "location", "location": "location",
    # age
    "date_of_birth": "age", "age": "age",
    # gender
    "gender": "gender",
    # medical
    "medical_condition": "medical", "diagnosis": "medical", "medication": "medical",
}


def gretel_pii_to_unified(item: Dict[str, Any], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Map GretelSyntheticPII pii_spans label values to the unified attribute list.

    Each span: {start, end, label}. Labels are lowercase snake_case type names.
    """
    result = {attr: 0 for attr in unified_attrs}
    for span in item.get("gretel_pii_spans") or []:
        if not isinstance(span, dict):
            continue
        label = str(span.get("label", "")).lower().strip()
        attr = _GRETEL_PII_LABEL_MAP.get(label)
        if attr and attr in result:
            result[attr] = 1
    return result


# ── Generic / HR-VISPR direct mapper ─────────────────────────────────────────

def generic_to_unified(privacy_labels: List[str], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Fallback mapper for datasets that already store attribute names in
    privacy_labels (e.g. HR-VISPR).  Equivalent to hrvispr_to_unified.
    """
    return hrvispr_to_unified(privacy_labels, unified_attrs)


# ── Public dispatcher ─────────────────────────────────────────────────────────

def get_input_labels(item: Dict[str, Any], unified_attrs: List[str]) -> Dict[str, int]:
    """
    Return binary input labels for the unified attribute list.

    Dispatches to the correct dataset-specific mapper based on item metadata:
        - label_source == "sroie_entities" → sroie_to_unified
        - vignette / seed present (PrivacyLens) → privacylens_to_unified
        - privacy_labels present (HR-VISPR / flat datasets) → hrvispr_to_unified
        - otherwise → all zeros
    """
    label_source = item.get("label_source", "")

    if label_source == "sroie_entities":
        entities = item.get("sroie_entities") or {}
        return sroie_to_unified(entities, unified_attrs)

    if label_source == "synthpai":
        return synthpai_to_unified(item, unified_attrs)

    if label_source == "openpii":
        return openpii_to_unified(item, unified_attrs)

    if label_source == "gretel_pii":
        return gretel_pii_to_unified(item, unified_attrs)

    if "seed" in item or "vignette" in item or "trajectory" in item:
        return privacylens_to_unified(item, unified_attrs)

    privacy_labels = item.get("privacy_labels") or []
    if privacy_labels:
        return hrvispr_to_unified(privacy_labels, unified_attrs)

    # Default: all zeros
    return {attr: 0 for attr in unified_attrs}
