import time

from backend.ir.schema_v3 import AnalyzeIRv3
from backend.llm.analyze_v3 import _derive_candidate_level_from_experience
from backend.scoring.scoring_engine_v3 import score_ir_v3, score_to_public_dict


def _month_name(month: int) -> str:
    return (
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    )[month - 1]


def test_associate_with_two_years_relevant_experience_is_junior_to_mid():
    now = time.gmtime()
    start_year = now.tm_year - 2
    month = _month_name(now.tm_mon)
    resume = f"""
EXPERIENCE
Cloud Support Associate – AI/ML | Amazon Web Services
{month} {start_year} – PRESENT
- Investigate and troubleshoot AWS cloud, networking, IAM, and performance issues.
- Guide customers on reliable cloud practices and collaborate with engineering teams.

EARLIER EXPERIENCE
Commercial Analyst | Example Co - JAN 2021 – JUN 2024
- Built financial models, forecasts, and management reports.
"""

    for family in ("tech_swe", "tech_data"):
        level, reason, entries = _derive_candidate_level_from_experience(
            resume, job_family=family
        )

        assert level == "junior_to_mid"
        assert "relevant_experience_24m" in reason
        assert entries[0]["title"].startswith("Cloud Support Associate")
        assert all("Commercial Analyst" not in entry["title"] for entry in entries)


def test_explicit_recent_senior_title_remains_strong_evidence():
    now = time.gmtime()
    start_year = now.tm_year - 1
    month = _month_name(now.tm_mon)
    resume = f"""
Senior Cloud Engineer | Example Co
{month} {start_year} – PRESENT
- Own cloud infrastructure and mentor engineers.
"""

    level, reason, _entries = _derive_candidate_level_from_experience(
        resume, job_family="tech_swe"
    )

    assert level == "senior"
    assert reason == "recent_explicit_12m"


def test_education_date_range_is_not_counted_as_work_experience():
    now = time.gmtime()
    start_year = now.tm_year - 2
    month = _month_name(now.tm_mon)
    resume = f"""
PROFESSIONAL EXPERIENCE
Cloud Support Associate | Example
{month} {start_year} – PRESENT
- Troubleshoot cloud services and guide customers.

EDUCATION
Diploma of Applied Technology | Example Institute
{month} {start_year - 2} – {month} {start_year}
"""

    level, reason, entries = _derive_candidate_level_from_experience(
        resume, job_family="tech_swe"
    )

    assert level == "junior_to_mid"
    assert "relevant_experience_24m" in reason
    assert len(entries) == 1
    assert "Cloud Support Associate" in entries[0]["title"]


def test_unknown_candidate_seniority_stays_unknown_and_neutral():
    ir = AnalyzeIRv3(
        job_title="Cloud Engineer",
        company="Example",
        job_seniority_signal="mid",
        candidate_seniority_signal="unknown",
    )

    result = score_ir_v3(ir)
    constraints = result.debug_breakdown.constraints

    assert constraints["cand_level"] == "unknown"
    assert constraints["cand_level_reason"] == "unknown_neutral_no_seniority_evidence"
    assert constraints["gap"] == 0.0


def test_junior_to_mid_band_uses_numeric_half_step():
    ir = AnalyzeIRv3(
        job_title="Cloud Engineer",
        company="Example",
        job_seniority_signal="mid",
        candidate_seniority_signal="junior_to_mid",
    )

    result = score_ir_v3(ir)
    constraints = result.debug_breakdown.constraints

    assert constraints["cand_level"] == "junior"
    assert constraints["cand_numeric_level"] == 1.5
    assert constraints["cand_level_reason"] == "experience_band_junior_to_mid"
    contract = score_to_public_dict(result)
    seniority_layer = next(layer for layer in contract["layers"] if layer["id"] == "seniority")
    assert seniority_layer["meta"]["candidate_level"] == "junior_to_mid"
