from backend.llm.analyze_v3 import extract_tools_from_jd


def test_extract_tools_from_jd_devsecops_line():
    page_text = (
        "DevSecOps: GitHub/GitLab/Azure DevOps, Actions/Pipelines, IaC (Terraform), "
        "SAST/DAST/SCA, secrets management"
    )
    tools = extract_tools_from_jd(page_text)
    # Preserve literal groups instead of expanding them through an IT-specific
    # alias table (for example, "Actions" must not invent "GitHub Actions").
    for expected in [
        "GitHub/GitLab/Azure DevOps",
        "Actions/Pipelines",
        "IaC",
        "Terraform",
        "SAST/DAST/SCA",
        "secrets management",
    ]:
        assert expected in tools


def test_extract_tools_from_non_technical_structured_and_direct_requirements():
    page_text = """Software: Microsoft Excel, Salesforce
    Certifications: CPA, CFA
    Experience with Tableau and Power BI."""

    tools = extract_tools_from_jd(page_text)

    assert tools == ["Microsoft Excel", "Salesforce", "CPA", "CFA", "Tableau", "Power BI"]
