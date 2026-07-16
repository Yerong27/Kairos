from backend.llm.analyze_v3 import extract_tools_from_jd


def test_extract_tools_from_jd_devsecops_line():
    page_text = (
        "DevSecOps: GitHub/GitLab/Azure DevOps, Actions/Pipelines, IaC (Terraform), "
        "SAST/DAST/SCA, secrets management"
    )
    tools = extract_tools_from_jd(page_text)
    for expected in [
        "GitHub",
        "GitLab",
        "Azure DevOps",
        "GitHub Actions",
        "Terraform",
        "SAST",
        "DAST",
        "SCA",
        "Secrets Management",
    ]:
        assert expected in tools
