"""Generate the bundled SYNTHETIC resume sample (data/sample/resumes_sample.json).

The real Kaggle dataset ("Resume Entities for NER", ~220 resumes) needs a login, so this script builds a small
stand-in in exactly the same JSON-lines format (``content`` + ``annotation`` with ``label`` / ``points`` /
``start`` / ``end`` / ``text``; ``end`` is inclusive like in the original). All people, employers, colleges, e-mails and
phone numbers are fictional. The resumes exist so the pipeline, tests and demo run out of the box; they are NOT
real candidates and their statistics are not meaningful.

Usage:  python scripts/build_sample_resumes.py [--per-family 5] [--seed 11]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from talent_rag.lexicons import FEMALE_FIRST_NAMES, LAST_NAMES, MALE_FIRST_NAMES, NEUTRAL_FIRST_NAMES

SHARED_SKILLS = ["Git", "Agile", "Communication", "Teamwork", "Problem Solving"]

FAMILIES = {
    "data_analyst": {
        "designations": ("Junior Data Analyst", "Data Analyst", "Senior Data Analyst"),
        "core": ["SQL", "Excel", "Tableau", "Power BI", "Python", "Pandas", "Data Visualization", "Statistics", "ETL", "Google Analytics"],
        "bullets": [
            "Built interactive dashboards in {s1} to track sales and customer metrics.",
            "Wrote complex {s2} queries to clean, join and analyse large datasets.",
            "Automated weekly reporting with {s3}, saving about six hours per week.",
            "Presented data-driven insights to business stakeholders and recommended pricing changes.",
            "Designed KPI reports in {s1} for the marketing and operations teams.",
            "Validated data quality and documented metric definitions for the analytics team.",
        ],
        "degrees": ["B.Sc. Statistics", "B.Sc. Computer Science", "BBA Business Analytics", "M.Sc. Data Science"],
    },
    "java_developer": {
        "designations": ("Junior Java Developer", "Java Developer", "Senior Java Developer"),
        "core": ["Java", "Spring Boot", "Hibernate", "REST APIs", "MySQL", "Maven", "Microservices", "JUnit", "Docker", "Kafka"],
        "bullets": [
            "Developed backend services in {s1} and {s2} handling thousands of requests per minute.",
            "Designed REST APIs and integrated them with {s3} databases.",
            "Refactored a legacy monolith into microservices to improve deployment speed.",
            "Wrote unit and integration tests and improved code coverage above 80 percent.",
            "Optimised slow {s3} queries and reduced API latency by 40 percent.",
            "Participated in code reviews and mentored junior engineers on {s1} best practices.",
        ],
        "degrees": ["B.E. Computer Engineering", "B.Sc. Software Engineering", "BS Computer Science", "M.Sc. Computer Science"],
    },
    "web_developer": {
        "designations": ("Junior Web Developer", "Web Developer", "Senior Frontend Developer"),
        "core": ["JavaScript", "React", "HTML", "CSS", "Node.js", "TypeScript", "MongoDB", "REST APIs", "Redux", "Tailwind CSS"],
        "bullets": [
            "Built responsive user interfaces with {s1} and {s2} for e-commerce customers.",
            "Implemented reusable components and improved page load time by 35 percent.",
            "Connected the frontend to {s3} services and handled authentication flows.",
            "Collaborated with designers to turn mock-ups into accessible web pages.",
            "Wrote end-to-end tests and fixed cross-browser issues in {s1} applications.",
            "Maintained a component library used by four product teams.",
        ],
        "degrees": ["BS Software Engineering", "B.Sc. Information Technology", "BS Computer Science", "BCA"],
    },
    "ml_engineer": {
        "designations": ("Junior Machine Learning Engineer", "Machine Learning Engineer", "Senior Machine Learning Engineer"),
        "core": ["Python", "TensorFlow", "PyTorch", "scikit-learn", "NLP", "Deep Learning", "SQL", "Docker", "MLflow", "Computer Vision"],
        "bullets": [
            "Trained and evaluated {s1} models for text classification and recommendation tasks.",
            "Built data pipelines and feature engineering workflows with {s2} and {s3}.",
            "Deployed models as REST services in {s3} containers and monitored drift.",
            "Improved model accuracy by 8 percent through error analysis and hyper-parameter tuning.",
            "Fine-tuned transformer models for {s1} use cases and documented experiments.",
            "Worked with product managers to define evaluation metrics for ML features.",
        ],
        "degrees": ["M.Sc. Artificial Intelligence", "BS Computer Science", "M.Sc. Data Science", "B.E. Electronics"],
    },
    "devops_engineer": {
        "designations": ("Junior DevOps Engineer", "DevOps Engineer", "Senior DevOps Engineer"),
        "core": ["AWS", "Docker", "Kubernetes", "Jenkins", "Terraform", "Linux", "CI/CD", "Ansible", "Python", "Prometheus"],
        "bullets": [
            "Automated build and release pipelines with {s1} and {s2}, cutting deployment time by half.",
            "Managed cloud infrastructure on {s3} using infrastructure as code.",
            "Containerised services and operated {s2} clusters for production workloads.",
            "Set up monitoring and alerting and reduced incident response time.",
            "Hardened Linux servers and automated patching across environments.",
            "Documented runbooks and supported on-call rotations.",
        ],
        "degrees": ["B.E. Computer Engineering", "BS Information Technology", "B.Sc. Computer Science", "BS Software Engineering"],
    },
    "accountant": {
        "designations": ("Junior Accountant", "Accountant", "Senior Accountant"),
        "core": ["Tally", "QuickBooks", "Excel", "Taxation", "Auditing", "Financial Reporting", "SAP FICO", "Accounts Payable", "Budgeting", "GAAP"],
        "bullets": [
            "Maintained general ledgers and prepared monthly closing statements in {s1}.",
            "Prepared {s2} filings and reconciled bank statements and vendor accounts.",
            "Supported internal and external audits and resolved reported discrepancies.",
            "Produced budget versus actual reports for department heads using {s3}.",
            "Processed accounts payable and receivable and improved collection time.",
            "Trained new staff on accounting procedures and compliance requirements.",
        ],
        "degrees": ["B.Com", "M.Com", "ACCA (Affiliate)", "BBA Accounting and Finance"],
    },
    "hr_manager": {
        "designations": ("HR Executive", "HR Manager", "Senior HR Manager"),
        "core": ["Recruitment", "Talent Acquisition", "Payroll", "Employee Relations", "HRIS", "Onboarding", "Performance Management", "Training and Development", "Labour Law", "Excel"],
        "bullets": [
            "Managed end-to-end {s1} for technical and non-technical roles.",
            "Administered {s2} and maintained employee records in the HR system.",
            "Designed {s3} programmes that improved new-hire retention.",
            "Handled employee grievances and advised managers on policy matters.",
            "Ran annual appraisal cycles and prepared workforce analytics reports.",
            "Partnered with department heads to plan headcount and hiring budgets.",
        ],
        "degrees": ["MBA Human Resources", "BBA", "M.A. Psychology", "MBA"],
    },
    "mechanical_engineer": {
        "designations": ("Graduate Mechanical Engineer", "Mechanical Engineer", "Senior Mechanical Engineer"),
        "core": ["AutoCAD", "SolidWorks", "CATIA", "ANSYS", "GD&T", "Lean Manufacturing", "Six Sigma", "Project Management", "CNC", "Thermodynamics"],
        "bullets": [
            "Created detailed 3D models and drawings in {s1} and {s2} for new components.",
            "Performed stress and thermal analysis and reduced part weight by 12 percent.",
            "Applied {s3} methods to cut scrap and improve production efficiency.",
            "Coordinated with suppliers and the shop floor to resolve manufacturing issues.",
            "Prepared technical documentation and bills of materials for production.",
            "Led small projects from concept to commissioning within budget.",
        ],
        "degrees": ["B.E. Mechanical Engineering", "BS Mechanical Engineering", "M.Sc. Mechanical Engineering", "B.Tech Mechanical"],
    },
    "qa_engineer": {
        "designations": ("Junior QA Engineer", "QA Engineer", "Senior QA Automation Engineer"),
        "core": ["Selenium", "Java", "TestNG", "JIRA", "Manual Testing", "API Testing", "SQL", "Postman", "Cypress", "Test Automation"],
        "bullets": [
            "Designed and executed test cases for web applications and tracked defects in {s1}.",
            "Built automated regression suites with {s2} and {s3}.",
            "Tested REST APIs and validated data in the database with SQL queries.",
            "Worked in agile teams and joined sprint planning and retrospectives.",
            "Reduced regression testing time by 60 percent through test automation.",
            "Wrote clear bug reports and verified fixes before each release.",
        ],
        "degrees": ["BS Computer Science", "B.Sc. Information Technology", "BS Software Engineering", "BCA"],
    },
}

COMPANIES = [
    "Nexora Analytics", "Brightwell Systems", "Corvus Technologies", "Maple Retail Group", "Halden Software", "Pinecrest Bank",
    "Orion Consulting", "Bluepeak Solutions", "Zenith Industries", "Lakeview Health", "Northgate Telecom", "Riverstone Logistics",
    "Silvergate Financial", "Cobalt Labs", "Evergreen Manufacturing", "Quantum Retail", "Harbour Digital", "Sunrise Foods",
]
COLLEGES = [
    "Northfield Institute of Technology", "Riverside University", "Harbour City College", "Kingsbridge University",
    "Westport Institute of Management", "Ashford Technical University", "Lakeside College of Commerce", "Summit University",
    "Eastvale Engineering College", "Greenfield University",
]
CITIES = ["Islamabad", "Lahore", "Karachi", "Rawalpindi", "Peshawar", "Bangalore", "Hyderabad", "Pune", "London", "Dubai", "Toronto", "Berlin"]
MARITAL = ["Single", "Married"]


class Builder:
    """Accumulates text and records annotated entity spans (inclusive end, like the Kaggle file)."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.points: dict[str, list[dict]] = {}

    def add(self, text: str, label: str | None = None) -> None:
        if label:
            self.points.setdefault(label, []).append({"start": self.pos, "end": self.pos + len(text) - 1, "text": text})
        self.parts.append(text)
        self.pos += len(text)

    def line(self, text: str = "", label: str | None = None) -> None:
        self.add(text, label)
        self.add("\n")

    def record(self) -> dict:
        return {
            "content": "".join(self.parts),
            "annotation": [{"label": [label], "points": pts} for label, pts in self.points.items()],
            "extras": None,
        }


def make_resume(family: str, rng: random.Random, used_names: set[str]) -> dict:
    spec = FAMILIES[family]
    pool = MALE_FIRST_NAMES + FEMALE_FIRST_NAMES + NEUTRAL_FIRST_NAMES
    while True:
        first, last = rng.choice(pool), rng.choice(LAST_NAMES)
        if f"{first} {last}" not in used_names:
            used_names.add(f"{first} {last}")
            break
    pronoun = {"m": ("He", "his"), "f": ("She", "her")}.get("m" if first in MALE_FIRST_NAMES else "f" if first in FEMALE_FIRST_NAMES else "", None)
    years = rng.choice([1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 10])
    level = 0 if years <= 2 else 1 if years <= 5 else 2
    designation = spec["designations"][level]
    n_core = rng.randint(5, 7)
    core = rng.sample(spec["core"], n_core)
    skills = core + rng.sample(SHARED_SKILLS, rng.randint(1, 2))
    companies = rng.sample(COMPANIES, 2 if years > 2 else 1)
    degree = rng.choice(spec["degrees"])
    college = rng.choice(COLLEGES)
    grad_year = str(2024 - min(years, 10) - rng.randint(0, 1))
    city = rng.choice(CITIES)
    email = f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@examplemail.com"
    phone = f"+92 3{rng.randint(0, 4)}{rng.randint(0, 9)} {rng.randint(1000000, 9999999)}"

    b = Builder()
    b.line(f"{first} {last}", "Name")
    b.line(designation, "Designation")
    b.line()
    b.add("Email: ")
    b.line(email, "Email Address")
    b.line(f"Phone: {phone}")
    b.add("Location: ")
    b.line(city, "Location")
    b.line()
    b.line("Summary")
    exp = f"{years} year{'s' if years != 1 else ''}"
    if pronoun and rng.random() < 0.4:  # third-person summary with a gendered pronoun (used by the bias counterfactual test)
        b.add(f"{first} is a {designation.lower()} with ")
        b.add(exp, "Years of Experience")
        b.line(f" of experience in {core[0]} and {core[1]}. {pronoun[0]} enjoys solving practical problems and works closely with {pronoun[1]} team.")
    else:
        b.add(f"{designation} with ")
        b.add(exp, "Years of Experience")
        b.line(f" of experience in {core[0]}, {core[1]} and {core[2]}. Looking for challenging opportunities to grow.")
    b.line()
    b.line("Skills")
    b.add("", None)
    start_skills = b.pos
    skill_text = "\n".join(f"\u2022 {s}" for s in skills)
    b.points.setdefault("Skills", []).append({"start": start_skills, "end": start_skills + len(skill_text) - 1, "text": skill_text})
    b.parts.append(skill_text)
    b.pos += len(skill_text)
    b.add("\n\n")
    b.line("Experience")
    for i, company in enumerate(companies):
        b.add(designation if i == 0 else spec["designations"][max(level - 1, 0)], "Designation")
        b.add(" - ")
        b.add(company, "Companies worked at")
        b.line(f" ({2024 - years + i * 2}-{'Present' if i == 0 else 2024 - years + i * 2 + 2})")
        s = rng.sample(core, min(3, len(core)))
        s += [core[0]] * (3 - len(s))
        for tpl in rng.sample(spec["bullets"], 3):
            b.line("\u2022 " + tpl.format(s1=s[0], s2=s[1], s3=s[2]))
        b.line()
    b.line("Education")
    b.add(degree, "Degree")
    b.add(" - ")
    b.add(college, "College Name")
    b.add(", ")
    b.line(grad_year, "Graduation Year")
    if rng.random() < 0.5:  # typical personal-detail block that an anonymiser should remove
        b.line()
        b.line("Personal Details")
        b.line(f"Date of Birth: {rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(1985, 2001)}")
        b.line(f"Marital Status: {rng.choice(MARITAL)}")
        b.line("Nationality: Pakistani")
    return b.record()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "data" / "sample" / "resumes_sample.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    used: set[str] = set()
    records = [make_resume(fam, rng, used) for fam in FAMILIES for _ in range(args.per_family)]
    rng.shuffle(records)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} synthetic resumes to {out}")


if __name__ == "__main__":
    main()
