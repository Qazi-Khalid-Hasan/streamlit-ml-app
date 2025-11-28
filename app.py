# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional parsers
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx2txt
except Exception:
    docx2txt = None

st.set_page_config(page_title="Qazi Resume Analyzer", layout="wide")
st.title("Qazi Resume Analyzer")
st.markdown(
    "Upload your resume (PDF / DOCX / TXT) and paste a job description. "
    "This app will extract skills, compute a skill match, and provide an overall similarity score."
)

# Simple skills dictionary (starter) — you can expand this list
COMMON_SKILLS = [
    "python", "r", "sql", "excel", "pandas", "numpy", "scikit-learn", "tensor*",
    "deep learning", "machine learning", "data analysis", "data visualization", "power bi",
    "tableau", "communication", "leadership", "project management", "aws", "azure", "gcp",
    "docker", "kubernetes", "git", "nlp", "computer vision", "statistics", "regression",
    "classification", "time series", "spark", "hadoop", "matlab", "sas", "keras", "pytorch",
    "openai", "transformer", "bash", "linux", "javascript", "react", "flask", "django"
]

def extract_text_from_pdf(file_bytes):
    if pdfplumber is None:
        return ""
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes):
    if docx2txt is None:
        return ""
    # docx2txt only reads from a path; write to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    text = docx2txt.process(tmp_path)
    return text or ""

def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(data)
    elif name.endswith(".txt") or name.endswith(".csv"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return str(data)
    else:
        # fallback: try decode
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-\.\,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_skills(text, skills_list):
    found = set()
    text_l = text.lower()
    for s in skills_list:
        # support wildcard token like tensor*
        if "*" in s:
            prefix = s.replace("*", "")
            if prefix and prefix in text_l:
                found.add(s.replace("*", ""))
        else:
            # word boundary search
            pattern = r"\b" + re.escape(s.lower()) + r"\b"
            if re.search(pattern, text_l):
                found.add(s.lower())
    return sorted(found)

def compute_similarity_score(resume_text, job_text):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    docs = [resume_text, job_text]
    tfidf = vec.fit_transform(docs)
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(score)

# UI columns
col1, col2 = st.columns([1,2])

with col1:
    st.header("Upload Resume")
    uploaded = st.file_uploader("Choose resume file (PDF, DOCX, TXT)", type=["pdf", "docx", "doc", "txt"])
    st.markdown("**Pro tip:** Small resumes (<2 MB) upload faster. If your PDF is image-only, text extraction may fail.")
    if st.checkbox("Use sample resume text (demo)"):
        uploaded = None
        sample_resume = """
        John Doe
        Data Scientist with 4 years experience in python, pandas, numpy, scikit-learn, and machine learning.
        Experience with aws, docker and SQL. Built predictive models, classification and regression systems.
        """
        resume_text = sample_resume
    else:
        resume_text = extract_text(uploaded)

with col2:
    st.header("Job Description")
    job_text = st.text_area("Paste the job description here (or a job link text)", height=300)
    if st.button("Analyze"):
        if not resume_text and uploaded is None:
            st.error("Please upload a resume (or check the sample resume checkbox).")
        elif not job_text.strip():
            st.error("Please paste a job description to compare.")
        else:
            # process
            rnorm = normalize_text(resume_text)
            jnorm = normalize_text(job_text)
            st.subheader("Resume preview (extracted)")
            st.write(rnorm[:4000] + ("..." if len(rnorm) > 4000 else ""))
            st.subheader("Job description preview")
            st.write(jnorm[:4000] + ("..." if len(jnorm) > 4000 else ""))

            # skill extraction
            resume_skills = find_skills(rnorm, COMMON_SKILLS)
            job_skills = find_skills(jnorm, COMMON_SKILLS)

            # skill overlap
            resume_set = set(resume_skills)
            job_set = set(job_skills)
            intersect = resume_set.intersection(job_set)
            if job_set:
                skill_match_pct = len(intersect) / len(job_set) * 100
            else:
                skill_match_pct = 0.0

            # similarity score
            sim = compute_similarity_score(rnorm, jnorm)

            # Show metrics
            st.metric("Overall similarity (TF-IDF cosine)", f"{sim:.2f}")
            st.metric("Skill match % (found in resume vs job)", f"{skill_match_pct:.0f}%")

            st.write("### Skills found in job description")
            st.write(sorted(job_skills))
            st.write("### Skills found in resume")
            st.write(sorted(resume_skills))

            st.write("### Matched skills")
            st.write(sorted(list(intersect)))

            # Suggestions: missing skills
            missing = sorted(list(job_set - resume_set))
            st.write("### Suggested skills to add to resume (if relevant)")
            if missing:
                st.info(", ".join(missing))
            else:
                st.success("No missing skills detected (based on the skill list).")

            # Simple resume summary improvement (smart template)
            st.write("### Suggested improved summary (editable)")
            # get first line or create simple summary
            first_line = rnorm.split(".")[0] if rnorm else ""
            suggested_summary = f"{first_line}. Skilled in {' '.join(resume_skills[:6])}." if resume_skills else first_line
            st.text_area("Improved summary (copy & paste into your resume)", value=suggested_summary, height=120)

            # allow downloading a small CSV report
            report = {
                "metric": ["similarity", "skill_match_pct", "resume_skills", "job_skills", "matched_skills"],
                "value":[f"{sim:.3f}", f"{skill_match_pct:.1f}", ";".join(resume_skills), ";".join(job_skills), ";".join(intersect)]
            }
            df_report = pd.DataFrame(report)
            csv = df_report.to_csv(index=False).encode("utf-8")
            st.download_button("Download report (CSV)", csv, file_name="qazi_resume_report.csv", mime="text/csv")
