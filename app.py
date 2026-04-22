import gradio as gr
import requests
import os
import re
import fitz
from docx import Document
from dotenv import load_dotenv
import hashlib
import time

# --- Load Environment Variables ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- Configuration ---
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# --- Cache for Deterministic Results ---
_cache = {}

def get_cache_key(text1, text2, prompt_type):
    """Generate a cache key for consistent results"""
    content = f"{prompt_type}:{text1[:1000]}:{text2[:1000]}"
    return hashlib.md5(content.encode()).hexdigest()

# --- Text Extraction ---
def extract_text(file):
    if file is None:
        return ""
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file.name)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = Document(file.name)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""

# --- Query LLM ---
def query_llm(prompt, max_tokens=600, cache_key=None):
    # Check cache first for deterministic results
    if cache_key and cache_key in _cache:
        return _cache[cache_key]

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0  # Deterministic results
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        # Cache the result
        if cache_key:
            _cache[cache_key] = result
        return result
    return f"API Error {response.status_code}: {response.text}"

# --- ATS Score ---
def get_ats_score(resume_text, job_text):
    cache_key = get_cache_key(resume_text, job_text, "ats_score")
    prompt = f"""You are an ATS scoring system. Compare the resume to the job description.

Return ONLY this format, nothing else:

SCORE: [number between 0 and 100]
SUMMARY: [one sentence explaining the score]

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_text[:2000]}"""

    result = query_llm(prompt, max_tokens=100, cache_key=cache_key)
    score = 0
    summary = ""

    for line in result.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = min(int("".join(filter(str.isdigit, line.split(":", 1)[1]))), 100)
            except:
                score = 0
        if line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    return score, summary

# --- Keyword Analysis ---
def get_keywords(resume_text, job_text):
    cache_key = get_cache_key(resume_text, job_text, "keywords")
    prompt = f"""You are an ATS keyword analyst. Extract the TOP 10 most important technical skills, tools, methodologies, and role-specific competencies from the job description. Focus on:

1. Technical skills (programming languages, frameworks, databases)
2. Tools and software
3. Methodologies (Agile, Scrum, DevOps)
4. Industry-specific terms
5. Certifications and qualifications

IMPORTANT: Return exactly 10 keywords maximum, ranked by importance. Do not include education degrees, company names, or generic soft skills.

From the JOB DESCRIPTION below, extract the most important keywords. Then check which ones appear in the RESUME.

Return ONLY this exact format, nothing else:

MATCHED: keyword1, keyword2, keyword3
MISSING: keyword1, keyword2, keyword3

JOB DESCRIPTION:
{job_text[:2000]}

RESUME:
{resume_text[:3000]}"""

    result = query_llm(prompt, max_tokens=200, cache_key=cache_key)
    matched = ""
    missing = ""

    for line in result.splitlines():
        line = line.strip()
        if line.startswith("MATCHED:"):
            matched = line.replace("MATCHED:", "").strip()
        if line.startswith("MISSING:"):
            missing = line.replace("MISSING:", "").strip()

    return matched, missing

# --- Improvement Suggestions ---
def get_suggestions(resume_text, job_text, matched_keywords=""):
    cache_key = get_cache_key(resume_text + matched_keywords, job_text, "suggestions")
    
    matched_list = ", ".join([k.strip() for k in matched_keywords.split(",") if k.strip()])
    matched_info = f"KEYWORDS ALREADY MATCHED IN RESUME:\n{matched_list}\n\n" if matched_list else ""
    
    prompt = f"""You are a professional resume coach presenting to an executive audience.

Analyze the resume against the job description and return exactly 3 specific, actionable improvement suggestions.

{matched_info}CRITICAL RULES - READ CAREFULLY:
1. ONLY suggest improvements for skills that are COMPLETELY MISSING from the resume.
2. NEVER recommend any of these matched keywords: {matched_list if matched_list else "(none)"}
3. Do NOT recommend anything from the matched keywords list - the resume already has these.
4. Focus ONLY on gaps that the job description requires but the resume completely lacks.
5. If the resume mentions experience that meets or exceeds a requirement, do NOT recommend it.
6. If there are fewer than 3 true gaps, suggest ways to strengthen presentation or impact.

Use this format exactly. Each suggestion must start with the number, followed by a short title in bold, then a colon, then one to two sentences of explanation:

**1. Short Title:** Explanation here.
**2. Short Title:** Explanation here.
**3. Short Title:** Explanation here.

Do not add any text before or after the three suggestions. Do not use asterisks anywhere except for the bold titles.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_text[:2000]}"""

    return query_llm(prompt, max_tokens=400, cache_key=cache_key)

# --- Score Display ---
def score_display(score, summary):
    if score >= 75:
        color = "#00C4B4"
        label = "Strong Match"
    elif score >= 50:
        color = "#FFD93D"
        label = "Moderate Match"
    else:
        color = "#FF6B6B"
        label = "Weak Match"

    return f"""
    <div style="border:1px solid #e0e0e0; border-radius:8px; padding:24px;
                margin-bottom:16px; background:#fafafa; text-align:left;">
        <div style="display:flex; align-items:center; gap:24px;">
            <div style="text-align:center; flex-shrink:0;">
                <div style="font-size:48px; font-weight:700; color:{color};
                            line-height:1;">{score}</div>
                <div style="font-size:13px; color:#888; margin-top:4px;">out of 100</div>
            </div>
            <div style="flex:1;">
                <div style="font-size:18px; font-weight:700; color:{color};
                            margin-bottom:6px;">{label}</div>
                <div style="font-size:14px; color:#555; line-height:1.6;">{summary}</div>
            </div>
        </div>
    </div>
    """

# --- Keyword Display ---
def keyword_display(matched, missing):
    def is_empty(val):
        return not val or val.strip().lower() in ("", "none", "n/a")

    html = '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; margin-bottom:16px; text-align:left;">'

    if not is_empty(matched):
        html += '<div style="margin-bottom:16px;">'
        html += '<div style="font-weight:700; margin-bottom:10px; color:#00C4B4; font-size:14px;">Matched Keywords</div>'
        html += '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        for kw in [k.strip() for k in matched.split(",") if k.strip()][:12]:
            html += f'<span style="background:#e8faf9; color:#00C4B4; padding:4px 12px; border-radius:4px; font-size:13px; border:1px solid #00C4B4;">{kw}</span>'
        html += '</div></div>'

    if not is_empty(missing):
        html += '<div>'
        html += '<div style="font-weight:700; margin-bottom:10px; color:#FF6B6B; font-size:14px;">Missing Keywords</div>'
        html += '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        for kw in [k.strip() for k in missing.split(",") if k.strip()][:12]:
            html += f'<span style="background:#fff0f0; color:#FF6B6B; padding:4px 12px; border-radius:4px; font-size:13px; border:1px solid #FF6B6B;">{kw}</span>'
        html += '</div></div>'

    html += '</div>'

    if is_empty(matched) and is_empty(missing):
        return '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">No keywords analyzed yet.</div>'

    return html

# --- Suggestions Display ---
def suggestions_display(text):
    if not text or text.startswith("Please provide"):
        return '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Suggestions will appear after analysis.</div>'

    text = text.strip()
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace('\n', ' ')
    parts = re.split(r'(?=\s*[2-9]\.\s)', text)
    text = '<br><br>'.join(p.strip() for p in parts if p.strip())

    return f'<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; text-align:left;"><div style="font-weight:700; margin:0; color:#00C4B4; font-size:14px;">Improvement Recommendations</div><div style="color:#444; line-height:1.8; font-size:14px; margin-top:5px;">{text}</div></div>'

# --- Loading State ---
def loading_state():
    loading_html = '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#F97316; text-align:left; font-size:14px;">Analyzing... please wait.</div>'
    return loading_html, loading_html, loading_html

# --- Main Analysis ---
def analyze(resume_file, resume_paste, job_description):
    time.sleep(1)
    if resume_file is not None:
        resume_text = extract_text(resume_file)
    else:
        resume_text = resume_paste.strip() if resume_paste else ""

    if not resume_text:
        return (
            '<div style="border:1px solid #FF6B6B; border-radius:8px; padding:20px; background:#fff0f0; color:#FF6B6B; text-align:left;">Please upload a resume file or paste your resume text.</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Awaiting resume input...</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Awaiting resume input...</div>'
        )

    if not job_description or not job_description.strip():
        return (
            '<div style="border:1px solid #FF6B6B; border-radius:8px; padding:20px; background:#fff0f0; color:#FF6B6B; text-align:left;">Please paste a job description.</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Awaiting job description...</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Awaiting job description...</div>'
        )

    score, summary = get_ats_score(resume_text, job_description)
    matched, missing = get_keywords(resume_text, job_description)
    suggestions = get_suggestions(resume_text, job_description, matched)

    return (
        score_display(score, summary),
        keyword_display(matched, missing),
        suggestions_display(suggestions)
    )

# --- CSS ---
css = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

button.primary {
    background-color: #F97316 !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

button.primary:hover {
    background-color: #EA6C0A !important;
}

label {
    font-weight: 500 !important;
    color: #333 !important;
}

textarea, input {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input:focus {
    border-color: #F97316 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.1) !important;
    background-color: #ffffff !important;
}

textarea:hover, input:hover {
    border-color: #F97316 !important;
}

.tab-nav button {
    font-size: 13px !important;
    font-weight: 500 !important;
}

.tab-nav button.selected {
    border-bottom: 2px solid #F97316 !important;
    color: #F97316 !important;
}

footer {
    display: none !important;
}

.gr-html, .gr-html > div {
    text-align: left !important;
}

"""

# --- Gradio Interface ---
with gr.Blocks(css=css, theme=gr.themes.Base(), title="ResumeIQ") as demo:

    gr.HTML("""
        <div style="text-align:center; padding:5px 0 5px 0;">
            <h1 style="font-size:42px; font-weight:800; margin-bottom:8px;
                       color:#1a1a1a; letter-spacing:-0.5px;">ResumeIQ</h1>
            <p style="font-size:16px; color:#555; margin-bottom:8px;">
                AI-powered resume analysis and ATS match scoring
            </p>
            <p style="font-size:14px; color:#888; max-width:700px; margin:0 auto;">
                Upload your resume or paste text to receive your ATS score,
                keyword analysis, and targeted improvement suggestions.
            </p>
        </div>
    """)

    gr.Markdown("---")

    with gr.Row():

        # --- Left Panel ---
        with gr.Column(scale=1):
            gr.Markdown("## Resume")

            with gr.Tabs():
                with gr.TabItem("Upload File"):
                    resume_file = gr.File(
                        label="Upload Resume (PDF or DOCX)",
                        file_types=[".pdf", ".docx"],
                        height=120
                    )
                with gr.TabItem("Paste Text"):
                    resume_text = gr.Textbox(
                        label="Paste Resume Text",
                        lines=8,
                        placeholder="Paste your resume content here..."
                    )

            gr.Markdown("## Job Description")
            job_input = gr.Textbox(
                label="Job Description",
                lines=12,
                placeholder="Paste the job description here..."
            )
            analyze_btn = gr.Button("Analyze Resume", variant="primary")

            gr.HTML("""
                <div style="background:#f8f9fa; border-radius:8px; padding:16px;
                            margin-top:16px; border:1px solid #e0e0e0;">
                    <p style="font-weight:600; margin:0 0 10px 0; color:#333;
                               font-size:14px;">💡 Tips for best results</p>
                    <ul style="color:#666; font-size:13px; line-height:1.8; margin:0; padding-left:20px;">
                        <li>Include complete work experience with dates</li>
                        <li>List technical skills and tools you know</li>
                        <li>Add measurable achievements (%, $, numbers)</li>
                        <li>Tailor your resume to each job description</li>
                        <li>Use standard headings: Experience, Skills, Education</li>
                    </ul>
                </div>
            """)

        # --- Right Panel ---
        with gr.Column(scale=1):
            gr.Markdown("## Results")
            score_output = gr.HTML(
                value='<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">See your ATS match score after running the analysis.</div>'
            )
            keywords_output = gr.HTML(
                value='<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Get your matched keywords and uncover what’s missing.</div>'
            )
            suggestions_output = gr.HTML(
                value='<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa; text-align:left;">Receive tailored suggestions to strengthen your resume.</div>'
            )

    analyze_btn.click(
        fn=loading_state,
        inputs=None,
        outputs=[score_output, keywords_output, suggestions_output],
        queue=False
    ).then(
        fn=analyze,
        inputs=[resume_file, resume_text, job_input],
        outputs=[score_output, keywords_output, suggestions_output]
    )

    gr.Markdown("---")

    gr.Markdown("""
    <div style="text-align:center; color:#888; font-size:13px;">
        Built with Gradio &middot; Python &middot; HuggingFace LLMs &nbsp;&middot;&nbsp;
        <a href="https://github.com/kduffuor/ResumeIQ" target="_blank"
           style="color:#888; text-decoration:none;">GitHub</a> &nbsp;&middot;&nbsp;
        <a href="https://linkedin.com/in/kduffuor" target="_blank"
           style="color:#888; text-decoration:none;">LinkedIn</a>
    </div>
    """)

if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        demo.launch(server_name="0.0.0.0")
    else:
        demo.launch()
