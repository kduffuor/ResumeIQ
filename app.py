import gradio as gr
import requests
import os
import fitz
from docx import Document
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- Configuration ---
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

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
def query_llm(prompt, max_tokens=600):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return f"API Error {response.status_code}: {response.text}"

# --- ATS Score ---
def get_ats_score(resume_text, job_text):
    prompt = f"""You are an ATS scoring system. Compare the resume to the job description.

Return ONLY this format, nothing else:

SCORE: [number between 0 and 100]
SUMMARY: [one sentence explaining the score]

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_text[:1500]}"""

    result = query_llm(prompt, max_tokens=100)
    score = 0
    summary = ""

    for line in result.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = int("".join(filter(str.isdigit, line.split(":", 1)[1])))
            except:
                score = 0
        if line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    return score, summary

# --- Keyword Analysis ---
def get_keywords(resume_text, job_text):
    prompt = f"""You are an ATS keyword analyst.

Return ONLY this format, nothing else:

MATCHED: keyword1, keyword2, keyword3
MISSING: keyword1, keyword2, keyword3

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_text[:1500]}"""

    result = query_llm(prompt, max_tokens=150)
    matched = ""
    missing = ""

    for line in result.splitlines():
        if line.startswith("MATCHED:"):
            matched = line.split(":", 1)[1].strip()
        if line.startswith("MISSING:"):
            missing = line.split(":", 1)[1].strip()

    return matched, missing

# --- Suggestions ---
def get_suggestions(resume_text, job_text):
    prompt = f"""You are a professional resume coach presenting to an executive audience.

Analyze the resume against the job description and return exactly 3 specific,
actionable improvement suggestions.

Use this format exactly for each suggestion:
**1. [Short title]:** [One to two sentence explanation in plain prose.]
**2. [Short title]:** [One to two sentence explanation in plain prose.]
**3. [Short title]:** [One to two sentence explanation in plain prose.]

Do not add any text before or after the three suggestions.

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_text[:1500]}"""

    return query_llm(prompt, max_tokens=300)

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
                margin-bottom:16px; background:#fafafa;">
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
    html = '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; margin-bottom:16px;">'

    if matched:
        html += '<div style="margin-bottom:16px;">'
        html += '<div style="font-weight:700; margin-bottom:10px; color:#00C4B4; font-size:14px;">Matched Keywords</div>'
        html += '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        for kw in [k.strip() for k in matched.split(",") if k.strip()][:12]:
            html += f'<span style="background:#e8faf9; color:#00C4B4; padding:4px 12px; border-radius:4px; font-size:13px; border:1px solid #00C4B4;">{kw}</span>'
        html += '</div></div>'

    if missing:
        html += '<div>'
        html += '<div style="font-weight:700; margin-bottom:10px; color:#FF6B6B; font-size:14px;">Missing Keywords</div>'
        html += '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        for kw in [k.strip() for k in missing.split(",") if k.strip()][:12]:
            html += f'<span style="background:#fff0f0; color:#FF6B6B; padding:4px 12px; border-radius:4px; font-size:13px; border:1px solid #FF6B6B;">{kw}</span>'
        html += '</div></div>'

    html += '</div>'

    if not matched and not missing:
        return '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">No keywords analyzed yet.</div>'
    return html

# --- Suggestions Display ---
def suggestions_display(text):
    if not text or text.startswith("Please provide"):
        return '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Suggestions will appear after analysis.</div>'

    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    return f"""
    <div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px;">
        <div style="font-weight:700; margin-bottom:14px; color:#00C4B4; font-size:14px;">
            Improvement Recommendations
        </div>
        <div style="color:#444; line-height:1.8; font-size:14px;">
            {text.replace(chr(10), "<br>")}
        </div>
    </div>
    """

# --- Main Analysis ---
def analyze(resume_file, resume_paste, job_description):
    # Check if file was uploaded
    if resume_file is not None:
        resume_text = extract_text(resume_file)
    else:
        resume_text = resume_paste.strip() if resume_paste else ""
    
    if not resume_text:
        return (
            '<div style="border:1px solid #FF6B6B; border-radius:8px; padding:20px; background:#fff0f0; color:#FF6B6B;">Please upload a resume file or paste your resume text.</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Awaiting resume input...</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Awaiting resume input...</div>'
        )

    if not job_description or not job_description.strip():
        return (
            '<div style="border:1px solid #FF6B6B; border-radius:8px; padding:20px; background:#fff0f0; color:#FF6B6B;">Please paste a job description.</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Awaiting job description...</div>',
            '<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Awaiting job description...</div>'
        )

    score, summary = get_ats_score(resume_text, job_description)
    matched, missing = get_keywords(resume_text, job_description)
    suggestions = get_suggestions(resume_text, job_description)

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

footer {
    display: none !important;
}
"""

# --- Gradio Interface ---
with gr.Blocks(css=css, theme=gr.themes.Base(), title="ResumeIQ") as demo:

    gr.HTML("""
        <div style="text-align:center; padding:32px 0 16px 0;">
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

        # --- Right Panel ---
        with gr.Column(scale=1):
            gr.Markdown("## Results")
            score_output = gr.HTML(
                value='<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Run analysis to see your ATS match score.</div>'
            )
            keywords_output = gr.HTML(
                value='<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Get matched and missing keyword analysis.</div>'
            )
            suggestions_output = gr.HTML(
                value='<div style="border:1px solid #e0e0e0; border-radius:8px; padding:20px; color:#aaa;">Improve your resume with tailored suggestions.</div>'
            )

    analyze_btn.click(
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