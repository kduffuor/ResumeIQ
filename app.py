import gradio as gr
import requests
import os
import re
import fitz
import docx
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
        doc = docx.Document(file.name)
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
        color = "#2E7D32"
        border = "#4CAF50"
        label = "Strong Match"
    elif score >= 50:
        color = "#ED6C02"
        border = "#FF9800"
        label = "Moderate Match"
    else:
        color = "#D32F2F"
        border = "#F44336"
        label = "Weak Match"

    return f"""
    <div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 24px; margin-bottom: 24px; background: #FAFAFA;">
        <div style="display: flex; align-items: center; gap: 24px;">
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: 600; color: {color};">{score}</div>
                <div style="font-size: 14px; color: #666;">out of 100</div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 18px; font-weight: 600; color: {color}; margin-bottom: 8px;">{label}</div>
                <div style="font-size: 14px; color: #555; line-height: 1.5;">{summary}</div>
            </div>
        </div>
    </div>
    """

# --- Keyword Display ---
def keyword_display(matched, missing):
    html = '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; margin-bottom: 24px;">'
    
    if matched:
        html += '<div style="margin-bottom: 20px;">'
        html += '<div style="font-weight: 600; margin-bottom: 12px; color: #2E7D32;">Matched Keywords</div>'
        html += '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for kw in [k.strip() for k in matched.split(',')[:10]]:
            html += f'<span style="background: #E8F5E9; color: #2E7D32; padding: 4px 12px; border-radius: 4px; font-size: 13px;">{kw}</span>'
        html += '</div></div>'
    
    if missing:
        html += '<div>'
        html += '<div style="font-weight: 600; margin-bottom: 12px; color: #D32F2F;">Missing Keywords</div>'
        html += '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for kw in [k.strip() for k in missing.split(',')[:10]]:
            html += f'<span style="background: #FFEBEE; color: #D32F2F; padding: 4px 12px; border-radius: 4px; font-size: 13px;">{kw}</span>'
        html += '</div></div>'
    
    html += '</div>'
    return html if (matched or missing) else '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">No keywords analyzed yet.</div>'

# --- Suggestions Display ---
def suggestions_display(text):
    if not text or text.startswith("Please provide"):
        return '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Suggestions will appear after analysis.</div>'
    
    # Markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return f"""
    <div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px;">
        <div style="font-weight: 600; margin-bottom: 16px; color: #1976D2;">Improvement Recommendations</div>
        <div style="color: #444; line-height: 1.6;">
            {text.replace(chr(10), '<br>')}
        </div>
    </div>
    """

# --- Main Analysis ---
def analyze(file, resume_paste, job_description):
    resume_text = ""

    if file is not None:
        resume_text = extract_text(file)
    elif resume_paste and resume_paste.strip():
        resume_text = resume_paste.strip()

    if not resume_text:
        return (
            '<div style="border: 1px solid #FFCDD2; border-radius: 8px; padding: 20px; background: #FFEBEE; color: #D32F2F;">Error: Please upload a resume or paste your resume text.</div>',
            '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Awaiting resume input...</div>',
            '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Awaiting resume input...</div>'
        )

    if not job_description or not job_description.strip():
        return (
            '<div style="border: 1px solid #FFCDD2; border-radius: 8px; padding: 20px; background: #FFEBEE; color: #D32F2F;">Error: Please paste a job description.</div>',
            '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Awaiting job description...</div>',
            '<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Awaiting job description...</div>'
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

.gr-button-primary {
    background-color: #F97316 !important;
    border: none !important;
    font-weight: 600 !important;
}

.gr-button-primary:hover {
    background-color: ##EA6C0A !important;
}

.gr-box, .gr-form {
    border: 1px solid #E0E0E0 !important;
    border-radius: 8px !important;
}

label {
    font-weight: 500 !important;
    color: #333 !important;
}

input, textarea {
    border: 1px solid #E0E0E0 !important;
    border-radius: 6px !important;
    font-family: monospace !important;
    font-size: 13px !important;
}

input:focus, textarea:focus {
    border-color: #1976D2 !important;
    outline: none !important;
}
"""

# --- Gradio Interface ---
with gr.Blocks(css=css, title="ResumeIQ") as demo:
    
    # Header - ResumeIQ
    gr.HTML("""
        <div style="text-align: center; padding: 10px 0 10px 0;">
            <h1 style="font-size: 42px; font-weight: 800; margin-bottom: 8px; color: #1a1a1a;">ResumeIQ</h1>
            <p style="font-size: 16px; color: #555; margin-bottom: 8px;">AI-powered resume analysis and ATS match scoring</p>
            <p style="font-size: 14px; color: #888; max-width: 700px; margin: 0 auto;">
                Upload your resume and paste the job description to receive your ATS score, keyword analysis, and targeted improvement suggestions.
            </p>
        </div>
    """)
    
    gr.Markdown("---")
    
    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            gr.Markdown("## Resume")
            
            file_input = gr.UploadButton(
            label="Upload PDF or DOCX",
            file_types=[".pdf", ".docx"],
            file_count="single"
            )
            
            gr.Markdown("— OR —")
            
            resume_text = gr.Textbox(
                label="Paste Resume Text",
                lines=12,
                placeholder="Paste your resume content here..."
            )
            
            gr.Markdown("## Job Description")
            
            job_input = gr.Textbox(
                label="Job Description",
                lines=12,
                placeholder="Paste the job description here..."
            )
            
            analyze_btn = gr.Button("Analyze Resume", variant="primary")
        
        # Output Column
        with gr.Column(scale=1):
            gr.Markdown("## Results")
            
            score_output = gr.HTML(
                value='<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Run analysis to see your ATS match score</div>'
            )
            
            keywords_output = gr.HTML(
                value='<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Get keyword comparison results</div>'
            )
            
            suggestions_output = gr.HTML(
                value='<div style="border: 1px solid #E0E0E0; border-radius: 8px; padding: 20px; color: #666;">Get targeted improvement suggestions after analysis</div>'
            )
    
    gr.Markdown("---")
    
    gr.Markdown("""
    <div style="text-align: center; color: #666; font-size: 13px;">
        Powered by Llama 3.1 8B • ATS Keyword Analysis • Match Scoring • 
        <a href="https://github.com/kduffuor/ResumeIQ.git" target="_blank" style="color: #666; text-decoration: none;">GitHub</a> • 
        <a href="https://linkedin.com/in/kduffuor" target="_blank" style="color: #666; text-decoration: none;">LinkedIn</a>
    </div>
    """)
    
    analyze_btn.click(
        fn=analyze,
        inputs=[file_input, resume_text, job_input],
        outputs=[score_output, keywords_output, suggestions_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)