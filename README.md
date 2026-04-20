# ResumeIQ
AI-powered resume analyzer with ATS match scoring, keyword analysis, and targeted improvement suggestions.

## Features
- ATS match score with colour-coded indicator  
- Matched and missing keyword analysis  
- Three targeted improvement recommendations  
- Natural language interface — no technical knowledge required  

## Tech Stack
| Layer            | Technology                     |
|------------------|--------------------------------|
| Frontend         | Gradio                         |
| LLM Integration  | HuggingFace Inference API      |
| Language Model   | Meta Llama 3.1 8B Instruct     |
| Language         | Python 3.10                    |

## Local Setup

### 1. Clone the repository
    git clone https://github.com/kduffuor/ResumeIQ.git
    cd ResumeIQ

### 2. Create and activate a virtual environment
    python -m venv venv
    venv\Scripts\activate        # Windows
    source venv/bin/activate     # Mac/Linux

### 3. Install dependencies
    pip install -r requirements.txt

### 4. Create a `.env` file
    HF_API_TOKEN=your_huggingface_token_here

### 5. Run the app
    python app.py

## Use Cases
- Job seekers looking to improve ATS compatibility  
- Students preparing resumes for internships  
- Professionals tailoring resumes for specific roles  
- Career switchers optimizing keyword relevance 

## Why ResumeIQ
Most resumes fail ATS screening due to missing keywords and poor formatting.  
ResumeIQ helps bridge that gap by providing instant, actionable feedback powered by AI.