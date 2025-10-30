import csv
import os
import zipfile
import re
from pathlib import Path
import fitz  # PyMuPDF
import ollama
import shutil

# ===== CONFIGURATION =====
SUBMISSIONS_DIR = "submissions"  # Folder containing student ZIP files
ASSIGNMENT_PDF = "Project-1 Phase-2_as.pdf"  # PDF rubric file
OUTPUT_CSV = "grades.csv"
DEBUG_MODE = True  # Set to False to disable raw response logging
# =========================

def extract_requirements(pdf_path):
    """Extract text from PDF requirements file"""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            page_text = page.get_text()
            if len("\n".join(full_text)) + len(page_text) < 2500:
                full_text.append(page_text)
        return "\n".join(full_text)
    except FileNotFoundError:
        print(f"Error: Requirement PDF not found at {pdf_path}")
        exit(1)

def parse_llm_response(text):
    """Robust parsing using regex with fallbacks"""
    try:
        # Use regex to handle variations in response format
        score_match = re.search(r"Score\|([0-9.]+)\|?", text)
        feedback_match = re.search(r"Feedback\|(.*?)(\||$)", text)
        
        score = float(score_match.group(1)) if score_match else 5.0
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback parsed"
        
        # Clamp score between 0-10
        score = max(0.0, min(10.0, score))
        return score, feedback
    except Exception as e:
        if DEBUG_MODE:
            print(f"Parse error: {str(e)}")
        return 5.0, "Could not parse analysis results"

def analyze_submission(file_path, content, requirements):
    """Analyze file content against requirements using CodeLlama"""
    prompt = f"""EVALUATION TASK:
1. Review this code/config file against these requirements:
{requirements}

2. File: {file_path}
3. Content (truncated if long):
{content[:2500]}

FORMAT REQUIREMENTS:
- Respond ONLY with these 2 lines:
Score|<0-10 with decimal>|
Feedback|<concise issues>|
- No other text or explanations"""

    try:
        response = ollama.generate(
            model="codellama:7b",
            prompt=prompt,
            format="json",
            options={
                'temperature': 0.2,  # Reduce randomness
                'timeout': 45
            }
        )
        
        if DEBUG_MODE:
            print(f"\nRAW LLM RESPONSE ({file_path}):")
            print(response['response'])
            print("----END RAW RESPONSE----\n")
            
        return parse_llm_response(response['response'])
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return 0.0, "Analysis failed"

def process_submissions():
    requirements = extract_requirements(Path(ASSIGNMENT_PDF).resolve())
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ASURITE', 'File', 'Score', 'Feedback', 'Total'])
        
        for zip_path in Path(SUBMISSIONS_DIR).glob("*.zip"):
            asurite_id = zip_path.stem.split("-")[0]
            total_score = 0
            print(f"\nProcessing submission: {asurite_id}")
            
            with zipfile.ZipFile(zip_path) as zf:
                extract_dir = Path(f"temp/{asurite_id}")
                zf.extractall(extract_dir)
                
                for file in extract_dir.rglob('*'):
                    if file.is_file() and file.suffix in ('.py', '.yaml', '.yml'):
                        try:
                            content = file.read_text(encoding='utf-8', errors='ignore')
                            rel_path = file.relative_to(extract_dir)
                            print(f"  Analyzing: {rel_path}")
                            
                            score, feedback = analyze_submission(
                                str(rel_path), 
                                content, 
                                requirements
                            )
                            total_score += score
                            
                            print(f"    Score: {score:.1f}/10")
                            print(f"    Feedback: {feedback}")
                            
                            writer.writerow([
                                asurite_id,
                                str(rel_path),
                                f"{score:.1f}/10",
                                feedback,
                                f"{total_score:.1f}/40"
                            ])
                            csvfile.flush()
                            
                        except Exception as e:
                            print(f"Error processing {file}: {str(e)}")
                
                shutil.rmtree(extract_dir, ignore_errors=True)

if __name__ == "__main__":
    try:
        print("Checking Ollama connection...")
        models = ollama.list()['models']
        
        if not any(m['model'] == 'codellama:7b' for m in models):
            print("Model 'codellama:7b' not found. Install with:")
            print("docker exec ollama ollama pull codellama:7b")
            exit(1)
            
        process_submissions()
        print(f"\nGrading complete. Results saved to {OUTPUT_CSV}")
        
    except ConnectionError:
        print("""
ERROR: Could not connect to Ollama. For Windows users:
1. Ensure Docker Desktop is running
2. Start Ollama in Docker:
   docker run -d -p 11434:11434 --name ollama ollama/ollama
3. Pull the required model:
   docker exec ollama ollama pull codellama:7b
4. Verify it's working:
   docker exec ollama ollama list
5. Run this script again.""")
        exit(1)
