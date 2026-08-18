import streamlit as st
import sqlite3
import json
import logging
import os
import re
import math
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import fitz  # PyMuPDF
import pdfplumber
import google.generativeai as genai
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION & SETTINGS
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NEW: Smart API Key Fetcher ---
def get_api_key(key_name: str) -> str:
    """
    Fetches the API key from Streamlit Cloud Secrets first.
    If it's not running on Cloud, it falls back to the local .env file.
    """
    try:
        # Check Streamlit Secrets first
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        # If st.secrets throws an error (e.g., running locally without a .streamlit folder)
        pass
    
    # Fallback to local environment / .env file
    return os.getenv(key_name, "")


class Settings(BaseSettings):
    gemini_keys: List[str] = Field(default_factory=lambda: [
        get_api_key("GEMINI_API_KEY_1"),
        get_api_key("GEMINI_API_KEY_2"),
        get_api_key("GEMINI_API_KEY_3"),
        get_api_key("GEMINI_API_KEY_4")
    ])
    database_path: str = Field(default="ca_prediction.db")
    
SETTINGS = Settings()

class ScoringWeights(BaseModel):
    w_frequency: float = 0.20
    w_recency: float = 0.15
    w_rotation: float = 0.15
    w_source_convergence: float = 0.20
    w_amendment_relevance: float = 0.15
    w_template_trend: float = 0.05
    w_chapter_weight: float = 0.05
    w_semantic_evidence: float = 0.05
    recency_decay_lambda: float = 0.25

DEFAULT_WEIGHTS = ScoringWeights()

class PaperStructure(BaseModel):
    name: str
    max_marks: int
    chapters: List[str]

DEFAULT_EXAM_SCHEME: Dict[str, PaperStructure] = {
    "Paper 1: Advanced Accounting": PaperStructure(
        name="Advanced Accounting", max_marks=100,
        chapters=[
            "Accounting Standards (AS 1, 3, 10, 11, 12, 13, 16, etc.)",
            "Financial Statements of Companies",
            "Buyback of Securities & Equity with Differential Rights",
            "Amalgamation and Reconstruction of Companies",
            "Accounting for Branches Including Foreign Branches",
            "Framework for Preparation and Presentation of Financial Statements"
        ]
    ),
    "Paper 2: Corporate and Other Laws": PaperStructure(
        name="Corporate and Other Laws", max_marks=100,
        chapters=[
            "Company Law: Incorporation & Incidental Matters",
            "Prospectus and Allotment of Securities",
            "Share Capital and Debentures",
            "Management & Administration",
            "Accounts of Companies & Audit",
            "The General Clauses Act, 1897"
        ]
    ),
    "Paper 3: Taxation": PaperStructure(
        name="Taxation", max_marks=100,
        chapters=[
            "Income Tax: Basic Concepts & Residential Status",
            "Heads of Income: Salaries & House Property",
            "Heads of Income: PGBP & Capital Gains",
            "Total Income & Tax Computation",
            "GST: Concept, Charge & Composition Scheme"
        ]
    )
}

EXAM_ATTEMPTS_CHRONOLOGY = [
    {"attempt": "May 2022", "order": 1, "cutoff_date": "2022-04-30"},
    {"attempt": "Nov 2022", "order": 2, "cutoff_date": "2022-10-31"},
    {"attempt": "May 2023", "order": 3, "cutoff_date": "2023-04-30"},
    {"attempt": "Nov 2023", "order": 4, "cutoff_date": "2023-10-31"},
    {"attempt": "May 2024", "order": 5, "cutoff_date": "2024-04-30"},
    {"attempt": "Sep 2024", "order": 6, "cutoff_date": "2024-08-31"},
    {"attempt": "Jan 2025", "order": 7, "cutoff_date": "2024-12-31"},
    {"attempt": "May 2025", "order": 8, "cutoff_date": "2025-04-30"},
    {"attempt": "Sep 2025", "order": 9, "cutoff_date": "2025-08-31"},
    {"attempt": "Jan 2026", "order": 10, "cutoff_date": "2025-12-31"},
    {"attempt": "May 2026", "order": 11, "cutoff_date": "2026-04-30"},
    {"attempt": "Sep 2026", "order": 12, "cutoff_date": "2026-08-31"},
    {"attempt": "Jan 2027", "order": 13, "cutoff_date": "2026-12-31"},
    {"attempt": "May 2027", "order": 14, "cutoff_date": "2027-04-30"}
]

# ==========================================
# 2. DATABASE MODELS & REPOSITORY
# ==========================================
class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    source_type: str
    paper: str
    exam_attempt: str
    publication_date: str
    extraction_quality: float
    ocr_used: bool = False
    warning_logs: Optional[str] = None
    upload_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    file_hash: str

class QuestionRecord(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    paper: str
    exam_attempt: str
    source_type: str
    section: str = "Descriptive"
    question_number: str
    marks: float
    question_text: str
    question_type: str 
    chapter: str
    topic: str
    concept: str
    question_template: str
    difficulty: str = "medium"
    amendment_sensitive: bool = False
    applicability_status: str = "Applicable"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class Repository:
    def __init__(self, db_path: str = SETTINGS.database_path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY, filename TEXT, source_type TEXT, paper TEXT,
                exam_attempt TEXT, publication_date TEXT, extraction_quality REAL,
                ocr_used INTEGER, warning_logs TEXT, upload_date TEXT, file_hash TEXT UNIQUE
            );""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY, document_id TEXT, paper TEXT, exam_attempt TEXT,
                source_type TEXT, section TEXT, question_number TEXT, marks REAL,
                question_text TEXT, question_type TEXT, chapter TEXT, topic TEXT,
                concept TEXT, question_template TEXT, difficulty TEXT,
                amendment_sensitive INTEGER, applicability_status TEXT, created_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (document_id)
            );""")
            conn.commit()

    def insert_document(self, doc: DocumentRecord) -> bool:
        try:
            with self._get_connection() as conn:
                conn.cursor().execute("""
                INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (doc.document_id, doc.filename, doc.source_type, doc.paper, doc.exam_attempt, 
                      doc.publication_date, doc.extraction_quality, 1 if doc.ocr_used else 0, 
                      doc.warning_logs, doc.upload_date, doc.file_hash))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def insert_questions(self, questions: List[QuestionRecord]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for q in questions:
                cursor.execute("""
                INSERT OR REPLACE INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (q.question_id, q.document_id, q.paper, q.exam_attempt, q.source_type, 
                      q.section, q.question_number, q.marks, q.question_text, q.question_type, 
                      q.chapter, q.topic, q.concept, q.question_template, q.difficulty,
                      1 if q.amendment_sensitive else 0, q.applicability_status, q.created_at))
            conn.commit()

    def get_questions_strictly_before_attempt(self, paper: str, target_attempt: str) -> List[QuestionRecord]:
        attempt_order_map = {item["attempt"]: item["order"] for item in EXAM_ATTEMPTS_CHRONOLOGY}
        target_order = attempt_order_map.get(target_attempt, 999)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT q.* FROM questions q JOIN documents d ON q.document_id = d.document_id WHERE q.paper = ? AND q.applicability_status != 'Not Applicable'", (paper,))
            rows = cursor.fetchall()
            valid_questions = []
            for r in rows:
                if attempt_order_map.get(r["exam_attempt"], 0) < target_order:
                    valid_questions.append(QuestionRecord(**dict(r)))
            return valid_questions

    def get_actual_paper_questions(self, paper: str, target_attempt: str) -> List[QuestionRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM questions WHERE paper = ? AND exam_attempt = ? AND source_type = 'PYQ'", (paper, target_attempt))
            return [QuestionRecord(**dict(r)) for r in cursor.fetchall()]

# ==========================================
# 3. DOCUMENT INGESTION & PARSING
# ==========================================
class DocumentParser:
    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @classmethod
    def parse_pdf(cls, file_bytes: bytes, filename: str) -> Tuple[str, float, bool, str]:
        extracted_text, warnings, ocr_needed = "", [], False
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            if total_pages == 0: return "", 0.0, True, "Empty PDF document"
            raw_text = "\n".join([doc[i].get_text("text") for i in range(total_pages)]).strip()
            doc.close()
            
            avg_chars = len(raw_text) / total_pages if total_pages > 0 else 0
            if avg_chars < 100:
                ocr_needed, quality_score = True, 30.0
                warnings.append("Low text density. Scanned PDF likely.")
            else:
                quality_score = min(100.0, 70.0 + (min(avg_chars, 2000) / 2000.0) * 30.0)

            if "table" in raw_text.lower() or quality_score < 80:
                try:
                    with pdfplumber.open(fitz.io.BytesIO(file_bytes)) as plumber_pdf:
                        table_text = []
                        for page in plumber_pdf.pages:
                            for tbl in page.extract_tables():
                                for row in tbl:
                                    clean_row = [str(c).strip() for c in row if c is not None]
                                    if clean_row: table_text.append(" | ".join(clean_row))
                        if table_text: raw_text += "\n\n--- TABLES ---\n" + "\n".join(table_text)
                except Exception as pe:
                    warnings.append(f"Table fallback failed: {str(pe)}")

            return raw_text, quality_score, ocr_needed, ("; ".join(warnings) if warnings else "Clean")
        except Exception as e:
            return "", 0.0, True, f"Error: {str(e)}"

class DeterministicQuestionExtractor:
    QUESTION_PATTERN = re.compile(r'(?:\n|\A)(?:Question\s*(?:No\.?)?\s*(\d+)\s*(?:\(([a-zA-Z])\))?|Q\.?\s*(\d+)\s*(?:\(([a-zA-Z])\))?|(\d+)\.\s*\(([a-zA-Z])\))', re.IGNORECASE)
    MARKS_PATTERN = re.compile(r'\((?:Marks\s*|Max\s*Marks\s*)?(\d+)\s*Marks?\)', re.IGNORECASE)

    @classmethod
    def extract(cls, text: str, doc_id: str, paper: str, attempt: str, src_type: str) -> List[QuestionRecord]:
        matches = list(cls.QUESTION_PATTERN.finditer(text))
        questions = []
        chapters = DEFAULT_EXAM_SCHEME.get(paper).chapters if paper in DEFAULT_EXAM_SCHEME else ["General"]

        if not matches:
            if len(text.strip()) > 50:
                questions.append(cls._build(doc_id, paper, attempt, src_type, "1", text.strip(), 10.0, chapters))
            return questions

        for idx, match in enumerate(matches):
            start_pos = match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            q_num_raw = match.group(1) or match.group(3) or match.group(5) or str(idx + 1)
            sub_part = match.group(2) or match.group(4) or match.group(6) or ""
            q_body = text[start_pos:end_pos].strip()
            
            if len(q_body) < 25: continue
            marks_match = cls.MARKS_PATTERN.search(q_body)
            marks = float(marks_match.group(1)) if marks_match else (5.0 if sub_part else 10.0)

            questions.append(cls._build(doc_id, paper, attempt, src_type, f"Q{q_num_raw}{f'({sub_part})' if sub_part else ''}", q_body, marks, chapters))
        return questions

    @classmethod
    def _build(cls, doc_id, paper, attempt, src_type, q_num, body, marks, chapters) -> QuestionRecord:
        assigned_chap = chapters[0]
        for chap in chapters:
            if any(k in body.lower() for k in [w.lower() for w in chap.split() if len(w)>4]):
                assigned_chap = chap
                break

        lb = body.lower()
        if any(w in lb for w in ["calculate", "compute", "journal"]): q_type, tmpl = "numerical", "Computation Ledger"
        elif any(w in lb for w in ["advise", "state whether", "provisions"]): q_type, tmpl = "case-study", "Statutory Case Application"
        elif any(w in lb for w in ["explain", "describe", "distinguish"]): q_type, tmpl = "theory", "Conceptual Theory"
        else: q_type, tmpl = "practical", "Standard Practical Scenario"

        return QuestionRecord(
            document_id=doc_id, paper=paper, exam_attempt=attempt, source_type=src_type,
            question_number=q_num, marks=marks, question_text=body, question_type=q_type,
            chapter=assigned_chap, topic=assigned_chap, concept=f"{assigned_chap.split(':')[0]} Principle",
            question_template=tmpl, amendment_sensitive="amendment" in lb
        )

# ==========================================
# 4. ANALYTICS & PREDICTION ENGINES
# ==========================================
class AnalyticsEngine:
    @staticmethod
    def get_frequencies(questions: List[QuestionRecord]) -> Dict[str, float]:
        if not questions: return {}
        attempts = set(q.exam_attempt for q in questions)
        total = max(1, len(attempts))
        appearances = {}
        for q in questions:
            appearances.setdefault(q.concept, set()).add(q.exam_attempt)
        return {c: min(100.0, (len(s)/total)*100.0) for c, s in appearances.items()}

    @staticmethod
    def get_recencies(questions: List[QuestionRecord], target_attempt: str, decay: float) -> Dict[str, float]:
        order_map = {item["attempt"]: item["order"] for item in EXAM_ATTEMPTS_CHRONOLOGY}
        target_order = order_map.get(target_attempt, 10)
        recency = {}
        for q in questions:
            age = max(1, target_order - order_map.get(q.exam_attempt, 0))
            weight = math.exp(-decay * age) * 100.0
            if q.concept not in recency or weight > recency[q.concept]:
                recency[q.concept] = weight
        return recency

    @staticmethod
    def get_convergences(questions: List[QuestionRecord]) -> Dict[str, Dict[str, Any]]:
        weights = {"PYQ": 0.30, "RTP": 0.25, "MTP": 0.20, "Study Material": 0.15, "Amendment": 0.10}
        sources = {}
        for q in questions:
            sources.setdefault(q.concept, set()).add(q.source_type)
        return {c: {"score": min(100.0, (sum(weights.get(s, 0.05) for s in src) / 0.90) * 100.0), "src": list(src)} 
                for c, src in sources.items()}

class PredictionScoringEngine:
    @staticmethod
    def calculate(questions: List[QuestionRecord], target: str, w: ScoringWeights) -> List[Dict[str, Any]]:
        if not questions: return []
        
        f_map = AnalyticsEngine.get_frequencies(questions)
        r_map = AnalyticsEngine.get_recencies(questions, target, w.recency_decay_lambda)
        c_map = AnalyticsEngine.get_convergences(questions)

        meta = {}
        for q in questions:
            if q.concept not in meta:
                meta[q.concept] = {"chapter": q.chapter, "template": q.question_template, "marks": [], "amend": q.amendment_sensitive}
            meta[q.concept]["marks"].append(q.marks)

        preds = []
        for concept, m in meta.items():
            f_sc = f_map.get(concept, 30.0)
            r_sc = r_map.get(concept, 40.0)
            conv_data = c_map.get(concept, {"score": 30.0, "src": ["Study Material"]})
            c_sc = conv_data["score"]
            a_sc = 100.0 if m["amend"] else 40.0

            final_score = (w.w_frequency*f_sc + w.w_recency*r_sc + w.w_source_convergence*c_sc + 
                           w.w_amendment_relevance*a_sc + w.w_rotation*50.0 + w.w_template_trend*75.0 + 
                           w.w_chapter_weight*70.0 + w.w_semantic_evidence*70.0)
            final_score = round(max(0.0, min(100.0, final_score)), 2)

            tier = "Extremely High Priority" if final_score >= 90 else "Very High Priority" if final_score >= 75 else "High Priority" if final_score >= 60 else "Medium Priority" if final_score >= 40 else "Low Priority"
            avg_marks = sum(m["marks"])/len(m["marks"]) if m["marks"] else 8.0

            preds.append({
                "concept": concept, "chapter": m["chapter"], "question_template": m["template"],
                "prediction_score": final_score, "confidence_tier": tier,
                "expected_marks_min": max(4.0, round(avg_marks - 2.0, 1)),
                "expected_marks_max": min(16.0, round(avg_marks + 2.0, 1)),
                "score_breakdown": {"Freq": round(f_sc,1), "Recency": round(r_sc,1), "Conv": round(c_sc,1)},
                "evidence_summary": f"Freq: {f_sc:.1f}/100; Recency: {r_sc:.1f}/100; Sources: {', '.join(conv_data['src'])}"
            })
        return sorted(preds, key=lambda x: x["prediction_score"], reverse=True)

# ==========================================
# 5. GENERATION & LLM PROVIDER
# ==========================================
class MultiKeyGeminiProvider:
    def __init__(self):
        self.keys = [k for k in SETTINGS.gemini_keys if k.strip()]
        self.idx = 0
        if self.keys: genai.configure(api_key=self.keys[self.idx])

    def generate(self, prompt: str) -> str:
        if not self.keys: return "Mock Generation (No API Keys Found):\n1. Practical Variant\n2. Case-Study Variant"
        attempts = 0
        while attempts < len(self.keys):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                return model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=600, temperature=0.3)).text
            except Exception as e:
                self.idx = (self.idx + 1) % len(self.keys)
                genai.configure(api_key=self.keys[self.idx])
                attempts += 1
        return "Failed to generate: All API keys rate-limited or invalid."

class QuestionGenerationEngine:
    @staticmethod
    def generate_candidate_variants(prediction: Dict[str, Any], llm: MultiKeyGeminiProvider) -> List[Dict[str, Any]]:
        prompt = f"""You are an expert ICAI CA Intermediate paper creator.
Design 3 brand-new original examination questions. DO NOT COPY COPYRIGHTED MATERIAL.
Concept: {prediction['concept']} | Chapter: {prediction['chapter']} | Target Marks: {prediction['expected_marks_max']}
Output Format:
Variant A (Practical ICAI Style): [Question Text]
Variant B (Case-Based): [Question Text]
Variant C (Twisted Adjustment): [Question Text]"""
        return [{"variant_type": "ICAI Synthetic Variants", "concept": prediction["concept"], "expected_marks": prediction["expected_marks_max"], "content": llm.generate(prompt)}]

# ==========================================
# 6. BACKTESTING ENGINE
# ==========================================
class EvaluationMetrics:
    @staticmethod
    def calculate_marks_coverage(predicted_concepts: Set[str], actual_questions: List[QuestionRecord]) -> Dict[str, Any]:
        if not actual_questions: return {"marks_coverage_pct": 0.0, "total_actual_marks": 0.0, "covered_marks": 0.0, "concept_match_pct": 0.0}
        
        total_marks = sum(q.marks for q in actual_questions)
        actual_concepts = set(q.concept for q in actual_questions)
        matched_concepts = set()
        covered_marks = 0.0

        for q in actual_questions:
            if q.concept in predicted_concepts:
                covered_marks += q.marks
                matched_concepts.add(q.concept)

        return {
            "marks_coverage_pct": round((covered_marks/total_marks)*100.0, 2) if total_marks else 0.0,
            "total_actual_marks": round(total_marks, 1),
            "covered_marks": round(covered_marks, 1),
            "concept_match_pct": round((len(matched_concepts)/len(actual_concepts))*100.0, 2) if actual_concepts else 0.0,
            "matched_concepts": list(matched_concepts),
            "missed_concepts": list(actual_concepts - matched_concepts)
        }

class BacktestingRunner:
    def __init__(self, repo: Repository):
        self.repo = repo

    def run(self, paper: str, target: str, top_k: int) -> Dict[str, Any]:
        past_qs = self.repo.get_questions_strictly_before_attempt(paper, target)
        actual_qs = self.repo.get_actual_paper_questions(paper, target)
        
        if not past_qs: return {"status": "error", "message": f"No training data before {target}."}
        if not actual_qs: return {"status": "error", "message": f"No actual PYQ data for {target} to evaluate against."}

        preds = PredictionScoringEngine.calculate(past_qs, target, DEFAULT_WEIGHTS)[:top_k]
        pred_concepts = set(p["concept"] for p in preds)
        metrics = EvaluationMetrics.calculate_marks_coverage(pred_concepts, actual_qs)

        return {"status": "success", "metrics": metrics, "top_predictions": preds}

# ==========================================
# 7. STREAMLIT USER INTERFACE
# ==========================================
def _seed_demo_data(repo: Repository):
    attempts = [item["attempt"] for item in EXAM_ATTEMPTS_CHRONOLOGY[:8]]
    chapters = DEFAULT_EXAM_SCHEME["Paper 1: Advanced Accounting"].chapters
    qs = []
    for att in attempts:
        doc = DocumentRecord(filename=f"Demo_{att}.pdf", source_type="PYQ", paper="Paper 1: Advanced Accounting", exam_attempt=att, publication_date="2023-01-01", extraction_quality=95.0, file_hash=f"hash_{att}")
        repo.insert_document(doc)
        for i, chap in enumerate(chapters):
            qs.append(QuestionRecord(
                document_id=doc.document_id, paper="Paper 1: Advanced Accounting", exam_attempt=att, source_type="PYQ",
                question_number=f"Q{i+1}", marks=10.0 if i%2==0 else 5.0, question_text=f"Sample text for {chap}",
                question_type="numerical", chapter=chap, topic=chap, concept=f"{chap.split(':')[0]} Principle",
                question_template="Adjustment", difficulty="medium", amendment_sensitive=(i==0)
            ))
    repo.insert_questions(qs)
    st.toast("Synthentic ICAI database loaded successfully!")

def render_app():
    st.set_page_config(page_title="CA Prediction AI", page_icon="⚖️", layout="wide")
    repo = Repository()
    llm = MultiKeyGeminiProvider()
    backtester = BacktestingRunner(repo)
    attempts_list = [item["attempt"] for item in EXAM_ATTEMPTS_CHRONOLOGY]

    st.sidebar.title("🎯 Control Panel")
    selected_paper = st.sidebar.selectbox("CA Intermediate Paper", list(DEFAULT_EXAM_SCHEME.keys()))
    target_attempt = st.sidebar.selectbox("Target Attempt", attempts_list, index=len(attempts_list)-1)
    
    st.sidebar.subheader("⚙️ Model Weights")
    DEFAULT_WEIGHTS.w_frequency = st.sidebar.slider("Frequency", 0.0, 0.5, 0.20, 0.05)
    DEFAULT_WEIGHTS.w_recency = st.sidebar.slider("Recency", 0.0, 0.5, 0.15, 0.05)
    DEFAULT_WEIGHTS.w_source_convergence = st.sidebar.slider("Source Convergence", 0.0, 0.5, 0.20, 0.05)

    st.title("🏛️ CA Intermediate Prediction AI")
    st.caption(f"Predicting **{selected_paper}** | Target: **{target_attempt}** (Strict Data Cutoff Applied)")

    eligible_questions = repo.get_questions_strictly_before_attempt(selected_paper, target_attempt)

    tabs = st.tabs(["📊 Dashboard", "📂 Ingestion", "📈 Predictions", "✍️ AI Generation", "⏪ Backtesting"])

    with tabs[0]:
        if not eligible_questions:
            st.warning("⚠️ No historical documents ingested before target date.")
            if st.button("🚀 Load Synthetic Demo Data"):
                _seed_demo_data(repo)
                st.rerun()
        else:
            preds = PredictionScoringEngine.calculate(eligible_questions, target_attempt, DEFAULT_WEIGHTS)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Training Questions", len(eligible_questions))
            c2.metric("Predicted Concepts", len(preds))
            c3.metric("Top Confidence", f"{preds[0]['prediction_score']:.1f}" if preds else "N/A")
            c4.metric("Cutoff Date", [x["cutoff_date"] for x in EXAM_ATTEMPTS_CHRONOLOGY if x["attempt"]==target_attempt][0])

            if preds:
                st.subheader("Top Predicted Topics")
                fig = px.bar(pd.DataFrame(preds[:10]), x="prediction_score", y="concept", orientation="h", color="prediction_score", color_continuous_scale="Blues")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Upload ICAI Material")
            doc_type = st.selectbox("Source Type", ["PYQ", "RTP", "MTP", "Study Material", "Amendment"])
            doc_attempt = st.selectbox("Document Exam Attempt", attempts_list, index=5)
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded_file and st.button("Ingest Document"):
                file_bytes = uploaded_file.read()
                file_hash = DocumentParser.compute_file_hash(file_bytes)
                text, quality, ocr, warnings = DocumentParser.parse_pdf(file_bytes, uploaded_file.name)
                doc = DocumentRecord(filename=uploaded_file.name, source_type=doc_type, paper=selected_paper, exam_attempt=doc_attempt, publication_date="2024-01-01", extraction_quality=quality, file_hash=file_hash)
                if repo.insert_document(doc):
                    qs = DeterministicQuestionExtractor.extract(text, doc.document_id, selected_paper, doc_attempt, doc_type)
                    repo.insert_questions(qs)
                    st.success(f"Success! Extracted {len(qs)} questions. Quality: {quality:.1f}%")
                else: st.error("Duplicate File Detected.")
        with c2:
            st.subheader("Database Overview")
            st.info(f"Total eligible questions: {len(eligible_questions)}")
            if eligible_questions: st.dataframe(pd.DataFrame([{"Chapter": q.chapter, "Marks": q.marks, "Type": q.question_type} for q in eligible_questions[:10]]))

    with tabs[2]:
        if eligible_questions:
            preds = PredictionScoringEngine.calculate(eligible_questions, target_attempt, DEFAULT_WEIGHTS)
            for p in preds[:10]:
                with st.expander(f"⭐ {p['concept']} — Score: {p['prediction_score']}/100 [{p['confidence_tier']}]"):
                    st.markdown(f"**Template:** `{p['question_template']}` | **Marks:** {p['expected_marks_min']} - {p['expected_marks_max']}")
                    st.markdown(f"**Evidence:** {p['evidence_summary']}")

    with tabs[3]:
        if eligible_questions:
            preds = PredictionScoringEngine.calculate(eligible_questions, target_attempt, DEFAULT_WEIGHTS)
            target_p = st.selectbox("Select Concept to Synthesize", [p["concept"] for p in preds[:8]])
            if st.button("Generate Question via Gemini AI"):
                with st.spinner("Synthesizing..."):
                    selected_pred = next(p for p in preds if p["concept"] == target_p)
                    res = QuestionGenerationEngine.generate_candidate_variants(selected_pred, llm)
                    st.info(res[0]["content"])

    with tabs[4]:
        st.subheader("Zero-Leakage Historical Backtesting")
        bt_target = st.selectbox("Select Historical Target to Test Against", attempts_list[2:10], index=3)
        top_k = st.slider("Top K Predictions", 5, 20, 10)
        if st.button("Run Backtest"):
            res = backtester.run(selected_paper, bt_target, top_k)
            if res["status"] == "error": st.error(res["message"])
            else:
                m = res["metrics"]
                st.success(f"Backtest complete for {bt_target}")
                b1, b2, b3 = st.columns(3)
                b1.metric("Marks Coverage", f"{m['marks_coverage_pct']}%")
                b2.metric("Concept Match", f"{m['concept_match_pct']}%")
                b3.metric("Actual Marks Tested", m["total_actual_marks"])
                c1, c2 = st.columns(2)
                c1.write("**✅ Covered Concepts**"); c1.write(m["matched_concepts"])
                c2.write("**❌ Missed Concepts**"); c2.write(m["missed_concepts"])

if __name__ == "__main__":
    render_app()
