import io
import os
import re
import json
import hashlib
import subprocess
import tempfile
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import pdfplumber
import requests
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="CA Intermediate Prediction AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAPERS = [
    "Paper 1: Advanced Accounting",
    "Paper 2: Corporate and Other Laws",
    "Paper 3: Taxation",
    "Paper 4: Cost and Management Accounting",
    "Paper 5: Auditing and Ethics",
    "Paper 6: Financial Management and Strategic Management",
]

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PAPER_ALIASES = {
    "advanced accounting": "Paper 1: Advanced Accounting",
    "advanced account": "Paper 1: Advanced Accounting",
    "accounting": "Paper 1: Advanced Accounting",
    "corporate and other laws": "Paper 2: Corporate and Other Laws",
    "corporate & other laws": "Paper 2: Corporate and Other Laws",
    "law": "Paper 2: Corporate and Other Laws",
    "taxation": "Paper 3: Taxation",
    "cost and management accounting": "Paper 4: Cost and Management Accounting",
    "cost & management accounting": "Paper 4: Cost and Management Accounting",
    "cost and management": "Paper 4: Cost and Management Accounting",
    "auditing and ethics": "Paper 5: Auditing and Ethics",
    "audit": "Paper 5: Auditing and Ethics",
    "financial management and strategic management":
        "Paper 6: Financial Management and Strategic Management",
    "financial management & strategic management":
        "Paper 6: Financial Management and Strategic Management",
    "fm & sm": "Paper 6: Financial Management and Strategic Management",
}

TARGET_DEFAULT = "Next Attempt"

# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "generated_report" not in st.session_state:
    st.session_state.generated_report = None


# ============================================================
# HELPERS
# ============================================================

def get_api_key() -> str:
    """Read Gemini keys from Streamlit Secrets or environment and return one."""
    available_keys = []
    
    try:
        # Check for the user's multiple keys
        for i in range(1, 5):
            key = st.secrets.get(f"GEMINI_API_KEY_{i}", "")
            if key:
                available_keys.append(key)
                
        # Fallback to standard key just in case
        standard_key = st.secrets.get("GEMINI_API_KEY", "")
        if standard_key:
            available_keys.append(standard_key)
    except Exception:
        pass

    # Also check OS environment variables 
    for i in range(1, 5):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if k:
            available_keys.append(k)
    k_orig = os.getenv("GEMINI_API_KEY", "")
    if k_orig:
        available_keys.append(k_orig)

    if available_keys:
        # Pick a random key from the pool to distribute load and avoid rate limits
        return random.choice(available_keys)
    return ""


def clean_json_text(text: str) -> str:
    """Remove Markdown code fences and isolate a JSON object/array."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # Prefer an object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]

    # Fall back to an array.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start:end + 1]

    return text


def safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(clean_json_text(text))
    except Exception:
        return default


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def latex_escape(text: str) -> str:
    """Escape common LaTeX special characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def detect_paper_from_text(text: str) -> str:
    t = (text or "").lower()

    # More specific phrases first.
    for alias, paper in sorted(PAPER_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in t:
            return paper

    # Numbered paper fallback.
    for n, paper in enumerate(PAPERS, start=1):
        if re.search(rf"\bpaper\s*[-:]?\s*{n}\b", t):
            return paper

    return ""


def detect_attempt_from_text(text: str) -> str:
    t = normalize_space(text).lower()

    patterns = [
        r"\b(may|january|jan|september|sep|november|nov)\s+(20\d{2})\b",
        r"\b(20\d{2})\s+(may|january|jan|september|sep|november|nov)\b",
    ]

    month_map = {
        "jan": "January",
        "january": "January",
        "may": "May",
        "sep": "September",
        "september": "September",
        "nov": "November",
        "november": "November",
    }

    for p in patterns:
        m = re.search(p, t, re.I)
        if m:
            if m.group(1).isdigit():
                year, month = m.group(1), m.group(2)
            else:
                month, year = m.group(1), m.group(2)
            return f"{month_map[month.lower()]} {year}"

    return ""


def extract_pdf(file_bytes: bytes) -> Tuple[str, float, str]:
    """
    Robust extraction:
    1. PyMuPDF for normal text.
    2. pdfplumber tables as a supplement.
    IMPORTANT: use io.BytesIO, not fitz.io.BytesIO.
    """
    warnings = []
    text_parts: List[str] = []

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)

        if page_count == 0:
            doc.close()
            return "", 0.0, "PDF has no pages."

        for page in doc:
            txt = page.get_text("text") or ""
            if txt.strip():
                text_parts.append(txt)

        doc.close()
    except Exception as exc:
        return "", 0.0, f"PyMuPDF extraction failed: {exc}"

    text = "\n".join(text_parts).strip()
    chars_per_page = len(text) / max(page_count, 1)

    if chars_per_page < 80:
        quality = 25.0
        warnings.append("Very little selectable text; the PDF may be scanned.")
    elif chars_per_page < 250:
        quality = 60.0
        warnings.append("Low text density.")
    else:
        quality = min(100.0, 75.0 + min(chars_per_page, 1500) / 1500 * 25)

    # Supplement with tables. This is where the previous code had the crash:
    # fitz.io.BytesIO(...) is invalid; io.BytesIO(...) is correct.
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            table_lines = []
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        cells = [normalize_space(str(c)) for c in (row or []) if c is not None]
                        cells = [c for c in cells if c]
                        if cells:
                            table_lines.append(" | ".join(cells))

            if table_lines:
                text += "\n\n[EXTRACTED TABLE DATA]\n" + "\n".join(table_lines)
    except Exception as exc:
        warnings.append(f"Table extraction warning: {exc}")

    if not text:
        warnings.append("No selectable text extracted.")

    return text, round(quality, 1), " | ".join(warnings) if warnings else "OK"


def download_pdf(url: str) -> Tuple[str, bytes]:
    """Download a PDF from a direct URL."""
    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 CA-Prediction-AI"},
    )
    r.raise_for_status()

    content_type = (r.headers.get("content-type") or "").lower()
    if "pdf" not in content_type and not url.lower().split("?")[0].endswith(".pdf"):
        # Some servers do not send application/pdf, so validate the magic bytes too.
        if not r.content.startswith(b"%PDF"):
            raise ValueError("The URL did not return a PDF file.")

    return url.split("/")[-1] or "downloaded_paper.pdf", r.content


# ============================================================
# GEMINI
# ============================================================

class Gemini:
    def __init__(self) -> None:
        self.key = get_api_key()
        self.client = genai.Client(api_key=self.key) if (self.key and genai) else None

    @property
    def available(self) -> bool:
        return self.client is not None

    def generate(self, prompt: str, temperature: float = 0.15) -> str:
        if not self.client:
            return ""

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        return response.text or ""


# ============================================================
# AI DOCUMENT CLASSIFIER
# ============================================================

def classify_document(
    gemini: Gemini,
    filename: str,
    text: str,
) -> Dict[str, Any]:
    deterministic_paper = detect_paper_from_text(text)
    deterministic_attempt = detect_attempt_from_text(text)

    if not gemini.available:
        return {
            "paper": deterministic_paper or "Unknown",
            "exam_attempt": deterministic_attempt or "Unknown",
            "document_type": "PYQ",
            "confidence": 0.55 if deterministic_paper else 0.2,
        }

    sample = text[:18000]

    prompt = f"""
You are identifying an ICAI CA Intermediate PDF.

The filename may be random and MUST NOT be trusted.
Identify the document from its actual contents.

Possible papers:
{json.dumps(PAPERS, ensure_ascii=False)}

Possible document types:
PYQ, RTP, MTP, Study Material, Amendment, Suggested Answer, Other

Return ONLY valid JSON:

{{
  "paper": "one exact value from the possible papers, or Unknown",
  "exam_attempt": "for example May 2026, September 2026, November 2026, or Unknown",
  "document_type": "one value from the allowed types",
  "confidence": 0.0,
  "reason": "one short evidence-based sentence"
}}

Filename:
{filename}

PDF text:
{sample}
"""

    result = safe_json_loads(gemini.generate(prompt), {})
    return {
        "paper": result.get("paper") or deterministic_paper or "Unknown",
        "exam_attempt": result.get("exam_attempt") or deterministic_attempt or "Unknown",
        "document_type": result.get("document_type") or "Other",
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "reason": result.get("reason", ""),
    }


# ============================================================
# QUESTION EXTRACTION / TOPIC MAPPING
# ============================================================

def analyze_document_questions(
    gemini: Gemini,
    paper: str,
    attempt: str,
    document_type: str,
    text: str,
) -> List[Dict[str, Any]]:
    if not gemini.available:
        # Small deterministic fallback. It still preserves the source.
        chunks = re.split(
            r"\n(?=(?:question\s*)?\d+\s*[\.\):])",
            text,
            flags=re.I,
        )
        chunks = [normalize_space(x) for x in chunks if len(normalize_space(x)) > 40]
        return [
            {
                "question": c[:1200],
                "chapter": "Needs AI classification",
                "topic": "Needs AI classification",
                "marks": 0,
                "type": "Unknown",
                "source": document_type,
                "asked_in": attempt,
            }
            for c in chunks[:40]
        ]

    sample = text[:50000]

    prompt = f"""
You are an expert CA Intermediate exam analyst.

Analyze this {document_type} for:
Paper: {paper}
Exam attempt: {attempt}

Extract the meaningful exam questions or question groups.
For every item identify:
- exact/condensed question
- chapter
- topic
- marks if visible, otherwise 0
- question type
- why the concept matters
- asked_in

Do not invent chapter names when there is insufficient evidence.
Use the actual question content.

Return ONLY JSON in this format:

{{
  "questions": [
    {{
      "question": "...",
      "chapter": "...",
      "topic": "...",
      "marks": 8,
      "type": "Numerical|Theory|Case Study|MCQ|Other",
      "importance_hint": "...",
      "asked_in": "{attempt}"
    }}
  ]
}}

PDF text:
{sample}
"""

    data = safe_json_loads(gemini.generate(prompt), {})
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return []

    cleaned = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        question = normalize_space(str(q.get("question", "")))
        if len(question) < 15:
            continue
        cleaned.append({
            "question": question,
            "chapter": str(q.get("chapter", "Unknown")),
            "topic": str(q.get("topic", "Unknown")),
            "marks": float(q.get("marks", 0) or 0),
            "type": str(q.get("type", "Other")),
            "importance_hint": str(q.get("importance_hint", "")),
            "asked_in": str(q.get("asked_in", attempt)),
            "source": document_type,
            "paper": paper,
        })

    return cleaned


# ============================================================
# PREDICTION
# ============================================================

def build_prediction(
    gemini: Gemini,
    selected_paper: str,
    target_attempt: str,
    all_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Compact history sent to the model.
    history = []
    for q in all_questions:
        history.append({
            "chapter": q.get("chapter"),
            "topic": q.get("topic"),
            "marks": q.get("marks"),
            "type": q.get("type"),
            "asked_in": q.get("asked_in"),
            "source": q.get("source"),
            "question": q.get("question", "")[:700],
        })

    # Simple transparent frequency support, NOT a complex exposed scoring engine.
    topic_stats: Dict[str, Dict[str, Any]] = {}
    for q in all_questions:
        key = f"{q.get('chapter','Unknown')} | {q.get('topic','Unknown')}"
        item = topic_stats.setdefault(
            key,
            {"chapter": q.get("chapter", "Unknown"), "topic": q.get("topic", "Unknown"),
             "count": 0, "years": set(), "sources": set(), "marks": []},
        )
        item["count"] += 1
        item["years"].add(q.get("asked_in", "Unknown"))
        item["sources"].add(q.get("source", "Other"))
        if q.get("marks", 0):
            item["marks"].append(float(q["marks"]))

    stats_for_ai = []
    for item in topic_stats.values():
        stats_for_ai.append({
            "chapter": item["chapter"],
            "topic": item["topic"],
            "appearances": item["count"],
            "asked_in": sorted(item["years"]),
            "sources": sorted(item["sources"]),
            "total_marks_examples": round(sum(item["marks"]), 1),
        })

    if not gemini.available:
        # Fallback ranking by raw recurrence.
        ranked = sorted(
            stats_for_ai,
            key=lambda x: (x["appearances"], len(x["sources"])),
            reverse=True,
        )
        important = [
            {
                "rank": i + 1,
                "chapter": x["chapter"],
                "topic": x["topic"],
                "importance": min(100, 45 + x["appearances"] * 10),
                "asked_in": x["asked_in"],
                "reason": "Repeated in the supplied source material.",
            }
            for i, x in enumerate(ranked[:15])
        ]
        return {"important_topics": important, "predicted_questions": [], "notice": "AI key not configured."}

    prompt = f"""
You are a very careful CA Intermediate exam analyst.

Target paper: {selected_paper}
Target attempt: {target_attempt}

You are given historical ICAI material supplied by the user.
Do NOT claim certainty.
Do NOT copy questions verbatim.
Do NOT invent an occurrence year.

Your job:

1. Rank the most important topics/concepts likely to be useful for the next paper.
2. Prefer topics supported by repeated appearance across the supplied history.
3. Give "asked_in" as actual supplied attempt labels.
4. Generate a strong predicted question set covering the most important concepts.
5. For every predicted question state:
   - chapter
   - topic
   - expected marks
   - related previous attempts
   - why it is important
6. Questions must be original, not copied from ICAI.
7. Keep the output exam-focused.

Return ONLY valid JSON:

{{
  "important_topics": [
    {{
      "rank": 1,
      "chapter": "...",
      "topic": "...",
      "importance": 95,
      "asked_in": ["May 2024", "Nov 2025"],
      "reason": "..."
    }}
  ],
  "predicted_questions": [
    {{
      "number": 1,
      "question": "...",
      "chapter": "...",
      "topic": "...",
      "marks": 8,
      "type": "Numerical|Theory|Case Study|MCQ|Other",
      "asked_in": ["May 2024", "Sep 2025"],
      "why_important": "..."
    }}
  ],
  "important_note": "..."
}}

Historical topic statistics:
{json.dumps(stats_for_ai, ensure_ascii=False)}

Historical question evidence:
{json.dumps(history[:450], ensure_ascii=False)}
"""

    data = safe_json_loads(gemini.generate(prompt, temperature=0.1), {})
    data.setdefault("important_topics", [])
    data.setdefault("predicted_questions", [])
    return data


# ============================================================
# LATEX / PDF REPORT
# ============================================================

def make_latex(
    selected_paper: str,
    target_attempt: str,
    prediction: Dict[str, Any],
) -> str:
    topics = prediction.get("important_topics", [])
    questions = prediction.get("predicted_questions", [])
    note = prediction.get("important_note", "")

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=1.8cm]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{enumitem}",
        r"\usepackage{titlesec}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\titleformat{\section}{\Large\bfseries}{}{0em}{}",
        r"\titleformat{\subsection}{\large\bfseries}{}{0em}{}",
        r"\begin{document}",
        r"\begin{center}",
        r"\LARGE\textbf{CA Intermediate Prediction Report}\\[6pt]",
        rf"\large\textbf{{{latex_escape(selected_paper)}}}\\",
        rf"Target Attempt: \textbf{{{latex_escape(target_attempt)}}}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"\hrule",
        r"\vspace{10pt}",
        r"\section*{Important Topics}",
    ]

    for t in topics:
        title = f"{t.get('rank', '')}. {latex_escape(str(t.get('topic', '')))}"
        lines.append(
            r"\textbf{" + title + r"} --- Chapter: "
            + latex_escape(str(t.get('chapter', '')))
            + r"; Importance: "
            + latex_escape(str(t.get('importance', '')))
        )
        asked = ", ".join(t.get("asked_in", []) or [])
        lines.append(rf"\newline Asked in: \textit{{{latex_escape(asked)}}}")
        lines.append(rf"\newline Why: {latex_escape(str(t.get('reason','')))}\par\vspace{{5pt}}")

    lines.append(r"\section*{Predicted Questions}")

    for q in questions:
        lines.append(
            rf"\subsection*{{Q{latex_escape(str(q.get('number','')))} "
            rf"--- {latex_escape(str(q.get('marks','')))} marks}}"
        )
        lines.append(rf"\textbf{{Chapter:}} {latex_escape(str(q.get('chapter','')))}\par")
        lines.append(rf"\textbf{{Topic:}} {latex_escape(str(q.get('topic','')))}\par")
        asked = ", ".join(q.get("asked_in", []) or [])
        lines.append(rf"\textbf{{Related previous attempts:}} {latex_escape(asked)}\par")
        lines.append(rf"\textbf{{Type:}} {latex_escape(str(q.get('type','')))}\par\vspace{{3pt}}")
        lines.append(latex_escape(str(q.get("question", ""))) + r"\par")
        lines.append(rf"\textit{{Why important: {latex_escape(str(q.get('why_important','')))}}}\par\vspace{{8pt}}")

    lines.append(r"\section*{Important Note}")
    lines.append(latex_escape(str(note)))
    lines.append(r"\vfill")
    lines.append(r"\hrule")
    lines.append(
        r"\small This report is a model-generated study aid. "
        r"It does not guarantee exact ICAI questions."
    )
    lines.append(r"\end{document}")

    return "\n".join(lines)


def compile_latex(latex_source: str, filename_stem: str = "ca_prediction_report") -> Tuple[bytes | None, str]:
    """
    Compile with pdflatex if present. Streamlit Cloud may not have TeX installed.
    """
    if not shutil_which("pdflatex"):
        return None, "pdflatex is not installed."

    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, f"{filename_stem}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_source)

        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", f"{filename_stem}.tex"],
                cwd=tmp,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=True,
            )
            pdf_path = os.path.join(tmp, f"{filename_stem}.pdf")
            with open(pdf_path, "rb") as f:
                return f.read(), "LaTeX PDF created."
        except Exception as exc:
            return None, f"LaTeX compilation failed: {exc}"


def shutil_which(command: str) -> str | None:
    import shutil
    return shutil.which(command)


def make_reportlab_pdf(
    selected_paper: str,
    target_attempt: str,
    prediction: Dict[str, Any],
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed.")

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=f"CA Intermediate Prediction - {selected_paper}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCA",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=8,
    )
    small = ParagraphStyle(
        "SmallCA",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        spaceAfter=3,
    )
    qstyle = ParagraphStyle(
        "QuestionCA",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=5,
    )

    story = [
        Paragraph("CA Intermediate Prediction Report", title_style),
        Paragraph(
            f"<b>{selected_paper}</b><br/>Target Attempt: <b>{target_attempt}</b>",
            styles["Heading2"],
        ),
        Spacer(1, 8),
        Paragraph("Important Topics", styles["Heading1"]),
    ]

    topics = prediction.get("important_topics", [])
    data = [["Rank", "Chapter", "Topic", "Importance", "Asked In"]]
    for t in topics:
        data.append([
            str(t.get("rank", "")),
            str(t.get("chapter", "")),
            str(t.get("topic", "")),
            str(t.get("importance", "")),
            ", ".join(t.get("asked_in", []) or []),
        ])

    if len(data) > 1:
        table = Table(data, repeatRows=1, colWidths=[1.0*cm, 4.0*cm, 5.0*cm, 2.0*cm, 4.0*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ]))
        story += [table, Spacer(1, 10)]

    story.append(Paragraph("Predicted Questions", styles["Heading1"]))

    for q in prediction.get("predicted_questions", []):
        story += [
            Paragraph(
                f"<b>Q{q.get('number','')} — {q.get('marks', '')} marks</b>",
                styles["Heading2"],
            ),
            Paragraph(
                f"<b>Chapter:</b> {q.get('chapter','')}<br/>"
                f"<b>Topic:</b> {q.get('topic','')}<br/>"
                f"<b>Related previous attempts:</b> {', '.join(q.get('asked_in', []) or [])}<br/>"
                f"<b>Type:</b> {q.get('type','')}",
                small,
            ),
            Paragraph(str(q.get("question", "")), qstyle),
            Paragraph(
                f"<i>Why important:</i> {q.get('why_important','')}",
                small,
            ),
            Spacer(1, 7),
        ]

    story += [
        Spacer(1, 8),
        Paragraph("Important Note", styles["Heading1"]),
        Paragraph(
            str(prediction.get("important_note", "")),
            small,
        ),
    ]

    doc.build(story)
    return buffer.getvalue()


# ============================================================
# UI
# ============================================================

def process_source(
    gemini: Gemini,
    filename: str,
    data: bytes,
) -> Dict[str, Any]:
    file_hash = sha256_bytes(data)

    text, quality, warning = extract_pdf(data)

    meta = classify_document(
        gemini=gemini,
        filename=filename,
        text=text,
    )

    paper = meta.get("paper", "Unknown")
    attempt = meta.get("exam_attempt", "Unknown")
    doc_type = meta.get("document_type", "Other")

    questions = []
    if paper != "Unknown":
        questions = analyze_document_questions(
            gemini=gemini,
            paper=paper,
            attempt=attempt,
            document_type=doc_type,
            text=text,
        )

    return {
        "id": file_hash,
        "filename": filename,
        "paper": paper,
        "attempt": attempt,
        "document_type": doc_type,
        "confidence": meta.get("confidence", 0),
        "reason": meta.get("reason", ""),
        "quality": quality,
        "warning": warning,
        "text": text,
        "questions": questions,
    }


def render():
    gemini = Gemini()

    st.title("📚 CA Intermediate Prediction AI")
    st.caption(
        "Simple workflow: add ICAI PDFs → AI identifies them → finds important topics → "
        "generates original predicted questions → exports a clean PDF."
    )

    with st.sidebar:
        st.header("Settings")
        target_attempt = st.text_input(
            "Target exam",
            value=TARGET_DEFAULT,
            help="Example: November 2026 or May 2027.",
        )
        selected_paper = st.selectbox("Focus paper", ["All Papers"] + PAPERS)

        st.divider()
        st.write(
            "The filename does not need to be correct. "
            "The app identifies the paper and exam attempt from PDF contents."
        )

    if not gemini.available:
        st.warning(
            "Gemini API key is not configured. Add GEMINI_API_KEY_1 (or up to 4 keys) to Streamlit Secrets "
            "for AI classification and prediction."
        )

    tabs = st.tabs(["📥 Add Papers", "🔎 Analyze", "📄 PDF Report"])

    # --------------------------------------------------------
    # ADD PAPERS
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("Add ICAI PDFs")

        upload_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.markdown("### Or add direct PDF URLs")
        urls_text = st.text_area(
            "One direct PDF URL per line",
            placeholder="https://example.com/paper.pdf",
            height=100,
        )

        if st.button("➕ Add & Identify Papers", type="primary"):
            sources: List[Tuple[str, bytes]] = []

            for f in upload_files or []:
                sources.append((f.name, f.getvalue()))

            for raw_url in (urls_text or "").splitlines():
                url = raw_url.strip()
                if not url:
                    continue
                try:
                    name, data = download_pdf(url)
                    sources.append((name, data))
                except Exception as exc:
                    st.error(f"Could not download {url}: {exc}")

            if not sources:
                st.warning("Please upload a PDF or provide a direct PDF URL.")
            else:
                progress = st.progress(0)
                for i, (filename, data) in enumerate(sources, start=1):
                    try:
                        existing = {d["id"] for d in st.session_state.documents}
                        record = process_source(gemini, filename, data)

                        if record["id"] in existing:
                            st.info(f"Skipped duplicate: {filename}")
                        else:
                            st.session_state.documents.append(record)
                            st.success(
                                f"{filename} → {record['paper']} | "
                                f"{record['attempt']} | {record['document_type']} | "
                                f"{len(record['questions'])} questions"
                            )
                    except Exception as exc:
                        st.error(f"Failed to process {filename}: {exc}")
                    progress.progress(i / len(sources))

        if st.session_state.documents:
            st.markdown("### Detected documents")

            rows = []
            for d in st.session_state.documents:
                rows.append({
                    "File": d["filename"],
                    "Paper": d["paper"],
                    "Attempt": d["attempt"],
                    "Type": d["document_type"],
                    "Questions": len(d["questions"]),
                    "Quality": f"{d['quality']:.0f}%",
                    "AI confidence": f"{float(d['confidence']):.0%}",
                })

            st.dataframe(rows, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear all uploaded data"):
            st.session_state.documents = []
            st.session_state.analysis = None
            st.session_state.generated_report = None
            st.rerun()

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("Find Important Topics & Predict Questions")

        all_questions = []
        relevant_docs = []

        for d in st.session_state.documents:
            if selected_paper != "All Papers" and d["paper"] != selected_paper:
                continue
            relevant_docs.append(d)
            all_questions.extend(d["questions"])

        st.info(
            f"Using {len(relevant_docs)} document(s) and "
            f"{len(all_questions)} extracted question(s)."
        )

        if st.button("🚀 Analyze & Generate Predictions", type="primary"):
            if not all_questions:
                st.error("Add historical ICAI PDFs first.")
            else:
                with st.spinner("Analyzing repeated concepts and generating original predicted questions..."):
                    st.session_state.analysis = build_prediction(
                        gemini,
                        selected_paper if selected_paper != "All Papers" else "Selected CA Intermediate Papers",
                        target_attempt,
                        all_questions,
                    )

                st.success("Analysis completed.")

        result = st.session_state.analysis

        if result:
            st.markdown("## 🔥 High-Importance Topics")

            topic_rows = []
            for t in result.get("important_topics", []):
                topic_rows.append({
                    "#": t.get("rank", ""),
                    "Chapter": t.get("chapter", ""),
                    "Topic": t.get("topic", ""),
                    "Importance": t.get("importance", ""),
                    "Asked In": ", ".join(t.get("asked_in", []) or []),
                })

            st.dataframe(topic_rows, use_container_width=True, hide_index=True)

            st.markdown("## 📝 Important / Predicted Questions")

            for q in result.get("predicted_questions", []):
                with st.expander(
                    f"Q{q.get('number','')} — {q.get('marks','')} marks — {q.get('topic','')}"
                ):
                    st.write(q.get("question", ""))
                    st.markdown(f"**Chapter:** {q.get('chapter', '')}")
                    st.markdown(f"**Topic:** {q.get('topic', '')}")
                    st.markdown(
                        f"**Previously asked in:** "
                        f"{', '.join(q.get('asked_in', []) or []) or 'Not established'}"
                    )
                    st.markdown(f"**Type:** {q.get('type', '')}")
                    st.caption(q.get("why_important", ""))

            if result.get("important_note"):
                st.info(result["important_note"])

    # --------------------------------------------------------
    # PDF REPORT
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("Well-formatted PDF Report")

        result = st.session_state.analysis

        if not result:
            st.info("Run the analysis first.")
        else:
            latex_source = make_latex(
                selected_paper,
                target_attempt,
                result,
            )

            pdf_bytes, compile_message = compile_latex(
                latex_source,
                "ca_intermediate_prediction",
            )

            if pdf_bytes:
                st.success(compile_message)
            elif REPORTLAB_AVAILABLE:
                st.warning(
                    "pdflatex is not installed on this server, so the app is using "
                    "a clean ReportLab PDF fallback. The .tex source is also available."
                )
                pdf_bytes = make_reportlab_pdf(
                    selected_paper,
                    target_attempt,
                    result,
                )
            else:
                st.error(
                    "Neither pdflatex nor ReportLab is available. "
                    "Install one of them before exporting PDF."
                )

            if pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name="CA_Intermediate_Prediction_Report.pdf",
                    mime="application/pdf",
                )

            st.download_button(
                "⬇️ Download LaTeX Source (.tex)",
                data=latex_source.encode("utf-8"),
                file_name="CA_Intermediate_Prediction_Report.tex",
                mime="text/plain",
            )


if __name__ == "__main__":
    render()
