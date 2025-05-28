import streamlit as st
import openai
import pandas as pd
import docx
import fitz  # PyMuPDF
import tempfile
import io
import re

# --- CONFIGURATION ---
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
MODEL = "gpt-4o"

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_file(filepath):
    if filepath.endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    elif filepath.endswith(".docx"):
        return extract_text_from_docx(filepath)
    else:
        raise ValueError("Unsupported file type")

def compare_clause(document_text, term_sheet_df):
    system_prompt = """
    You are a legal AI assistant. Your task is to evaluate an NDA against a list of 34 standard legal issues provided by the legal department.

    The term sheet defines preferred and fallback positions for both unilateral NDAs and mutual NDAs (MNDAs). Your steps are:
    1. Determine if the NDA content provided is unilateral or mutual.
    2. Go through each row in the term sheet.
    3. For each issue:
    - Check if the relevant term is present in the NDA.
    - Evaluate if it aligns with the preferred position (based on whether it's unilateral or mutual).
    - If it does not align, check if the fallback position is acceptable.
    - If neither, suggest a fallback.
    4. Build a markdown table with the following columns:
    - Issue
    - Compliance Status
    - Reference from NDA
    - Suggested Fallback (if needed) worded in functional legal terms
    5. Make sure none of the clauses in NDA_Term_Sheet.csv are missing in the output.
    Sort the table by Compliance Status (Missing → Non-compliant → Compliant). Numbering must start from 1. 
    Be concise but specific.

    Output the table in clean markdown format starting with a single header row.
    """

    user_prompt = f"""
    Below is the NDA content:
    {document_text}

    And here is the NDA_Term_Sheet.csv content:
    {term_sheet_df.to_csv(index=False)}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def parse_markdown_table(md_text):
    lines = md_text.strip().splitlines()
    table_lines = [line for line in lines if line.strip().startswith('|')]
    if len(table_lines) < 3:
        return pd.DataFrame()
    headers = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    rows = []
    for line in table_lines[2:]:  # Skip header and separator
        cols = [c.strip() for c in line.split('|') if c.strip()]
        if len(cols) == len(headers):
            rows.append(cols)
    return pd.DataFrame(rows, columns=headers)

# --- STREAMLIT UI ---
st.title("🔍 NDA Compliance Checker")

uploaded_file = st.file_uploader("Upload an NDA (.docx or .pdf)", type=["docx", "pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    document_text = parse_file(tmp_path)
    term_sheet_df = pd.read_csv("NDA_Term_Sheet.csv")

    st.info("Evaluating uploaded NDA against standard clauses...")
    with st.spinner("Calling OpenAI API..."):
        compliance_md = compare_clause(document_text, term_sheet_df)
        compliance_df = parse_markdown_table(compliance_md)

    st.subheader("🧾 Compliance Table")
    if not compliance_df.empty:
        st.dataframe(compliance_df, use_container_width=True)
    else:
        st.error("⚠️ Unable to parse table from GPT output. See raw response below:")
        st.markdown(compliance_md)
