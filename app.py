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
MODEL = "gpt-3.5-turbo"  # Example model; replace as needed

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        doc = fitz.open(tmp.name)
        text = "\n".join(page.get_text() for page in doc)
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".doc"):
        st.warning(".doc parsing support is limited.")
        return ""
    else:
        st.error("Unsupported file type.")
        return ""

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
   - Compliance Status: "Compliant", "Missing", or "Non-compliant"
   - Reference from NDA: a snippet or phrase from the NDA that matches or is closest
   - Suggested Fallback (if needed)

Sort the table by Compliance Status (Missing → Non-compliant → Compliant).
Be concise but specific.
"""

    user_prompt = f"""
Below is the NDA content:
"""
{document_text}
"""

And here is the NDA_Term_Sheet.csv content:
"""
{term_sheet_df.to_csv(index=False)}
"""

Generate only the final compliance table in markdown format using | and --- for table headers.
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
    lines = [line for line in md_text.splitlines() if '|' in line and not line.strip().startswith('|---')]
    if not lines:
        return pd.DataFrame()
    headers = [col.strip() for col in lines[0].split('|') if col.strip()]
    data = []
    for line in lines[1:]:
        cols = [col.strip() for col in line.split('|') if col.strip()]
        if len(cols) == len(headers):
            data.append(cols)
    return pd.DataFrame(data, columns=headers)

# --- STREAMLIT UI ---
st.title("🔍 NDA Compliance Checker with OpenAI")

uploaded_file = st.file_uploader("Upload NDA (.pdf, .docx, .doc)", type=["pdf", "docx", "doc"])

if uploaded_file:
    st.success("NDA uploaded successfully. Parsing...")

    document_text = parse_file(uploaded_file)
    standards_df = pd.read_csv("NDA_Term_Sheet.csv")

    st.subheader("📄 Document Preview")
    st.text_area("Extracted Text (First 1000 characters)", document_text[:1000], height=200)

    st.subheader("✅ Clause Compliance Table")
    with st.spinner("Evaluating NDA against all 34 issues using OpenAI..."):
        compliance_table_md = compare_clause(document_text, standards_df)
        st.markdown(compliance_table_md, unsafe_allow_html=False)

        # Convert markdown table to DataFrame for CSV download
        compliance_df = parse_markdown_table(compliance_table_md)
        if not compliance_df.empty:
            csv = compliance_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Compliance Table as .csv",
                data=csv,
                file_name="compliance_table.csv",
                mime="text/csv"
            )

        # Offer download as Markdown
        markdown_bytes = compliance_table_md.encode('utf-8')
        st.download_button(
            label="📄 Download Compliance Table as .md",
            data=markdown_bytes,
            file_name="compliance_table.md",
            mime="text/markdown"
        )

    st.info("Scroll down to copy or export the table as needed.")
