import streamlit as st
import openai
import pandas as pd
import docx
import fitz  # PyMuPDF
import tempfile

# --- CONFIGURATION ---
openai.api_key = st.secrets["openai"]["api_key"]
MODEL = "gpt-4"

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
    instructions = f"""
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

Below is the NDA content:
"""
{document_text}
"""

And here is the NDA_Term_Sheet.csv content:
"""
{term_sheet_df.to_csv(index=False)}
"""

Generate only the final compliance table.
    """

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": instructions}],
        temperature=0.2
    )

    return response['choices'][0]['message']['content']

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
    with st.spinner("Evaluating NDA against all 34 issues using GPT-4..."):
        compliance_table = compare_clause(document_text, standards_df)
        st.markdown(compliance_table)

    st.info("Scroll down to copy or export the table as needed.")
