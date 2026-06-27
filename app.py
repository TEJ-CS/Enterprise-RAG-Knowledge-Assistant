import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from io import BytesIO
from reportlab.pdfgen import canvas

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Knowledge Assistant",
    layout="wide"
)

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        rgba(0,0,0,0.28),
        rgba(0,0,0,0.35)
    ),
    url("https://images.unsplash.com/photo-1526378722484-bd91ca387e72");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

.header {
    text-align:center;
    font-size:28px;
    font-weight:700;
    padding:14px;
    background:rgba(255,255,255,0.06);
    border-radius:12px;
    margin-bottom:10px;
}

.card {
    background:rgba(255,255,255,0.08);
    padding:12px;
    border-radius:10px;
    margin:8px 0;
    border-left:3px solid #6366f1;
}

.user {
    background:#4f46e5;
    padding:10px;
    border-radius:10px;
    margin:8px 0;
    max-width:80%;
    margin-left:auto;
}

.bot {
    background:rgba(255,255,255,0.08);
    padding:10px;
    border-radius:10px;
    margin:8px 0;
    max-width:80%;
}

.stButton > button {
    width:100%;
    border-radius:10px;
    background:#6366f1;
    color:white;
}

section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.65);
}

</style>
""", unsafe_allow_html=True)

# =========================
# STATE
# =========================
if "db" not in st.session_state:
    st.session_state.db = None
    st.session_state.chat_sessions = {}
    st.session_state.current_chat = "default"

# =========================
# EMBEDDINGS
# =========================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# PDF PROCESS
# =========================
def process_pdf(file):

    MAX_PAGES = 1000

    path = f"temp_{file.name}"

    with open(path, "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader(path)
    docs = loader.load()

    if len(docs) > MAX_PAGES:
        st.error(
            f"{file.name} has {len(docs)} pages. Maximum allowed is {MAX_PAGES} pages."
        )
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embedding = load_embeddings()

    db = FAISS.from_documents(
        split_docs[:500],
        embedding
    )

    for i in range(500, len(split_docs), 500):
        db.add_documents(split_docs[i:i+500])

    return db
# =========================
# CHAT ANSWER
# =========================
def get_answer(db, q):

    docs = db.similarity_search(q, k=6)

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    context = context[:12000]

    response = llm.invoke(
        f"""
Answer only from the provided context.

Context:
{context}

Question:
{q}
"""
    )

    sources = list(
        set(
            [
                f"Page {d.metadata.get('page',0)+1}"
                for d in docs
            ]
        )
    )

    return response.content, sources

# =========================
# KEY POINTS
# =========================
def generate_key_points(db):

    docs = db.similarity_search(
        "important topics key concepts",
        k=100
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    context = context[:12000]

    prompt = f"""
Extract the most important key points.

Context:
{context}

Return concise bullet points.
"""

    response = llm.invoke(prompt)

    return response.content

# =========================
# QUESTIONS
# =========================
def generate_important_questions(db):

    docs = db.similarity_search(
        "important concepts important topics",
        k=100
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    context = context[:12000]

    prompt = f"""
Generate all possible important interview/exam questions from the document.
Rules:
- Cover every major topic.
- Include easy, medium and hard questions.
- Do not stop at 10.
- Generate as many high-quality questions as possible.
- Return only numbered questions.

Context:
{context}
"""

    response = llm.invoke(prompt)

    return response.content

# =========================
# SUMMARY
# =========================
def generate_summary(db):

    docs = db.similarity_search(
        "document summary overview",
        k=100
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    context = context[:12000]

    prompt = f"""
Create a clear summary.

Context:
{context}
"""

    response = llm.invoke(prompt)

    return response.content

# =========================
# INSIGHTS
# =========================
def generate_insights(db):

    docs = db.similarity_search(
        "overview main topics summary",
        k=100
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    context = context[:12000]

    response = llm.invoke(
        f"""
Analyze this document and return:

- Main Topic
- Sub Topics
- Difficulty Level
- Key Themes
- One Line Summary

Context:
{context}
"""
    )

    return response.content

# =========================
# EXPORT PDF
# =========================
def export_chat_pdf(chat_data):

    buffer = BytesIO()
    c = canvas.Canvas(buffer)

    y = 800

    c.setFont("Helvetica", 10)

    for q, a, _ in chat_data:

        c.drawString(50, y, f"Q: {q}")
        y -= 15

        for line in a.split("\n"):

            c.drawString(60, y, line[:90])
            y -= 12

        y -= 20

        if y < 100:
            c.showPage()
            y = 800

    c.save()

    buffer.seek(0)

    return buffer

# =========================
# HEADER
# =========================
st.markdown(
    "<div class='header'>🧠 AI Knowledge Assistant</div>",
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.subheader("📄 Upload PDFs")

    file = st.file_uploader(
        "",
        type=["pdf"],
        accept_multiple_files=False
    )

    if st.button("🚀 Build Knowledge Base"):

        if file:

            db = process_pdf(file)

            if db is not None:
                st.session_state.db = db
                st.success("Knowledge Base Ready")

        else:
            st.warning("Upload PDFs first")

    st.markdown("---")

    st.subheader("📤 Export")

    if st.button("⬇ Download Chat PDF"):

        if "default" in st.session_state.chat_sessions:

            pdf_file = export_chat_pdf(
                st.session_state.chat_sessions["default"]
            )

            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("No chat available")

# =========================
# NAVIGATION
# =========================
nav = st.radio(
    "Navigation",
    [
        "💬 Chat",
        "🧠 Key Points",
        "❓ Questions",
        "📝 Summary",
        "📊 Insights"
    ],
    horizontal=True
)

# =========================
# CHAT
# =========================
if nav == "💬 Chat":

    q = st.text_input(
        "Ask anything from PDFs"
    )

    if q:

        if st.session_state.db:

            ans, src = get_answer(
                st.session_state.db,
                q
            )

            if "default" not in st.session_state.chat_sessions:
                st.session_state.chat_sessions["default"] = []

            st.session_state.chat_sessions["default"].append(
                (q, ans, src)
            )

        else:
            st.warning("Upload PDFs first")

    chat = st.session_state.chat_sessions.get(
        "default",
        []
    )

    for q, a, src in reversed(chat[-8:]):

        st.markdown(
            f"<div class='user'>🧑 {q}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='bot'>🤖 {a}</div>",
            unsafe_allow_html=True
        )

        with st.expander("📌 Sources"):

            for s in src:
                st.write("📄", s)

# =========================
# KEY POINTS
# =========================
elif nav == "🧠 Key Points":

    st.subheader("🧠 Important Key Points")

    if st.session_state.db:

        if st.button("Generate Key Points"):

            with st.spinner(
                "Extracting key points..."
            ):

                result = generate_key_points(
                    st.session_state.db
                )

            for point in result.split("\n"):

                if point.strip():

                    st.markdown(
                        f"<div class='card'>🔹 {point}</div>",
                        unsafe_allow_html=True
                    )

    else:
        st.warning("Upload PDFs first")

# =========================
# QUESTIONS
# =========================
elif nav == "❓ Questions":

    st.subheader("❓ Important Questions")

    if st.session_state.db:

        if st.button("Generate Questions"):

            with st.spinner(
                "Generating questions..."
            ):

                result = generate_important_questions(
                    st.session_state.db
                )

            for question in result.split("\n"):

                if question.strip():

                    st.markdown(
                        f"<div class='card'>❓ {question}</div>",
                        unsafe_allow_html=True
                    )

    else:
        st.warning("Upload PDFs first")


# =========================
# SUMMARY
# =========================
elif nav == "📝 Summary":

    st.subheader("📝 Document Summary")

    if st.session_state.db:

        if st.button("Generate Summary"):

            with st.spinner(
                "Creating summary..."
            ):

                result = generate_summary(
                    st.session_state.db
                )

            st.markdown(
                f"""
<div class='card'>
{result}
</div>
""",
                unsafe_allow_html=True
            )

    else:
        st.warning("Upload PDFs first")


# =========================
# INSIGHTS
# =========================
elif nav == "📊 Insights":

    st.subheader("📊 Document Insights")

    if st.session_state.db:

        if st.button("Generate Insights"):

            with st.spinner(
                "Analyzing document..."
            ):

                result = generate_insights(
                    st.session_state.db
                )

            st.markdown(
                "### 🧠 AI Analysis Report"
            )

            st.markdown(
                f"""
<div class='card'>
{result}
</div>
""",
                unsafe_allow_html=True
            )

    else:
        st.warning(
            "Please upload PDFs first"
        )