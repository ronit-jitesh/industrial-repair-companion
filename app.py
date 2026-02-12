#!/usr/bin/env python3
"""
app.py
Phase 4: Streamlit UI — Industrial Repair Companion
Split-screen diagnostic assistant: Official Manual + Field Notes
"""

import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ─── Configuration ───────────────────────────────────────────────────────────

MANUALS_INDEX_DIR = "indices/manuals_index"
HISTORY_INDEX_DIR = "indices/history_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial Repair Companion",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background-color: #0e1117;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a5276 0%, #2e86c1 50%, #1abc9c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26, 188, 156, 0.2);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Result cards */
    .result-card {
        background: #1a1d23;
        border: 1px solid #2d333b;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .result-card:hover {
        border-color: #2e86c1;
    }

    .manual-card {
        border-left: 4px solid #2e86c1;
    }
    .history-card {
        border-left: 4px solid #1abc9c;
    }

    .card-source {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #2d333b;
    }

    .card-content {
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .card-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-fixed { background: #1a4731; color: #3fb950; }
    .badge-pending { background: #3d2e00; color: #d29922; }
    .badge-replaced { background: #4a1219; color: #f85149; }
    .badge-temp { background: #1c2541; color: #58a6ff; }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2d333b;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.1rem;
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .stat-item {
        background: #161b22;
        border: 1px solid #2d333b;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        flex: 1;
        text-align: center;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar styling */
    .sidebar-info {
        background: #161b22;
        border: 1px solid #2d333b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Loading animation */
    .stSpinner > div {
        border-top-color: #2e86c1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Resources (cached) ────────────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    """Load the embedding model (cached across reruns)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def load_indices():
    """Load both FAISS indices (cached across reruns)."""
    embeddings = load_embeddings()
    manuals_store = FAISS.load_local(
        MANUALS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    history_store = FAISS.load_local(
        HISTORY_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return manuals_store, history_store


@st.cache_data
def load_repair_data():
    """Load the cleaned repair logs for sidebar stats."""
    return pd.read_csv("data/repair_logs_cleaned.csv")


# ─── LLM Integration ────────────────────────────────────────────────────────

def generate_response(query, manual_context, history_context):
    """
    Generate a diagnostic response using Ollama (Llama 3).
    Falls back to displaying raw context if Ollama is unavailable.
    """
    try:
        import ollama

        system_prompt = """You are an expert Siemens SINAMICS VFD diagnostic assistant.
You have access to two sources of information:
1. OFFICIAL MANUAL: Technical documentation from Siemens SINAMICS manuals
2. FIELD HISTORY: Real repair logs from technicians who fixed similar issues

Based on the user's query, provide:
- A clear diagnosis based on the manual
- Practical field-tested solutions from historical repairs
- Safety warnings where applicable

Be concise, technical, and actionable. Reference specific fault codes and parameters."""

        user_prompt = f"""QUERY: {query}

MANUAL CONTEXT:
{manual_context}

FIELD HISTORY CONTEXT:
{history_context}

Provide a structured diagnostic response with:
1. DIAGNOSIS: What the fault code/symptom indicates
2. OFFICIAL PROCEDURE: Steps from the manual
3. FIELD-PROVEN FIXES: What actually worked based on historical data
4. SAFETY NOTES: Any relevant safety warnings"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"], True

    except Exception as e:
        return None, False


# ─── Outcome Badge ───────────────────────────────────────────────────────────

def outcome_badge(outcome):
    """Return HTML badge for outcome."""
    badge_map = {
        "Fixed": "badge-fixed",
        "Pending": "badge-pending",
        "Replaced_Unit": "badge-replaced",
        "Temporary_Fix": "badge-temp",
    }
    css = badge_map.get(outcome, "badge-temp")
    return f'<span class="card-badge {css}">{outcome}</span>'


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Industrial Repair Companion</h1>
        <p>Hybrid Diagnostic Assistant — Siemens SINAMICS VFD Systems</p>
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    try:
        manuals_store, history_store = load_indices()
        df = load_repair_data()
        indices_loaded = True
    except Exception as e:
        st.error(f"⚠️ Failed to load indices: {e}")
        st.info("Run `python ingest.py` first to build the FAISS indices.")
        indices_loaded = False
        return

    # ─── Sidebar ─────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Number of results
        k_manuals = st.slider("Manual results", 1, 10, 3, key="k_manuals")
        k_history = st.slider("History results", 1, 20, 5, key="k_history")

        st.divider()

        # Database stats
        st.markdown("### 📊 Database Stats")
        st.markdown(f"""
        <div class="sidebar-info">
            <div><strong>📖 Manual Chunks:</strong> {manuals_store.index.ntotal:,}</div>
            <div><strong>🔧 Repair Logs:</strong> {history_store.index.ntotal:,}</div>
            <div><strong>🏭 Sites:</strong> {df['Site_Location'].nunique()}</div>
            <div><strong>👷 Technicians:</strong> {df['Technician_ID'].nunique()}</div>
            <div><strong>⚡ Error Codes:</strong> {df['Error_Code'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Filters
        st.markdown("### 🔍 Filters")
        filter_error = st.multiselect(
            "Error Codes",
            options=sorted(df["Error_Code"].unique()),
            default=[],
            key="filter_error",
        )

        filter_site = st.multiselect(
            "Site Location",
            options=sorted(df["Site_Location"].unique()),
            default=[],
            key="filter_site",
        )

        st.divider()

        # Quick queries
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "Overheating error F30002 high temperature",
            "Motor encoder signal error with vibration",
            "DC link undervoltage during brownout",
            "Cooling fan failure maintenance procedure",
            "Ground fault on motor cable",
            "I2t overload frequent start-stop",
            "EEPROM data error after power loss",
        ]
        for eq in example_queries:
            if st.button(f"🔍 {eq[:40]}...", key=f"eq_{eq}", use_container_width=True):
                st.session_state.query_input = eq

    # ─── Main Content ────────────────────────────────────────────────────

    # Search input
    query = st.text_input(
        "🔍 Describe the symptom, error code, or issue:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., 'Overheating error F30002' or 'Motor vibration at high speed'",
        key="search_box",
    )

    if query:
        with st.spinner("🔍 Searching manuals and repair history..."):
            # Retrieve from both indices
            manual_results = manuals_store.similarity_search_with_score(query, k=k_manuals)
            history_results = history_store.similarity_search_with_score(query, k=k_history)

            # Apply filters to history results if any
            if filter_error or filter_site:
                filtered_history = []
                for doc, score in history_results:
                    if filter_error and doc.metadata.get("error_code") not in filter_error:
                        continue
                    if filter_site and doc.metadata.get("site_location") not in filter_site:
                        continue
                    filtered_history.append((doc, score))
                history_results = filtered_history if filtered_history else history_results

        # ─── Stats Bar ───────────────────────────────────────────────

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("📖 Manual Matches", len(manual_results))
        with col_s2:
            st.metric("🔧 History Matches", len(history_results))
        with col_s3:
            if history_results:
                fixed_count = sum(1 for d, _ in history_results if d.metadata.get("outcome") == "Fixed")
                fix_rate = fixed_count / len(history_results) * 100
                st.metric("✅ Fix Rate", f"{fix_rate:.0f}%")
            else:
                st.metric("✅ Fix Rate", "N/A")
        with col_s4:
            if manual_results:
                best_score = manual_results[0][1]
                st.metric("🎯 Best Match", f"{(1-best_score)*100:.0f}%")
            else:
                st.metric("🎯 Best Match", "N/A")

        st.divider()

        # ─── Split Screen ────────────────────────────────────────────

        col_manual, col_history = st.columns(2)

        # LEFT: Official Manual Procedure
        with col_manual:
            st.markdown("""
            <div class="section-header">
                <h3>📖 Official Manual Procedure</h3>
            </div>
            """, unsafe_allow_html=True)

            if manual_results:
                for i, (doc, score) in enumerate(manual_results):
                    source = doc.metadata.get("source", "Unknown Manual")
                    filename = doc.metadata.get("filename", "")
                    section = doc.metadata.get("section", "")
                    chunk_id = doc.metadata.get("chunk_id", "?")
                    relevance = (1 - score) * 100

                    with st.container():
                        st.markdown(f"""
                        <div class="result-card manual-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <strong style="color:#58a6ff;">#{i+1} — {source}</strong>
                                <span class="card-badge" style="background:#1c2541;color:#58a6ff;">
                                    {relevance:.0f}% match
                                </span>
                            </div>
                            <div class="card-content">{doc.page_content[:500]}</div>
                            <div class="card-source">
                                📄 Source: {source} | Chunk #{chunk_id}
                                {f' | Section: {section}' if section else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No manual results found for this query.")

        # RIGHT: Field Notes / Historical Fixes
        with col_history:
            st.markdown("""
            <div class="section-header">
                <h3>🔧 Field Notes / Historical Fixes</h3>
            </div>
            """, unsafe_allow_html=True)

            if history_results:
                for i, (doc, score) in enumerate(history_results):
                    log_id = doc.metadata.get("log_id", "?")
                    tech_name = doc.metadata.get("technician_name", "Unknown")
                    tech_id = doc.metadata.get("technician_id", "")
                    date = doc.metadata.get("date", "")
                    machine = doc.metadata.get("machine_id", "")
                    error_code = doc.metadata.get("error_code", "")
                    error_desc = doc.metadata.get("error_description", "")
                    notes = doc.metadata.get("technician_notes", "")
                    outcome = doc.metadata.get("outcome", "")
                    temp = doc.metadata.get("operating_temp", "")
                    vib = doc.metadata.get("vibration_level", "")
                    site = doc.metadata.get("site_location", "")
                    relevance = (1 - score) * 100

                    with st.container():
                        st.markdown(f"""
                        <div class="result-card history-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <strong style="color:#3fb950;">#{i+1} — Log #{log_id}</strong>
                                <div>
                                    {outcome_badge(outcome)}
                                    <span class="card-badge" style="background:#1c2541;color:#58a6ff;">
                                        {relevance:.0f}% match
                                    </span>
                                </div>
                            </div>
                            <div class="card-content">
                                <strong>{error_code}</strong> — {error_desc}<br/>
                                🛠️ <em>{notes}</em>
                            </div>
                            <div style="display:flex; gap:1rem; margin-top:0.5rem; flex-wrap:wrap;">
                                <span style="color:#8b949e; font-size:0.8rem;">🌡️ {temp}°C</span>
                                <span style="color:#8b949e; font-size:0.8rem;">📳 Vib: {vib}</span>
                                <span style="color:#8b949e; font-size:0.8rem;">🏭 {site}</span>
                                <span style="color:#8b949e; font-size:0.8rem;">⚙️ {machine}</span>
                            </div>
                            <div class="card-source">
                                👷 Source: Log #{log_id} | {tech_name} ({tech_id}) | {date}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No historical results found for this query.")

        # ─── AI-Generated Diagnosis (Ollama) ─────────────────────────

        st.divider()
        st.markdown("### 🤖 AI-Generated Diagnostic Summary")

        # Prepare context strings
        manual_context = "\n\n".join([
            f"[{doc.metadata.get('source', 'Manual')}]: {doc.page_content[:300]}"
            for doc, _ in manual_results
        ])
        history_context = "\n\n".join([
            f"[Log #{doc.metadata.get('log_id', '?')} | {doc.metadata.get('outcome', '?')}]: "
            f"{doc.metadata.get('technician_notes', '')} (Temp: {doc.metadata.get('operating_temp', '?')}°C, "
            f"Machine: {doc.metadata.get('machine_id', '?')})"
            for doc, _ in history_results
        ])

        with st.spinner("🤖 Generating AI diagnosis with Llama 3..."):
            ai_response, llm_available = generate_response(query, manual_context, history_context)

        if llm_available and ai_response:
            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid #a333c8;">
                <div class="card-content">{ai_response}</div>
                <div class="card-source">
                    🤖 Generated by Llama 3 via Ollama | Based on {len(manual_results)} manual chunks + {len(history_results)} repair logs
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Ollama/Llama 3 not available. Showing raw retrieved context below.")
            with st.expander("📖 Raw Manual Context", expanded=True):
                st.text(manual_context)
            with st.expander("🔧 Raw History Context", expanded=True):
                st.text(history_context)

    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding:3rem 0;">
            <h2 style="color:#58a6ff;">Welcome to the Industrial Repair Companion</h2>
            <p style="color:#8b949e; font-size:1.1rem; max-width:600px; margin:1rem auto;">
                Enter a symptom, error code, or issue description above to get
                <strong>dual-source diagnostics</strong> from official Siemens manuals
                and real-world repair history.
            </p>
            <div style="display:flex; justify-content:center; gap:2rem; margin-top:2rem;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;">📖</div>
                    <div style="color:#2e86c1; font-weight:600;">Official Manuals</div>
                    <div style="color:#8b949e; font-size:0.85rem;">Siemens SINAMICS docs</div>
                </div>
                <div style="font-size:2rem; color:#2d333b;">+</div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;">🔧</div>
                    <div style="color:#1abc9c; font-weight:600;">Field History</div>
                    <div style="color:#8b949e; font-size:0.85rem;">10,500+ repair logs</div>
                </div>
                <div style="font-size:2rem; color:#2d333b;">→</div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;">🤖</div>
                    <div style="color:#a333c8; font-weight:600;">AI Diagnosis</div>
                    <div style="color:#8b949e; font-size:0.85rem;">Powered by Llama 3</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show quick stats
        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📖 Manual Chunks", f"{manuals_store.index.ntotal:,}")
        with col2:
            st.metric("🔧 Repair Logs", f"{history_store.index.ntotal:,}")
        with col3:
            st.metric("🏭 Sites", f"{df['Site_Location'].nunique()}")
        with col4:
            st.metric("👷 Technicians", f"{df['Technician_ID'].nunique()}")
        with col5:
            st.metric("⚡ Error Codes", f"{df['Error_Code'].nunique()}")


if __name__ == "__main__":
    main()
