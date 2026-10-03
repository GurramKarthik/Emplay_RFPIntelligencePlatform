import streamlit as st
import os

from ingestion.pipeline import run_ingestion_pipeline
from search.indexer import run_indexing_pipeline
from search import bm25_index

st.set_page_config(page_title="RFP Intelligence Platform", layout="wide")

st.title("RFP Intelligence Platform")
st.markdown("Automated RAG Search Engine & Multi-Agent System")

# Create tabs for the 3 steps
tab1, tab2, tab3 = st.tabs(["1. Index Bid Folder", "2. Extraction Mode", "3. Query Agent"])

# ==========================================
# STEP 1: Folder Upload / Indexing
# ==========================================
with tab1:
    st.header("Step 1: Index a New Bid")
    st.markdown("Enter the path to a new RFP bid folder to process (Part A) and index it (Part B).")
    
    col1, col2 = st.columns(2)
    with col1:
        bid_folder = st.text_input("Folder Path (e.g., ./Bid1)")
    with col2:
        bid_id = st.text_input("Unique Bid ID (e.g., Bid1)")
        
    if st.button("Process & Index Bid", type="primary"):
        if not bid_folder or not bid_id:
            st.error("Please provide both a folder path and a Bid ID.")
        elif not os.path.exists(bid_folder):
            st.error(f"The folder path '{bid_folder}' does not exist.")
        else:
            with st.status(f"Processing Bid: {bid_id}...", expanded=True) as status:
                try:
                    # Run Part A
                    st.write("🏃 Running Part A: Ingestion & Parsing...")
                    summary = run_ingestion_pipeline(bid_id=bid_id, bid_folder=bid_folder)
                    st.write(f"✅ Parsed {len(summary.processed)} files successfully.")
                    
                    if summary.failed:
                        st.warning(f"⚠️ Failed to parse {len(summary.failed)} files.")
                        
                    # Run Part B
                    st.write("🏃 Running Part B: RAG Indexing (Qdrant & BM25)...")
                    result = run_indexing_pipeline(bid_id=bid_id)
                    
                    if result["status"] == "success":
                        st.write(f"✅ Indexed {result['chunks_indexed']} chunks successfully.")
                        status.update(label="Indexing Complete!", state="complete", expanded=False)
                        st.success(f"Bid '{bid_id}' is now fully indexed and ready for AI Extraction!")
                    else:
                        st.error(f"Indexing failed: {result.get('reason')}")
                        status.update(label="Indexing Failed", state="error")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    status.update(label="Process Failed", state="error")

# ==========================================
# STEP 2: Extraction Mode
# ==========================================
with tab2:
    st.header("Step 2: AI Extraction Mode")
    st.markdown("Run the Multi-Agent System to extract structured fields from an indexed bid.")
    
    bids = bm25_index.list_indexed_bids()
    if not bids:
        st.info("No bids indexed yet. Please go to Step 1 to index a bid first.")
    else:
        ext_bid_id = st.selectbox("Select a Bid to Extract", bids)
        
        all_fields = [
            "Bid Number", "Title", "Due Date", "Bid Submission Type",
            "Term of Bid", "Pre Bid Meeting", "Installation",
            "Bid Bond Requirement", "Delivery Date", "Payment Terms",
            "Any Additional Documentation Required", "MFG for Registration",
            "Contract or Cooperative to use", "Model_no", "Part_no",
            "Product", "contact_info", "company_name",
            "Bid Summary", "Product Specification"
        ]
        
        extract_all = st.checkbox("Extract all 20 fields (Takes longer)", value=False)
        selected_fields = []
        if extract_all:
            selected_fields = all_fields
        else:
            selected_fields = st.multiselect("Or select specific fields to test quickly:", all_fields, default=["Due Date", "Bid Number"])
            
        st.info("💡 **Note on API Limits:** The free tier of Gemini is limited to 15 Requests Per Minute (RPM). Because this platform uses parallel processing to extract all 20 fields simultaneously for maximum speed, you may hit this rate limit, causing some fields to return as `null`. If you see `null` fields, simply **wait for 2 minutes** and click the button again to retry them!")
            
        if st.button("Run Multi-Agent Extraction", type="primary"):
            if not selected_fields:
                st.warning("Please select at least one field to extract.")
            else:
                with st.status("Agents are reading the bid...", expanded=True) as status:
                    from agents.orchestrator import app as langgraph_app
                    initial_state = {
                        "bid_id": ext_bid_id,
                        "task_mode": "extraction",
                        "field_plan": selected_fields
                    }
                    st.write(f"🚀 Launching pipeline for {len(selected_fields)} fields...")
                    try:
                        final_state = langgraph_app.invoke(initial_state)
                        status.update(label="Extraction Complete!", state="complete", expanded=False)
                        st.success(f"Final output saved to: ./output/{ext_bid_id}_output.json")
                        
                        out = final_state.get("final_output")
                        if out:
                            st.json(out.model_dump())
                    except Exception as e:
                        status.update(label="Extraction Failed", state="error")
                        st.error(f"Error: {str(e)}")

# ==========================================
# STEP 3: Query Agent
# ==========================================
with tab3:
    st.header("Step 3: QA & Query Agent")
    st.markdown("Ask free-form questions about the bids. The agent will search and cite its sources.")
    
    # We must retrieve bids again because it might have changed, but st.selectbox gets it from the top anyway
    qa_bid_options = ["__global__ (Search All Bids)"] + bm25_index.list_indexed_bids()
    qa_bid_id = st.selectbox("Select Target", qa_bid_options)
    
    query = st.text_input("Ask a question:", placeholder="What is the warranty for Bid1?")
    
    if st.button("Ask Agent", type="primary"):
        if not query:
            st.warning("Please enter a question.")
        else:
            target_bid = "__global__" if "global" in qa_bid_id else qa_bid_id
            with st.status("Agent is searching and thinking...", expanded=True) as status:
                from agents.orchestrator import app as langgraph_app
                initial_state = {
                    "bid_id": target_bid,
                    "task_mode": "qa",
                    "query": query
                }
                try:
                    final_state = langgraph_app.invoke(initial_state)
                    status.update(label="Answer Ready!", state="complete", expanded=False)
                    out = final_state.get("final_output")
                    if out and out.extracted_fields:
                        answer = out.extracted_fields.get("answer", "No answer generated.")
                        st.markdown(f"### Answer\n{answer}")
                        
                        sources = out.extracted_fields.get("sources", [])
                        if sources:
                            st.markdown("### Citations")
                            for s in sources:
                                st.caption(f"- **{s.get('file', 'Unknown')}** (Page {s.get('page', '?')})")
                except Exception as e:
                    status.update(label="Query Failed", state="error")
                    st.error(f"Error: {str(e)}")
