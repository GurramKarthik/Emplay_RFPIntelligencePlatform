# RFP Intelligence Platform

## 1. Brief of the System
The **RFP Intelligence Platform** is a production-grade, multi-agent AI system designed to automatically ingest, parse, search, and extract highly structured data from complex Request for Proposal (RFP) and solicitation documents. 

Public sector bids are notoriously dense, often containing critical requirements hidden inside tables, addendums, and hundreds of pages of unstructured text. This system solves that problem by combining:
- **Structure-Aware Chunking** (preserving tables and headings).
- **Hybrid Search & Reranking** (combining Vector Embeddings with BM25 keyword matching and a Cross-Encoder for maximum accuracy).
- **Multi-Agent Orchestration** (using specialized LangGraph agents for Retrieval, Extraction, Addendum Reconciliation, and Validation).

The result is a highly accurate system capable of achieving exact-match retrieval (100% Recall) on critical fields like SKUs, Deadlines, and Manufacturer Requirements, ultimately saving hundreds of hours of manual review.

---

## Directory Structure & Outputs

The platform is organized logically by pipeline parts. Below is a map of where the logic lives and exactly where the intermediate and final outputs are stored.

```text
Emplay_RFPIntelligencePlatform/
│
├── input/                  # 📥 Raw Input Data
│   └── Bid1/               # Place raw PDFs, HTML, and images here
│
├── data/                   # 💾 Intermediate Pipeline Outputs
│   ├── parse_jobs.db       # [Part A Output] SQLite tracker for ingestion
│   ├── chunks/             # [Part A Output] Extracted chunks stored as JSONL
│   └── bm25_indices/       # [Part B Output] Pickled BM25 keyword indices
│
├── qdrant_storage/         # 🧠 Vector DB Storage
│   └── ...                 # [Part B Output] Qdrant local collection data
│
├── output/                 # 📤 Final Deliverables
│   └── Bid1_output.json    # [Part D Output] The final extracted JSON data
│
├── logs/                   # 📜 Execution Logs
│   └── Bid1_agent_trace.log
│
├── ingestion/              # ⚙️ Part A Logic: Parsers, Chunker, Tracker
├── search/                 # ⚙️ Part B Logic: Indexer, Retriever, Query Understander
├── agents/                 # ⚙️ Part C/D Logic: LangGraph Orchestrator & Agents
├── eval/                   # 🧪 Evaluation scripts
├── tests/                  # 🧪 Unit Tests
├── Arch/                   # 🏗️ Detailed Architecture Documentation
│
├── app.py                  # Streamlit Web UI Entrypoint
└── main.py                 # CLI Entrypoint
```

# Results & Deliverables

- **Structured Output (JSON):** The system successfully extracts 20 highly structured data fields from complex, unstructured RFPs (including those with addendums). You can view the exact AI-generated extractions for the test bids in the `./output/` directory (e.g., `Bid1_output.json`, `Bid2_output.json`).
- **Retrieval Evaluation Report:** To review the empirical performance and accuracy of our Hybrid Search vs Vector-Only Search, please refer to the dedicated **[Retrieval Evaluation Report](./Retrieval_Evaluation_Report.md)**.  This report details our exact hit-rate metrics, testing methodology (using the `eval/run_eval.py` script), and explains the concrete data behind why we chose the Hybrid + Cross-Encoder architecture to achieve maximum extraction precision.
- **QA Semantic Search:** The QA Agent successfully parses conversational queries and grounds its answers precisely in the vector database. For a comprehensive demonstration of 10 real-world queries and their exact chunk citations, please review the **[Sample Q&A Log](./Sample_QA_Log.md)**.
- **Agent Observability Trace:** To see exactly how the LangGraph orchestrator delegates work between the Retrieval, Extraction, Reconciliation, and Validation agents, please view the **[Sample Agent Trace](./logs/Sample_Agent_Trace.log)**. It demonstrates a perfectly clean "happy path" extraction flow.
- **Comprehensive Unit Tests:** We've implemented automated unit tests verifying the integrity of the data pipeline (Part A), hybrid retriever (Part B), and agent orchestrator (Part C). Refer to the **[Running Unit Tests](#5-running-unit-tests)** section below for details on how to run them.


## 2. How to Setup the System

To ensure a smooth setup without errors, please follow these steps sequentially:

### Prerequisites
- **Python 3.10+** (Python 3.12 or 3.13 recommended)
- **Git** (optional, for cloning)

### Step 1: Create a Virtual Environment
It is highly recommended to isolate dependencies using a virtual environment. Open your terminal in the project root folder and run:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
With your virtual environment activated, install the required packages. The `requirements.txt` includes all necessary libraries like `qdrant-client`, `fastembed`, `langgraph`, and `streamlit`.
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
The system relies on LLMs for query understanding and data extraction. You need to provide API keys.
1. Create a file named `.env` in the root directory.
2. Add your API keys and configuration to the `.env` file:
```env
# Example .env file
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=your_gemini_api_key_here


LOG_LEVEL=INFO
```

---

## 3. How to Run the System

You can interact with the system via the Command Line Interface (CLI) for batch processing, or via the Streamlit Web UI for interactive exploration.

### 3.a Using the CLI (`main.py`)
The CLI is split into "Parts" to allow you to run the pipeline incrementally. This modularity prevents you from having to re-parse PDFs every time you want to test the LLM extraction.

*Note: Ensure no other scripts (like the Streamlit app) are running at the same time to avoid local database locking issues.*

**Part A: Ingestion & Parsing**
Ingests PDFs, HTML, and images, performs OCR, and intelligently chunks the text.
```bash
python main.py --part a --bid-id Bid1 --bid ./input/Bid1
```
*Why?* This converts unstructured raw files into clean, readable JSONL chunks stored in `./data/chunks`.

**Part B: Indexing**
Embeds the chunks and creates the Hybrid Search index (Qdrant Vector DB + BM25).
```bash
python main.py --part b --bid-id Bid1
```
*Why?* Makes the documents instantly searchable. We separate this from Part A so you can experiment with different chunking sizes without waiting for re-embedding.

**Part C: Multi-Agent Extraction**
Runs the LangGraph orchestration to extract all predefined fields (like Due Date, Specs, and SKUs) and saves them to a final JSON file.
```bash
python main.py --part c --bid-id Bid1 --mode extraction
```
*Why?* This is the core engine that delegates tasks to the Retrieval, Extraction, and Reconciliation agents to formulate the final output. The results will be saved to `./output/Bid1_output.json`.

**Part D: QA Agent (Ad-Hoc Querying)**
Allows you to ask free-form questions about the bid directly from the CLI.
```bash
python main.py --part c --bid-id Bid1 --mode qa --query "What is the warranty period?"
```
*Why?* This lets you test the semantic retrieval and QA Agent directly from your terminal without needing to run the full extraction batch or launch the UI.

*(Optional) Run the entire pipeline (Parts A, B, and C Extraction) end-to-end at once:*
```bash
python main.py --part all --bid-id Bid1 --bid ./input/Bid1
```

### 3.b Using the Streamlit UI (`app.py`)
For a visual and interactive experience, you can use the Streamlit application. 

**To launch the app:**
```bash
streamlit run app.py
```
**What it does:**
The Streamlit app opens a web dashboard in your browser (usually at `http://localhost:8501`) featuring three main tabs:

- **Tab 1: Index Bid Folder:** Allows you to input a local folder path (e.g., `c:/users/**/**/input/Bid1`) and a Bid ID. With one click, it runs Part A (Ingestion/Parsing) and Part B (Vector Indexing). It gives you a real-time progress bar of how many files were successfully parsed and chunked.
- **Tab 2: Extraction Mode:** Lets you select a previously indexed bid from a dropdown menu. You can choose to extract all 20 predefined fields (like Due Date, Product Specifications, Installation) or select a few to test quickly. It triggers the LangGraph agents and displays the final JSON output right in the browser, while saving it to `./output/`.
- **Tab 3: Query Agent:** A conversational interface where you can ask free-form questions (e.g., *"What is the warranty for Bid1?"*). You can search within a specific bid or globally across all indexed bids. The QA Agent will read the retrieved chunks, generate an answer, and list the exact file and page number citations as evidence.
---

## 4. How the System Works
The system is logically divided into four major components to ensure robustness, speed, and high accuracy.

**Key Design Decision: Chunk Size (512 Tokens)**
*Before diving into the pipeline, it is important to highlight our chunking strategy. We specifically configured the system to use chunks of **512 tokens** with a **64-token overlap**. Why this number? 512 tokens is the sweet spot for Retrieval-Augmented Generation (RAG). It is small enough to maintain high semantic density (so specific technical specs or keywords don't get drowned out by noise), but large enough to encapsulate a full cohesive paragraph or a complete tabular row without losing its surrounding context.*

### System Architecture & Data Flow Diagram

[![System Architecture & Data Flow Diagram](https://mermaid.ink/img/pako:eNqNVm1v4jgQ_iuWV3tqpcCFtwJZaaUWyu3etlcK1X7YpFq5yQR8DTFykgWu7X-_sR2nYaHaNRKMxzPPjOfNPNFQREA9GidiEy6ZzMndOJBBSnC9f0_GEPMUyDzfJZBZfpiwLMMTkuVCsgWQmCeJ9y4eqo-T5VI8gveu0-mUdGPDo3zptdfbDwcQaylCyLISAlpxL4YKwu2wYdz9DRR0Is2tG3HcAbfCiOPhwHV_A0MU-bqwIDCIe_B6l27I4t4bILVoTVX8zj3yOV1AlnNRnWXFw0Ky9fJQgvxBRssifeTpwgqrNRZh5s_YRhHFCi-XBUE6HU_In-TT3fXVPWk0PiqwDKR_zeQjyIY5ne7w974OZaS0graEGvNcFmFeSGicb5gEy99Tu5MsVLIn89srnoPdn6LpZuPjs95mZCrFQmL-nkszdYQSVVv-e37zz5Wvvw0_q4xBGh0L44UKUgRbDAzGaAa55PCDJccjisKfdg-SR4eCahm7yo_L1QNEEV5rwrJcb-6PC15ct3vfP0db_0RRxpXTPVkLpcVvI8nS3D8xv-QrhNgaZHyxr1KnbwuQO_9cF-4cmAyXhmVSO4eVYfpIISIPS6H7Awwt_wV2pTxSGyGjY-J1ujJQc79-XgHuRePNy5hrK9HZbOLPIOSqsVlCZix9JJMiw1Lf88UiWp36GW4NGxBV1eBIiixrXKZqVsmK_asKGmHdjD1yXSQ5b5hA3-CFsO8ke7s5j2hdblEjrKuo9YolpH_F0sVfGqTO1rl8RliWJJCQCYckwlbBGtXAflWsRO_3AmSFTH5UovdPTQy0gTuxJl_I5Q8eQRrCs_LX4L86fsSAldIGvrLEqCDBI-X7EQ0rhBokoBPGE4jISKSxMUxO1H12pwEts2fw30aY4uSFqBIPRWrDgmTIE86OuH6Y7Bs9uC3jFUfDTnjKEtXWvqascJBe8Kj13cz85r9ZWZ17j4Iewo4eCU4FUz4T-8-HHY6O6QOnKu7ygfxJ2oxKp5yPjh0kTtWTTtV9DvaCY5Pt1GvLvpw_YduoOza9jo25U4uMfi8_UIcucGZSD58DcOgK5IqpLX1SiAHNl7CCgHpISoiKbSMUiZABDdIXVF2z9JsQK6stRbFYUi9mSYa7Yo1FBGPOsLNWFVdi8kCORJHm1Ou6GoN6T3RLvU6_2zxruYPWmdvtdFtd_PNAd9RrD5uts26_33Pd9qA97PTPXhz6nzbrNgf97hBXf9gb9jv4_fI_VfrTfg)](https://mermaid.live/edit#pako:eNqNVm1v4jgQ_iuWV3tqpcCFtwJZaaUWyu3etlcK1X7YpFq5yQR8DTFykgWu7X-_sR2nYaHaNRKMxzPPjOfNPNFQREA9GidiEy6ZzMndOJBBSnC9f0_GEPMUyDzfJZBZfpiwLMMTkuVCsgWQmCeJ9y4eqo-T5VI8gveu0-mUdGPDo3zptdfbDwcQaylCyLISAlpxL4YKwu2wYdz9DRR0Is2tG3HcAbfCiOPhwHV_A0MU-bqwIDCIe_B6l27I4t4bILVoTVX8zj3yOV1AlnNRnWXFw0Ky9fJQgvxBRssifeTpwgqrNRZh5s_YRhHFCi-XBUE6HU_In-TT3fXVPWk0PiqwDKR_zeQjyIY5ne7w974OZaS0graEGvNcFmFeSGicb5gEy99Tu5MsVLIn89srnoPdn6LpZuPjs95mZCrFQmL-nkszdYQSVVv-e37zz5Wvvw0_q4xBGh0L44UKUgRbDAzGaAa55PCDJccjisKfdg-SR4eCahm7yo_L1QNEEV5rwrJcb-6PC15ct3vfP0db_0RRxpXTPVkLpcVvI8nS3D8xv-QrhNgaZHyxr1KnbwuQO_9cF-4cmAyXhmVSO4eVYfpIISIPS6H7Awwt_wV2pTxSGyGjY-J1ujJQc79-XgHuRePNy5hrK9HZbOLPIOSqsVlCZix9JJMiw1Lf88UiWp36GW4NGxBV1eBIiixrXKZqVsmK_asKGmHdjD1yXSQ5b5hA3-CFsO8ke7s5j2hdblEjrKuo9YolpH_F0sVfGqTO1rl8RliWJJCQCYckwlbBGtXAflWsRO_3AmSFTH5UovdPTQy0gTuxJl_I5Q8eQRrCs_LX4L86fsSAldIGvrLEqCDBI-X7EQ0rhBokoBPGE4jISKSxMUxO1H12pwEts2fw30aY4uSFqBIPRWrDgmTIE86OuH6Y7Bs9uC3jFUfDTnjKEtXWvqascJBe8Kj13cz85r9ZWZ17j4Iewo4eCU4FUz4T-8-HHY6O6QOnKu7ygfxJ2oxKp5yPjh0kTtWTTtV9DvaCY5Pt1GvLvpw_YduoOza9jo25U4uMfi8_UIcucGZSD58DcOgK5IqpLX1SiAHNl7CCgHpISoiKbSMUiZABDdIXVF2z9JsQK6stRbFYUi9mSYa7Yo1FBGPOsLNWFVdi8kCORJHm1Ou6GoN6T3RLvU6_2zxruYPWmdvtdFtd_PNAd9RrD5uts26_33Pd9qA97PTPXhz6nzbrNgf97hBXf9gb9jv4_fI_VfrTfg)

### Part A: Document Ingestion & Structure-Aware Chunking
**1. Technology & Libraries Used**
- **pdfplumber & PyMuPDF:** Used to extract text from PDFs. `pdfplumber` is the primary extractor because of its excellent table extraction capabilities (converting tables directly into clean Markdown strings), while `PyMuPDF` (fitz) acts as a robust fallback for complex, multi-column layouts.
- **beautifulsoup4:** Used to parse HTML documents accurately.
- **sqlite3:** Used to track ingestion progress and intermediate results. 

**2. Architecture & Explanation**
When a bid folder is processed, the system first registers every file in a local **SQLite database**. 
*Why SQLite?* Parsing complex PDFs (especially with OCR) is time-consuming and prone to random API failures or crashes. By using an embedded SQLite database to track the state (`PENDING`, `SUCCESS`, `FAILED`) of each file, the system can gracefully resume from exactly where it left off if a crash occurs, completely avoiding redundant and expensive processing.

After parsing, the text is passed to a **Structure-Aware Chunker**. Standard chunking destroys tables by splitting them mid-row. Our custom chunker splits tables intelligently by row while retaining the header in every chunk, guaranteeing that the LLM understands tabular relationships (e.g., knowing that a specific row corresponds to the "Tier 1 Requirements" column). 
*Implementation Detail:* We achieved this by detecting Markdown tables (`|---|`) during the chunking phase, explicitly extracting the first row as the header, and programmatically prepending this header to every subsequent chunk that contains data rows from that same table.

👉 **Deep Dive:** For a comprehensive technical breakdown of this process, see [Part A Architecture](./Architecture/part_a_architecture.md).
### Part B: Hybrid Retrieval & Reranking
**1. Technology & Libraries Used**
- **Qdrant:** The primary Vector Database. *Why Qdrant?* It is extremely fast, supports local embedded storage (no Docker required for the MVP), and handles metadata filtering natively.
- **FastEmbed:** Used for creating embeddings. *Why FastEmbed?* It runs locally on the CPU at extremely high speeds, removing the latency and cost associated with hitting external APIs for embeddings.
- **rank_bm25:** Used for keyword-based search.
- **sentence-transformers (Cross-Encoder):** Used to re-rank the combined search results.

**2. Architecture & Explanation**
RFP extraction requires both semantic understanding (*"What are the software requirements?"*) and exact exact matching (*"Model CC7802"*).
To achieve this, the system uses a **Hybrid Retriever**. It runs a semantic vector search (via Qdrant) and a keyword search (via BM25) simultaneously. 
The results from both are combined using Reciprocal Rank Fusion (RRF). Finally, a Cross-Encoder scores the top chunks against the user's query to surface the absolute best context to the top before passing it to the AI.

👉 **Deep Dive:** For a comprehensive technical breakdown of this process, see [Part B Architecture](./Architecture/part_b_architecture.md).
### Part C: Multi-Agent Orchestration (LangGraph)
**1. Technology & Libraries Used**
- **LangGraph:** Used to build cyclic, stateful multi-agent workflows. *Why LangGraph?* Standard LLM chains are strictly linear. LangGraph allows us to build a state machine where an agent can conditionally loop back and retry an extraction if validation fails.
- **concurrent.futures (ThreadPoolExecutor):** Native Python library used to run extractions in parallel.

**2. Architecture & Explanation**
Extracting 20 highly detailed fields sequentially using an LLM would take minutes. To solve this, the orchestrator groups the fields and processes them **in parallel** using a ThreadPoolExecutor. 
*Why Parallelism?* It reduces the total extraction time from minutes down to roughly 20-30 seconds, vastly improving the user experience.
The LangGraph state machine controls the flow:
1. **Retrieval Agent:** Fetches the top chunks for a specific field based on strict internal descriptions.
2. **Extraction Agent:** Uses the chunks to extract the exact value in JSON format.
3. **Reconciliation Agent:** Reviews the extracted data against any addendums to ensure newer updates override original requirements.
4. **Validator Agent:** Checks if the extraction meets confidence thresholds. If it fails, LangGraph explicitly loops the field back to the start for a retry.

**Ad-Hoc Querying (QA Agent)**
In addition to the automated batch extraction flow, the LangGraph orchestrator has a separate routing path for a dedicated **QA Agent**. This agent is designed to handle free-form, conversational questions from the user. It performs its own targeted semantic retrieval and generates a human-readable answer complete with exact source citations.

**Clean Observability Logging**
To maintain production-grade observability without the overhead of external tools like LangSmith, the orchestrator features a custom trace logger. Every action taken by the agents (Retrieval, Extraction, Validation) is recorded with execution latency, status, and precise output into a highly readable, formatted text log (e.g., `logs/BidX_agent_trace.log`). This allows developers to open a single clean file to instantly trace the exact chain of thought and execution time of the entire multi-agent system.

**Anti-Hallucination Guardrails & Error Handling**
To guarantee high fidelity and prevent the LLM from hallucinating, the system employs strict architectural guardrails:
- **Strict Grounding:** The Extraction Agent is explicitly prompted to return `null` if the exact requested value is not present in the retrieved chunks. It is strictly forbidden from inferring or guessing.
- **Schema Enforcement:** Every output must strictly adhere to a defined Pydantic schema. If the LLM returns malformed JSON or invalid types, a custom error handler catches the `ValidationError` and instructs the LangGraph state machine to automatically retry the extraction.
- **Confidence Thresholds:** The AI must assign a confidence score to its own extraction. The Validator Agent acts as a final gatekeeper, forcefully failing and looping the field back for retry if the confidence score drops below the minimum acceptable threshold.

👉 **Deep Dive:** For a comprehensive technical breakdown of this process, see [Part C Architecture](./Architecture/part_c_architecture.md).
### Part D: Reconciliation & Output Generation
**1. Technology & Libraries Used**
- **Pydantic:** Used to enforce strict type checking and schema validation on the final output.
- **JSON & Local File System:** Used for simple, portable persistence of the extracted data.

**2. Architecture & Explanation**
RFPs often have multiple "Addendums" (updates) that change original requirements. 
Before final output generation, a **Reconciliation Agent** reviews the extracted data. If it detects that a newer addendum contradicts the original RFP (e.g., *"Due date changed from July 2 to July 9"*), it overrides the old value and logs the reason. 
Finally, the validated and reconciled data is serialized via Pydantic into a clean `BidX_output.json` file, complete with confidence scores and chunk citations for every single field so you can easily trace the AI's reasoning.

👉 **Deep Dive:** For a comprehensive technical breakdown of this process, see [Part D Architecture](./Architecture/part_d_architecture.md).

---
### Quick Reference: I/O by Pipeline Part
- **Part A (Ingestion):** Reads from `./input/BidX/`. Writes to `./data/parse_jobs.db` and `./data/chunks/BidX_chunks.jsonl`.
- **Part B (Indexing):** Reads from `./data/chunks/`. Writes to `./qdrant_storage/` and `./data/bm25_indices/`.
- **Part C & D (Extraction):** Reads from Vector DBs. Writes the final structured output to `./output/BidX_output.json`.

---

## 5. Running Unit Tests

To ensure the core pipeline logic remains stable, we have implemented automated unit tests for Parts A, B, and C. The tests utilize `pytest` and mock external dependencies (like the LLM) to guarantee they run blazingly fast without incurring API costs.

You can run the entire test suite from the root directory:
```bash
pytest tests/ -v
```

**What is tested?**
- `test_part_a.py`: Verifies the Markdown/table chunker accurately splits rows while retaining headers.
- `test_part_b.py`: Mocks Qdrant to ensure the Hybrid Retriever properly fuses BM25 and Dense scores.
- `test_part_c.py`: Mocks the LangGraph state machine to verify the Error Handler correctly triggers retries when the Validator Agent flags low confidence.
