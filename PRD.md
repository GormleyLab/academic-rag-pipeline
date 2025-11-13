# Product Requirements Document: Academic RAG Pipeline for Research & Citation Management

## Document Information

**Version:** 1.0  
**Date:** November 11, 2025  
**Author:** Adam (Associate Professor, Biomedical Engineering, Rutgers University)  
**Status:** Initial Release

---

## Executive Summary

The Academic RAG Pipeline is a locally-hosted, privacy-focused system designed to transform how researchers interact with their paper libraries during grant writing and academic writing. By combining advanced document processing (Docling), vector search (LanceDB), and AI assistance (Claude via MCP), this tool enables intelligent citation management, contextual paper retrieval, and automated bibliography generation for LaTeX documents, with particular focus on NIH grant applications.

---

## 1. Product Vision

### 1.1 Problem Statement

Academic researchers face several challenges when writing grants and papers:

- **Fragmented knowledge**: Papers stored as PDFs lack searchable structure and context
- **Manual citation management**: Time-consuming bibliography creation and verification
- **Context loss**: Difficulty remembering which papers support specific claims
- **Citation accuracy**: Ensuring proper attribution and complete reference lists
- **Literature discovery**: Finding related work and understanding research lineage

### 1.2 Solution Overview

A RAG (Retrieval Augmented Generation) pipeline that:

1. Processes PDF research papers into structured, searchable chunks
2. Extracts complete citation metadata automatically (via DOI/CrossRef API)
3. Provides intelligent search through conversational AI (Claude Desktop)
4. Generates publication-ready BibTeX files for LaTeX documents
5. Maintains URL links to online versions for easy access
6. Runs entirely locally for maximum privacy and control

### 1.3 Target Users

**Primary:** Academic researchers (PhD students, postdocs, faculty) writing:
- NIH and NSF grant applications
- Journal articles and conference papers
- Literature reviews and dissertations

**Use Case Focus:** Users with personal PDF libraries (100-10,000 papers) who write in LaTeX and need sophisticated citation management.

---

## 2. Core Architecture

### 2.1 Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Document Processing** | Docling | Superior academic PDF parsing, preserves structure, handles equations/tables |
| **Vector Database** | LanceDB | High-performance local storage, serverless, Python-native |
| **Embeddings** | OpenAI `text-embedding-3-large` | State-of-the-art semantic search, 3072 dimensions |
| **Chunking** | Docling Hybrid Chunker | Balances semantic coherence with embedding model constraints |
| **AI Interface** | Claude Desktop + MCP | Natural language interaction, tool integration, local execution |
| **Citation Metadata** | CrossRef API | Authoritative source for DOI-based bibliographic data |
| **Bibliography Format** | BibTeX | Standard for LaTeX, compatible with natbib/biblatex |

### 2.2 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Desktop App                    │
│                  (User Interface Layer)                  │
└────────────────────┬────────────────────────────────────┘
                     │ MCP Protocol (stdio)
                     │
┌────────────────────▼────────────────────────────────────┐
│              MCP Server (Python)                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Tools:                                            │  │
│  │  - search_papers()                                │  │
│  │  - add_paper_from_file()                          │  │
│  │  - add_paper_from_url()                           │  │
│  │  - generate_bibliography()                        │  │
│  │  - get_paper_details()                            │  │
│  │  - verify_citation()                              │  │
│  │  - update_bibtex()                                │  │
│  │  - database_stats()                               │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Docling  │  │ LanceDB  │  │  CrossRef    │
│ Pipeline │  │  Vector  │  │     API      │
│          │  │   Store  │  │              │
└──────────┘  └──────────┘  └──────────────┘
      │              │              │
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│   PDF    │  │ Embedded │  │   BibTeX     │
│ Library  │  │  Chunks  │  │  Metadata    │
│  Folder  │  │          │  │              │
└──────────┘  └──────────┘  └──────────────┘
```

### 2.3 Data Flow

**Initial Setup:**
1. User points to folder containing PDF papers
2. System processes each PDF with Docling → Markdown
3. Extracts DOI from each paper
4. Queries CrossRef API for BibTeX metadata
5. Falls back to PDF metadata extraction if no DOI
6. Chunks documents using Hybrid Chunker
7. Generates embeddings via OpenAI API
8. Stores in LanceDB with complete metadata

**Query Flow:**
1. User asks Claude a question about research topic
2. Claude calls `search_papers()` MCP tool
3. Query converted to embedding via OpenAI
4. LanceDB performs similarity search
5. Returns relevant chunks with citation metadata
6. Claude synthesizes response with inline citations
7. User can request bibliography generation

**Addition Flow:**
1. User drags PDF into Claude or provides URL
2. Claude calls `add_paper_from_file()` or `add_paper_from_url()`
3. System processes document (same pipeline as initial setup)
4. Returns confirmation with extracted metadata
5. Paper immediately available for search

---

## 3. Functional Requirements (v1.0)

### 3.1 Initial Database Creation

**FR-1.1: PDF Directory Processing**
- **Description:** Process all PDFs in a specified directory to populate initial database
- **Input:** Path to folder containing PDF files
- **Process:**
  - Recursively discover all `.pdf` files
  - Process each with Docling converter
  - Export to Markdown format
  - Extract document structure (sections, paragraphs, tables, figures)
- **Output:** Structured document collection ready for chunking
- **Error Handling:** 
  - Skip corrupted/encrypted PDFs with warning
  - Continue processing remaining files
  - Log all failures for review

**FR-1.2: Document Chunking**
- **Description:** Break documents into semantically meaningful, searchable chunks
- **Method:** Docling Hybrid Chunker
- **Configuration:**
  - Embedding model: `text-embedding-3-large` (max 8191 tokens)
  - Target chunk size: ~1000 tokens
  - Overlap: 100-200 tokens for context preservation
- **Chunk Metadata:** Each chunk includes:
  - Original paper metadata (title, authors, year, etc.)
  - Section information (section title, hierarchy)
  - Page number
  - Element type (paragraph, table, figure, equation)
- **Output:** Collection of chunks ready for embedding

**FR-1.3: Citation Metadata Extraction**
- **Description:** Extract complete bibliographic metadata for each paper
- **Priority Order:**
  1. **DOI Extraction & CrossRef Lookup**
     - Search for DOI patterns in first 5 pages of document
     - Common patterns: `10.xxxx/xxxxx`, `doi:`, `DOI:`
     - Query CrossRef API: `https://api.crossref.org/works/{doi}/transform/application/x-bibtex`
     - Parse returned BibTeX entry
  2. **arXiv Identifier**
     - Search for arXiv ID pattern: `arXiv:YYMM.NNNNN`
     - Construct URL: `https://arxiv.org/abs/{arxiv_id}`
     - Fetch metadata from arXiv API
  3. **PubMed ID**
     - Search for PMID pattern: `PMID: 12345678`
     - Construct URL: `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`
  4. **PDF Metadata**
     - Extract from PDF properties (title, author, subject, keywords)
     - Use PyMuPDF to read metadata fields
  5. **Document Parsing**
     - Extract title from first page (largest text/heading)
     - Extract authors from below title
     - Find year in document text
     - Construct minimal BibTeX entry

**FR-1.4: BibTeX Entry Construction**
- **Description:** Create valid BibTeX entries with all required fields
- **Required Fields:**
  - `@article{key,` or `@inproceedings{key,` (entry type)
  - `title = {Paper Title},`
  - `author = {Last1, First1 and Last2, First2},`
  - `year = {2024},`
  - `bibtex_key = {FirstAuthorLastname2024}` (generated)
- **Optional Fields (when available):**
  - `journal = {Journal Name},`
  - `volume = {10},`
  - `pages = {1-20},`
  - `doi = {10.xxxx/xxxxx},`
  - `url = {https://doi.org/10.xxxx/xxxxx},`
  - `publisher = {Publisher Name},`
  - `abstract = {...},`
- **BibTeX Key Generation:**
  - Format: `FirstAuthorLastname + Year`
  - Example: `Smith2024`
  - Handle collisions by appending `a`, `b`, etc.: `Smith2024a`

**FR-1.5: URL Assignment**
- **Description:** Ensure every paper has an accessible online URL
- **Priority Order:**
  1. DOI URL: `https://doi.org/{doi}`
  2. arXiv URL: `https://arxiv.org/abs/{arxiv_id}`
  3. PubMed URL: `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`
  4. Publisher URL from metadata
  5. `null` if no URL available
- **Storage:** Store in metadata as `url` field

**FR-1.6: Vector Embedding Generation**
- **Description:** Generate semantic embeddings for all chunks
- **Model:** OpenAI `text-embedding-3-large`
- **Dimensions:** 3072
- **Input:** Chunk text (cleaned, no special characters)
- **Batch Processing:** Process in batches of 100 for efficiency
- **Error Handling:** Retry failed embeddings with exponential backoff

**FR-1.7: LanceDB Storage**
- **Description:** Store chunks with embeddings and metadata in LanceDB
- **Schema:**
```python
{
  "id": str,                    # Unique chunk ID: {paper_id}_chunk_{n}
  "text": str,                  # Chunk content
  "vector": List[float],        # 3072-dimensional embedding
  
  # Paper-level metadata
  "paper_id": str,              # Unique paper identifier (hash)
  "bibtex_key": str,            # Citation key (e.g., Smith2024)
  "bibtex_entry": str,          # Full BibTeX entry
  "title": str,                 # Paper title
  "authors": List[str],         # Author names
  "year": int,                  # Publication year
  "journal": str,               # Journal/venue name
  "doi": str,                   # Digital Object Identifier
  "url": str,                   # Online access URL
  "pdf_path": str,              # Local PDF path
  "pdf_hash": str,              # SHA256 hash for duplicate detection
  "date_added": str,            # ISO timestamp
  
  # Chunk-level metadata
  "chunk_index": int,           # Position in document
  "section_title": str,         # Section name
  "section_hierarchy": List[str], # Full section path
  "page_number": int,           # Page location
  "element_type": str,          # paragraph, table, figure, equation
  
  # Optional metadata
  "tags": List[str],            # User-defined tags
  "notes": str,                 # User notes
  "extraction_method": str      # crossref, arxiv, pubmed, pdf_metadata, parsed
}
```
- **Indexing:** Create ANN index for fast similarity search
- **Persistence:** Store database in local directory (e.g., `./lancedb_data/`)

### 3.2 MCP Server Implementation

**FR-2.1: MCP Server Setup**
- **Description:** Python-based MCP server for Claude Desktop integration
- **Transport:** stdio (standard input/output)
- **Framework:** `mcp` Python package
- **Configuration File:** `claude_desktop_config.json`
  - Location (macOS): `~/Library/Application Support/Claude/`
  - Location (Windows): `%APPDATA%\Claude\`
- **Server Registration:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "/path/to/project/.venv/Scripts/python.exe",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/project",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```
**Note:** Use `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on macOS/Linux.

**FR-2.2: Tool: search_papers**
- **Description:** Search the paper database for relevant content
- **Input Parameters:**
  - `query` (required): Search query string
  - `n_results` (optional, default=5): Number of results to return (1-20)
  - `filter_section` (optional): Filter by section type (Methods, Results, Discussion, Introduction)
  - `min_year` (optional): Only papers from this year onwards
  - `filter_tags` (optional): Filter by user-defined tags
- **Process:**
  1. Convert query to embedding via OpenAI API
  2. Perform similarity search in LanceDB
  3. Retrieve top-k chunks with metadata
  4. Deduplicate by paper (show each paper once)
  5. Format results with inline URLs
- **Output Format:**
```markdown
**[Smith2024]** Machine Learning for Polymer Design
📝 Smith, J., Jones, A., Chen, L. (2024)
📍 Nature Materials
🔗 [Read online](https://doi.org/10.1038/s41563-024-01234)

**Relevant excerpt** (Section: Results, Page 5):
Our deep learning model achieved 95% accuracy in predicting polymer
synthesis outcomes, representing a 30% improvement over previous methods...

---

**[Chen2023]** Graph Neural Networks for Materials Discovery
...
```

**FR-2.3: Tool: add_paper_from_file**
- **Description:** Add a PDF paper to the database via drag-and-drop
- **Input Parameters:**
  - `file_path` (required): Path to PDF file
  - `custom_tags` (optional): List of tags to associate with paper
- **Process:**
  1. Verify file exists and is a PDF
  2. Check for duplicates using PDF hash
  3. Process with Docling pipeline (same as initial setup)
  4. Extract metadata (DOI → CrossRef → fallback)
  5. Generate chunks and embeddings
  6. Store in LanceDB
  7. Return confirmation with extracted metadata
- **Output:** Success message with paper details and BibTeX key
- **Error Handling:**
  - Duplicate detection: Inform user, skip indexing
  - Parsing errors: Return detailed error message
  - Missing metadata: Use fallback methods, flag for review

**FR-2.4: Tool: add_paper_from_url**
- **Description:** Add a paper by providing DOI, arXiv ID, or URL
- **Input Parameters:**
  - `url_or_identifier` (required): DOI, arXiv ID, or URL
- **Process:**
  1. Identify type (DOI, arXiv, URL)
  2. Fetch metadata from appropriate API
  3. If URL provided:
     - Use Docling to extract from HTML/PDF
     - Process as regular paper
  4. If only metadata available:
     - Store metadata-only entry (no full-text chunks)
     - Flag as "not fully indexed"
  5. Store in database
- **Output:** Confirmation with paper details
- **Note:** Enables adding papers to bibliography without local PDF

**FR-2.5: Tool: generate_bibliography**
- **Description:** Create a .bib file with specified papers
- **Input Parameters:**
  - `bibtex_keys` (required): List of BibTeX keys to include
  - `output_path` (optional, default=`./references.bib`): Where to save file
  - `include_abstracts` (optional, default=false): Include abstract field
- **Process:**
  1. Query database for each BibTeX key
  2. Retrieve complete BibTeX entries
  3. Validate entries (check for required fields)
  4. Write to .bib file with proper formatting
  5. Return summary of included/missing citations
- **Output Format:**
```bibtex
@article{Smith2024,
  title = {Machine Learning Approaches for Polymer Design},
  author = {Smith, John and Jones, Alice and Chen, Li},
  year = {2024},
  journal = {Nature Materials},
  volume = {23},
  pages = {123-145},
  doi = {10.1038/s41563-024-01234},
  url = {https://doi.org/10.1038/s41563-024-01234}
}

@article{Chen2023,
  ...
}
```

**FR-2.6: Tool: get_paper_details**
- **Description:** Retrieve complete information about a specific paper
- **Input Parameters:**
  - `bibtex_key` (required): The BibTeX key of the paper
- **Output:** Full paper metadata including:
  - Title, authors, year, journal
  - DOI and URL
  - Number of chunks indexed
  - Date added to database
  - Tags and notes
  - Complete BibTeX entry
  - Extraction method used

**FR-2.7: Tool: verify_citation**
- **Description:** Review and validate citation metadata for accuracy
- **Input Parameters:**
  - `bibtex_key` (required): Paper to verify
- **Output:** 
  - Extracted metadata with confidence scores
  - Extraction method used
  - Any missing or questionable fields flagged
  - Comparison with CrossRef data (if DOI available)
- **Use Case:** User can review automatically extracted metadata before relying on it

**FR-2.8: Tool: update_bibtex**
- **Description:** Manually update or override BibTeX entry for a paper
- **Input Parameters:**
  - `bibtex_key` (required): Paper to update
  - `bibtex_entry` (optional): Complete replacement BibTeX entry
  - `field_updates` (optional): Specific fields to update (dict)
- **Process:**
  1. Locate paper in database
  2. Parse and validate new BibTeX entry
  3. Update metadata in all associated chunks
  4. Preserve chunk embeddings (only metadata changes)
  5. Log update with timestamp
- **Use Case:** Correct errors in automated extraction

**FR-2.9: Tool: import_bibtex_file**
- **Description:** Import an existing .bib file to override metadata
- **Input Parameters:**
  - `bib_file_path` (required): Path to .bib file
  - `match_strategy` (optional): How to match entries (doi, title, fuzzy)
- **Process:**
  1. Parse .bib file
  2. Match entries to papers in database
  3. Update metadata for matched papers
  4. Report successful matches and unmatched entries
- **Use Case:** Import hand-curated bibliography from previous work

**FR-2.10: Tool: database_stats**
- **Description:** Get statistics about the paper database
- **Output:**
  - Total papers indexed
  - Total chunks stored
  - Average publication year
  - Papers by year distribution
  - Most cited papers (if citation graph enabled)
  - Database size on disk
  - Papers with missing metadata

**FR-2.11: Tool: list_recent_papers**
- **Description:** Show recently added papers
- **Input Parameters:**
  - `n` (optional, default=10): Number of papers to show
  - `include_unverified` (optional, default=true): Include papers flagged for review
- **Output:** List of recent additions with key metadata

### 3.3 User Interface (Claude Desktop)

**FR-3.1: Natural Language Interaction**
- **Supported Query Types:**
  - Search: "Find papers about machine learning in polymer synthesis"
  - Addition: "Add this paper to my database" (with drag-drop PDF)
  - Bibliography: "Generate a .bib file with Smith2024, Jones2023, Chen2022"
  - Details: "Show me information about the Wang2024 paper"
  - Stats: "How many papers do I have on reinforcement learning?"
  - URL retrieval: "Get me the link to the Chen paper"

**FR-3.2: Drag-and-Drop PDF Support**
- Claude Desktop allows dragging PDFs directly into chat
- System extracts file path and calls appropriate MCP tool
- User receives immediate feedback on processing status

**FR-3.3: Conversation Context**
- Claude maintains conversation history
- Can reference previously discussed papers
- Builds bibliography incrementally across conversation
- Example flow:
  1. "Find papers about topic X" → Results with citations
  2. "Tell me more about [Smith2024]" → Detailed info
  3. "Add that to my bibliography" → Noted
  4. "Also include [Jones2023]" → Added
  5. "Now generate the .bib file" → Creates file

### 3.4 Error Handling & Edge Cases

**FR-4.1: Duplicate Detection**
- Generate SHA256 hash of PDF content
- Check against existing hashes before processing
- If duplicate found: Inform user, skip indexing
- Handle renamed files correctly

**FR-4.2: Metadata Validation**
- Check BibTeX entries for required fields
- Flag papers with missing critical metadata
- Provide confidence scores for extracted information
- Allow user review of questionable extractions

**FR-4.3: API Rate Limiting**
- Implement exponential backoff for CrossRef API
- Batch OpenAI API calls for efficiency
- Cache API responses locally
- Handle network failures gracefully

**FR-4.4: Corrupted/Encrypted PDFs**
- Detect and skip corrupted files
- Inform user about password-protected PDFs
- Log all processing failures
- Provide option to retry individual papers

**FR-4.5: Missing DOIs**
- Fallback to PDF metadata extraction
- Fallback to document parsing
- Flag papers for manual BibTeX entry
- Support manual DOI addition later

---

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-1.1: Processing Speed**
- Initial setup: Process 100 papers in < 30 minutes (standard laptop)
- Single paper addition: < 2 minutes per paper
- Search latency: < 2 seconds for query results
- Bibliography generation: < 5 seconds for 50 citations

**NFR-1.2: Scalability**
- Support libraries of 1,000-10,000 papers
- LanceDB memory footprint: < 8GB for 5,000 papers
- Embedding storage: ~500MB-5GB depending on library size

**NFR-1.3: Reliability**
- 99% uptime for MCP server (local)
- Graceful degradation if OpenAI API unavailable
- Automatic recovery from interrupted processing

### 4.2 Usability

**NFR-2.1: Setup Complexity**
- Initial setup: < 15 minutes for technical users
- Single configuration file
- Clear error messages with actionable guidance
- Documentation with examples

**NFR-2.2: Learning Curve**
- Core workflows learnable in < 30 minutes
- Natural language interface (no query syntax to learn)
- Comprehensive README with common use cases

### 4.3 Privacy & Security

**NFR-3.1: Local-First Architecture**
- All PDFs stored locally (no upload to cloud)
- Vector database stored locally
- Only API calls: OpenAI embeddings, CrossRef metadata
- No paper content sent to external services (only queries/metadata)

**NFR-3.2: Credential Management**
- OpenAI API key stored in environment variables
- Never logged or transmitted insecurely
- Support for .env files

### 4.4 Maintainability

**NFR-4.1: Code Organization**
- Modular architecture with clear separation of concerns
- Comprehensive docstrings and type hints
- Unit tests for critical functions
- Integration tests for MCP tools

**NFR-4.2: Dependency Management**
- Explicit version pinning for reproducibility
- requirements.txt and/or pyproject.toml
- Virtual environment recommended

**NFR-4.3: Logging**
- Structured logging for debugging
- Log levels: DEBUG, INFO, WARNING, ERROR
- Separate logs for: processing, API calls, errors
- Log rotation to prevent disk fill

---

## 5. Technical Specifications

### 5.1 Development Environment

**Language:** Python 3.10+

**Required Packages:**
```txt
# Document Processing
docling>=2.0.0
docling-core>=2.0.0

# Vector Database
lancedb>=0.3.0
pyarrow>=14.0.0

# Embeddings
openai>=1.0.0

# MCP Framework
mcp>=0.9.0

# Citation Metadata
requests>=2.31.0
pybtex>=0.24.0

# PDF Processing
pymupdf>=1.23.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

**Directory Structure:**
```
academic-rag-pipeline/
├── src/
│   ├── __init__.py
│   ├── mcp_server.py              # MCP server implementation
│   ├── document_processor.py       # Docling pipeline
│   ├── metadata_extractor.py       # Citation metadata extraction
│   ├── vector_store.py             # LanceDB operations
│   ├── embeddings.py               # OpenAI embedding generation
│   ├── bibliography.py             # BibTeX management
│   └── utils.py                    # Helper functions
├── config/
│   ├── config.yaml                 # Configuration file
│   └── claude_desktop_config.json  # MCP server config
├── data/
│   ├── lancedb/                    # Vector database
│   ├── pdfs/                       # PDF library
│   └── logs/                       # Application logs
├── tests/
│   ├── test_processor.py
│   ├── test_metadata.py
│   └── test_mcp_tools.py
├── docs/
│   ├── setup.md                    # Setup instructions
│   ├── usage.md                    # Usage guide
│   └── troubleshooting.md          # Common issues
├── scripts/
│   ├── initial_setup.py            # Initialize database
│   └── maintenance.py              # Database maintenance
├── .env.example                    # Environment variables template
├── requirements.txt
├── pyproject.toml
├── README.md
└── PRD.md                          # This document
```

### 5.2 Configuration

**config.yaml:**
```yaml
# PDF Library
pdf_library_path: "/path/to/pdfs"

# Database
lancedb_path: "./data/lancedb"
vector_dimension: 3072

# Embeddings
embedding_model: "text-embedding-3-large"
openai_api_key: "${OPENAI_API_KEY}"  # From environment
batch_size: 100

# Chunking
chunking_strategy: "hybrid"
max_chunk_tokens: 1000
chunk_overlap: 150

# Metadata
crossref_api_url: "https://api.crossref.org/works"
crossref_email: "user@example.com"  # For polite API use

# MCP Server
server_name: "academic-rag"
log_level: "INFO"

# Bibliography
default_bib_output: "./references.bib"
bibtex_entry_type: "article"  # Default if unknown
```

**.env:**
```bash
OPENAI_API_KEY=sk-...
CROSSREF_EMAIL=your.email@university.edu
```

### 5.3 API Specifications

**CrossRef API Integration:**
- Endpoint: `https://api.crossref.org/works/{doi}/transform/application/x-bibtex`
- Method: GET
- Headers: `User-Agent: AcademicRAG/1.0 (mailto:{email})`
- Rate Limit: 50 requests/second (polite pool)
- Retry Strategy: 3 attempts with exponential backoff

**OpenAI API Integration:**
- Model: `text-embedding-3-large`
- Dimensions: 3072
- Max input tokens: 8191
- Batch size: 100 embeddings per request
- Rate limit handling: Respect `Retry-After` headers

### 5.4 Data Persistence

**LanceDB Schema Version:** 1.0
- Schema migrations supported for future versions
- Backward compatibility maintained

**Backup Strategy:**
- LanceDB automatic snapshotting
- Metadata exported to JSON periodically
- PDF library maintained separately by user

---

## 6. Implementation Phases

### Phase 1: MVP (v1.0) - Target: 4-6 weeks

**Week 1-2: Core Infrastructure**
- [ ] Project setup and dependency management
- [ ] Docling integration for PDF processing
- [ ] Markdown export and chunking (Hybrid Chunker)
- [ ] OpenAI embedding generation
- [ ] LanceDB setup and basic CRUD operations

**Week 3-4: Metadata & Citation Management**
- [ ] DOI extraction from PDFs
- [ ] CrossRef API integration
- [ ] BibTeX parsing and construction
- [ ] PDF metadata extraction (fallback)
- [ ] URL assignment logic
- [ ] Metadata validation and error handling

**Week 5-6: MCP Server & Tools**
- [ ] MCP server framework setup
- [ ] Tool: search_papers
- [ ] Tool: add_paper_from_file
- [ ] Tool: generate_bibliography
- [ ] Tool: get_paper_details
- [ ] Tool: database_stats
- [ ] Claude Desktop integration and testing

**Deliverables:**
- Functional RAG pipeline processing PDF libraries
- MCP server integrated with Claude Desktop
- Core search and citation management features
- Documentation for setup and basic usage

### Phase 2: Enhanced Features (v1.5) - Target: 2-3 weeks

**Enhancements:**
- [ ] Tool: add_paper_from_url (DOI/arXiv/URL support)
- [ ] Tool: verify_citation (metadata review)
- [ ] Tool: update_bibtex (manual overrides)
- [ ] Tool: import_bibtex_file (.bib file import)
- [ ] Improved error handling and recovery
- [ ] Processing progress indicators
- [ ] Duplicate detection improvements

**Deliverables:**
- Advanced metadata management
- URL-based paper addition
- Robust error handling
- Enhanced documentation

### Phase 3: Citation Graph (v2.0) - Target: 4-6 weeks

**New Features:**
- [ ] Reference extraction from papers
- [ ] Citation relationship storage (separate graph DB)
- [ ] Tool: explore_citation_network
- [ ] Tool: find_foundational_papers
- [ ] Citation count tracking
- [ ] Related paper discovery
- [ ] Co-citation analysis

**Deliverables:**
- Full citation graph functionality
- Network visualization (optional)
- Research lineage tracking
- Foundational paper identification

### Phase 4: Advanced Retrieval (v2.5) - Target: 3-4 weeks

**New Features:**
- [ ] Section-aware retrieval (weight Methods/Results)
- [ ] Recency weighting for grants
- [ ] Figure/table extraction and storage
- [ ] Caption indexing
- [ ] Cross-reference validation
- [ ] Claim verification suggestions

**Deliverables:**
- Specialized retrieval modes
- Visual content support
- Improved accuracy for grant writing

---

## 7. Success Metrics

### 7.1 Technical Metrics

- **Processing Accuracy:** > 95% successful metadata extraction (DOI-based)
- **Search Relevance:** Top-5 results contain relevant papers > 90% of time
- **System Uptime:** > 99% availability for local MCP server
- **Query Latency:** < 2 seconds average response time

### 7.2 User Metrics

- **Setup Time:** < 15 minutes for initial configuration
- **Time Saved:** 50%+ reduction in citation management time
- **User Satisfaction:** Positive feedback on ease of use
- **Adoption:** Active use for grant/paper writing within 2 weeks

### 7.3 Quality Metrics

- **BibTeX Accuracy:** < 5% error rate in generated bibliographies
- **Duplicate Detection:** 100% accuracy in identifying duplicate PDFs
- **Citation Completeness:** > 95% of papers have valid BibTeX entries

---

## 8. Risks & Mitigations

### 8.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **OpenAI API costs** | Medium | High | Implement caching, batch processing; provide cost monitoring |
| **CrossRef API downtime** | Low | Low | Implement fallback to PDF metadata; cache successful lookups |
| **Docling parsing failures** | Medium | Medium | Multiple fallback strategies; allow manual BibTeX entry |
| **LanceDB performance degradation** | Medium | Low | Optimize indexing; implement pagination for large libraries |
| **PDF encryption/corruption** | Low | Medium | Graceful error handling; clear user messaging |

### 8.2 Usability Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Complex setup process** | High | Medium | Detailed documentation; setup script; video tutorial |
| **Learning curve for MCP** | Medium | Low | Natural language interface reduces complexity |
| **Metadata extraction errors** | High | Medium | Verification tools; manual override capability |

### 8.3 Scope Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Feature creep** | High | High | Strict phase-based development; MVP focus |
| **Citation graph complexity** | Medium | Medium | Phase 3 implementation; proven algorithms |
| **Over-engineering** | Medium | Medium | Agile approach; user feedback loops |

---

## 9. Dependencies & Constraints

### 9.1 External Dependencies

- **OpenAI API:** Required for embeddings (paid service)
- **CrossRef API:** Preferred for metadata (free, rate-limited)
- **Docling:** Open-source, actively maintained
- **LanceDB:** Open-source, Python-native
- **Claude Desktop:** Required for MCP integration

### 9.2 Constraints

- **Local execution:** System designed for local machines (not cloud)
- **LaTeX focus:** BibTeX output optimized for LaTeX workflows
- **English language:** Initial focus on English-language papers
- **PDF format:** Primary support for PDF documents

### 9.3 Assumptions

- User has local PDF library of research papers
- User writes in LaTeX (uses BibTeX for citations)
- User has OpenAI API access and budget
- User has Claude Desktop or API access
- Papers have DOIs or sufficient metadata

---

## 10. Future Enhancements (v3.0+)

### 10.1 Potential Features

**Advanced Analytics:**
- Research trend visualization
- Collaboration network mapping
- Impact factor tracking
- Citation velocity metrics

**Enhanced Integration:**
- Zotero/Mendeley import
- Google Scholar integration
- Semantic Scholar API
- PubMed bulk import
- LaTeX template integration

**Multi-User Support:**
- Shared lab databases
- Collaborative bibliographies
- Access control
- Cloud deployment option

**AI-Powered Features:**
- Automatic paper summarization
- Key findings extraction
- Research gap identification
- Related work suggestions
- Draft text generation with citations

**Content Features:**
- Full-text search in figures/tables
- Equation search
- Supplementary material indexing
- Dataset linking

**Quality Assurance:**
- Citation accuracy validation
- Cross-reference verification
- Retraction checking
- Preprint vs. published version tracking

### 10.2 Research Directions

- **Citation graph algorithms:** PageRank for paper importance
- **Semantic clustering:** Identify research themes automatically
- **Temporal analysis:** Track idea evolution over time
- **Multi-lingual support:** Non-English paper processing
- **Cross-modal search:** Query with figures, equations, or tables

---

## 11. Support & Maintenance

### 11.1 Documentation

**Required Documentation:**
- README.md: Quick start guide
- SETUP.md: Detailed installation instructions
- USAGE.md: Comprehensive usage guide
- TROUBLESHOOTING.md: Common issues and solutions
- API.md: MCP tool reference
- CONTRIBUTING.md: Guidelines for contributions

### 11.2 Testing Strategy

**Unit Tests:**
- Document processing pipeline
- Metadata extraction functions
- BibTeX parsing and generation
- Embedding generation
- Vector store operations

**Integration Tests:**
- MCP server tools
- End-to-end workflows
- API interactions
- Error handling scenarios

**User Acceptance Testing:**
- Real-world grant writing scenarios
- Edge cases (missing metadata, corrupted PDFs)
- Performance under load (large libraries)

### 11.3 Maintenance Plan

**Regular Tasks:**
- Update dependencies (quarterly)
- Review and update documentation
- Monitor API changes (OpenAI, CrossRef)
- Database optimization (annually)

**Issue Tracking:**
- GitHub Issues for bug reports
- Feature request categorization
- Priority tagging (critical, high, medium, low)

---

## 12. Appendices

### Appendix A: Glossary

- **RAG:** Retrieval Augmented Generation - AI technique combining information retrieval with text generation
- **MCP:** Model Context Protocol - Standard for connecting AI models to external tools
- **BibTeX:** Reference management system for LaTeX
- **DOI:** Digital Object Identifier - Unique identifier for scholarly articles
- **Vector Database:** Database optimized for similarity search using embeddings
- **Embedding:** Numerical representation of text capturing semantic meaning
- **Chunking:** Breaking documents into smaller pieces for processing

### Appendix B: Reference Architecture Diagram

*(See Section 2.2 for text-based architecture diagram)*

### Appendix C: Sample Configuration Files

**claude_desktop_config.json:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "/Users/adam/academic-rag-pipeline/.venv/bin/python",
      "args": [
        "/Users/adam/academic-rag-pipeline/src/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/adam/academic-rag-pipeline",
        "OPENAI_API_KEY": "sk-...",
        "CONFIG_PATH": "/Users/adam/academic-rag-pipeline/config/config.yaml"
      }
    }
  }
}
```
**Note:** Use `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on macOS/Linux.

### Appendix D: Example Workflows

**Workflow 1: Initial Setup**
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `.env` with API keys
4. Update `config.yaml` with PDF library path
5. Run initial setup: `python scripts/initial_setup.py`
6. Configure Claude Desktop MCP server
7. Restart Claude Desktop
8. Test: "How many papers are in my database?"

**Workflow 2: Adding a Paper**
1. Download PDF to any location
2. Open Claude Desktop
3. Drag PDF into chat
4. Say: "Add this paper to my database"
5. Claude processes and confirms with metadata
6. Paper immediately searchable

**Workflow 3: Writing Grant Significance Section**
1. "Find papers about machine learning in drug delivery from the last 3 years"
2. Review results
3. "Tell me more about [Smith2024] and [Chen2023]"
4. "Use these papers to help me write a Significance section about the current state of ML in drug delivery"
5. Claude drafts text with inline citations
6. "Generate a bibliography with all the papers you cited"
7. Claude creates references.bib
8. Copy text and .bib file to LaTeX project

### Appendix E: Estimated Costs

**Development Costs:**
- Development time: 160-240 hours (4-6 weeks full-time)
- Testing & documentation: 40-60 hours

**Operational Costs (per month):**
- OpenAI API (embeddings): $10-50 depending on library size and query frequency
  - Initial library (1000 papers): ~$15 one-time
  - Ongoing queries: ~$5-10/month for typical usage
  - Adding papers: ~$0.01-0.02 per paper
- Storage: Minimal (local storage)
- No hosting costs (local execution)

**Total First Year Cost:** ~$175-250 (API usage only)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-11 | Adam | Initial release |

---

**End of Product Requirements Document**