"""
Vector store module using LanceDB for storing and querying paper chunks.
"""

import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import lancedb
import pyarrow as pa

from src.utils import setup_logger
from src.metadata_extractor import PaperMetadata, ExtractionMethod
from src.document_processor import DocumentChunk


logger = setup_logger(__name__)


@dataclass
class ChunkRecord:
    """Complete record for a document chunk in the vector store."""
    # Unique identifiers
    id: str
    paper_id: str

    # Chunk content and embedding
    text: str
    vector: list[float]

    # Paper-level metadata
    bibtex_key: str
    bibtex_entry: str
    title: str
    authors: str  # Comma-separated list
    year: int
    journal: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    pdf_path: str
    pdf_hash: str
    date_added: str

    # Chunk-level metadata
    chunk_index: int
    section_title: str
    section_hierarchy: str  # JSON string of list
    page_number: Optional[int]
    element_type: str

    # Optional metadata
    tags: str  # Comma-separated list
    notes: Optional[str]
    extraction_method: str


class VectorStore:
    """
    Vector database for storing and querying paper chunks using LanceDB.
    """

    def __init__(self, db_path: Path, vector_dimension: int = 3072):
        """
        Initialize the vector store.

        Args:
            db_path: Path to LanceDB database directory
            vector_dimension: Dimension of embedding vectors
        """
        self.db_path = db_path
        self.vector_dimension = vector_dimension

        # Create database directory if it doesn't exist
        db_path.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB
        self.db = lancedb.connect(str(db_path))

        # Define schema
        self.schema = self._create_schema()

        logger.info(f"VectorStore initialized at {db_path}")

    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for the chunks table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("paper_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.vector_dimension)),
            pa.field("bibtex_key", pa.string()),
            pa.field("bibtex_entry", pa.string()),
            pa.field("title", pa.string()),
            pa.field("authors", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("journal", pa.string()),
            pa.field("doi", pa.string()),
            pa.field("url", pa.string()),
            pa.field("pdf_path", pa.string()),
            pa.field("pdf_hash", pa.string()),
            pa.field("date_added", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("section_title", pa.string()),
            pa.field("section_hierarchy", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("element_type", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("notes", pa.string()),
            pa.field("extraction_method", pa.string()),
        ])

    def initialize_table(self) -> None:
        """Initialize the chunks table if it doesn't exist."""
        try:
            # Try to open existing table
            self.db.open_table("chunks")
            logger.info("Using existing 'chunks' table")
        except Exception:
            # Create new table with schema
            logger.info("Creating new 'chunks' table")
            # Create empty table with schema
            self.db.create_table("chunks", schema=self.schema)

    def add_paper(
        self,
        metadata: PaperMetadata,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        pdf_path: Path,
        pdf_hash: str,
        tags: Optional[list[str]] = None
    ) -> int:
        """
        Add a paper and its chunks to the vector store.

        Args:
            metadata: Paper metadata
            chunks: List of document chunks
            embeddings: List of embedding vectors (one per chunk)
            pdf_path: Path to the PDF file
            pdf_hash: SHA256 hash of the PDF
            tags: Optional list of tags

        Returns:
            Number of chunks added

        Raises:
            ValueError: If chunks and embeddings lengths don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)})"
            )

        logger.info(f"Adding paper '{metadata.bibtex_key}' with {len(chunks)} chunks")

        # Prepare records
        records = []
        paper_id = pdf_hash[:16]  # Use first 16 chars of hash as paper ID
        date_added = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""
        authors_str = ",".join(metadata.authors)

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{paper_id}_chunk_{chunk.chunk_index}"

            # Convert section hierarchy to JSON string
            import json
            section_hierarchy_str = json.dumps(chunk.section_hierarchy)

            record = ChunkRecord(
                id=chunk_id,
                paper_id=paper_id,
                text=chunk.text,
                vector=embedding,
                bibtex_key=metadata.bibtex_key,
                bibtex_entry=metadata.bibtex_entry,
                title=metadata.title,
                authors=authors_str,
                year=metadata.year,
                journal=metadata.journal,
                doi=metadata.doi,
                url=metadata.url,
                pdf_path=str(pdf_path),
                pdf_hash=pdf_hash,
                date_added=date_added,
                chunk_index=chunk.chunk_index,
                section_title=chunk.section_title,
                section_hierarchy=section_hierarchy_str,
                page_number=chunk.page_number,
                element_type=chunk.element_type,
                tags=tags_str,
                notes=None,
                extraction_method=metadata.extraction_method.value
            )

            records.append(asdict(record))

        # Add to database
        table = self.db.open_table("chunks")
        table.add(records)

        logger.info(f"Successfully added {len(records)} chunks for paper '{metadata.bibtex_key}'")
        return len(records)

    def search(
        self,
        query_vector: list[float],
        n_results: int = 5,
        filter_section: Optional[str] = None,
        min_year: Optional[int] = None,
        filter_tags: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_vector: Query embedding vector
            n_results: Number of results to return
            filter_section: Filter by section title (e.g., "Methods", "Results")
            min_year: Only include papers from this year onwards
            filter_tags: Filter by tags

        Returns:
            List of search results with metadata
        """
        table = self.db.open_table("chunks")

        # Build query
        query = table.search(query_vector).limit(n_results * 3)  # Get more for filtering

        # Apply filters
        if min_year:
            query = query.where(f"year >= {min_year}")

        # Execute search
        results = query.to_list()

        # Apply additional filters
        filtered_results = []
        for result in results:
            # Section filter
            if filter_section:
                if filter_section.lower() not in result['section_title'].lower():
                    continue

            # Tags filter
            if filter_tags:
                result_tags = set(result['tags'].split(',')) if result['tags'] else set()
                if not any(tag in result_tags for tag in filter_tags):
                    continue

            filtered_results.append(result)

            if len(filtered_results) >= n_results:
                break

        logger.info(f"Search returned {len(filtered_results)} results")
        return filtered_results

    def get_paper_by_key(self, bibtex_key: str) -> Optional[dict]:
        """
        Get paper metadata by BibTeX key.

        Args:
            bibtex_key: BibTeX citation key

        Returns:
            Paper metadata dict or None if not found
        """
        table = self.db.open_table("chunks")

        # Search for any chunk with this bibtex_key
        results = table.search().where(f"bibtex_key = '{bibtex_key}'").limit(1).to_list()

        if results:
            # Return paper-level metadata from first chunk
            return {
                'bibtex_key': results[0]['bibtex_key'],
                'bibtex_entry': results[0]['bibtex_entry'],
                'title': results[0]['title'],
                'authors': results[0]['authors'].split(','),
                'year': results[0]['year'],
                'journal': results[0]['journal'],
                'doi': results[0]['doi'],
                'url': results[0]['url'],
                'pdf_path': results[0]['pdf_path'],
                'pdf_hash': results[0]['pdf_hash'],
                'date_added': results[0]['date_added'],
                'extraction_method': results[0]['extraction_method']
            }

        return None

    def get_paper_chunks(self, paper_id: str) -> list[dict]:
        """
        Get all chunks for a specific paper.

        Args:
            paper_id: Paper identifier (hash prefix)

        Returns:
            List of chunk records
        """
        table = self.db.open_table("chunks")
        results = table.search().where(f"paper_id = '{paper_id}'").to_list()
        return sorted(results, key=lambda x: x['chunk_index'])

    def check_duplicate(self, pdf_hash: str) -> bool:
        """
        Check if a PDF has already been indexed.

        Args:
            pdf_hash: SHA256 hash of the PDF

        Returns:
            True if duplicate exists, False otherwise
        """
        table = self.db.open_table("chunks")
        results = table.search().where(f"pdf_hash = '{pdf_hash}'").limit(1).to_list()
        return len(results) > 0

    def get_all_bibtex_keys(self) -> set[str]:
        """
        Get all BibTeX keys currently in the database.

        Returns:
            Set of BibTeX keys
        """
        table = self.db.open_table("chunks")

        # Get unique bibtex_keys
        results = table.search().to_list()
        keys = {r['bibtex_key'] for r in results}

        return keys

    def update_paper_metadata(self, bibtex_key: str, updated_metadata: dict) -> int:
        """
        Update metadata for all chunks of a paper.

        Args:
            bibtex_key: BibTeX key of the paper to update
            updated_metadata: Dictionary of fields to update

        Returns:
            Number of chunks updated
        """
        # Note: LanceDB doesn't support in-place updates easily
        # This would require reading, modifying, and re-adding data
        # For MVP, we'll log this as a TODO
        logger.warning("Metadata update not fully implemented in MVP")
        return 0

    def delete_paper(self, bibtex_key: str) -> int:
        """
        Delete all chunks for a paper.

        Args:
            bibtex_key: BibTeX key of the paper to delete

        Returns:
            Number of chunks deleted
        """
        table = self.db.open_table("chunks")

        # Get count before deletion
        pre_count = len(table.search().where(f"bibtex_key = '{bibtex_key}'").to_list())

        # Delete (LanceDB delete syntax)
        table.delete(f"bibtex_key = '{bibtex_key}'")

        logger.info(f"Deleted {pre_count} chunks for paper '{bibtex_key}'")
        return pre_count

    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database stats
        """
        table = self.db.open_table("chunks")
        all_chunks = table.search().to_list()

        # Calculate statistics
        total_chunks = len(all_chunks)
        unique_papers = len(set(c['paper_id'] for c in all_chunks))
        years = [c['year'] for c in all_chunks if c['year']]
        avg_year = sum(years) / len(years) if years else 0

        # Papers by year
        from collections import Counter
        year_distribution = dict(Counter(years))

        return {
            'total_papers': unique_papers,
            'total_chunks': total_chunks,
            'average_year': int(avg_year) if avg_year else None,
            'year_distribution': year_distribution,
            'database_path': str(self.db_path)
        }

    def list_recent_papers(self, n: int = 10) -> list[dict]:
        """
        List recently added papers.

        Args:
            n: Number of papers to return

        Returns:
            List of recent papers with metadata
        """
        table = self.db.open_table("chunks")
        all_chunks = table.search().to_list()

        # Get unique papers sorted by date_added
        papers_by_key = {}
        for chunk in all_chunks:
            key = chunk['bibtex_key']
            if key not in papers_by_key:
                papers_by_key[key] = {
                    'bibtex_key': key,
                    'title': chunk['title'],
                    'authors': chunk['authors'].split(','),
                    'year': chunk['year'],
                    'date_added': chunk['date_added'],
                    'extraction_method': chunk['extraction_method']
                }

        # Sort by date_added
        recent_papers = sorted(
            papers_by_key.values(),
            key=lambda x: x['date_added'],
            reverse=True
        )

        return recent_papers[:n]
