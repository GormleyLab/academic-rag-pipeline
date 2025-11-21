"""
Document processing module using Docling for PDF parsing and chunking.
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument, PictureItem, TableItem

from src.utils import setup_logger
from src.tokenizer import OpenAITokenizerWrapper


logger = setup_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    chunk_index: int
    section_title: str
    section_hierarchy: list[str]
    page_number: Optional[int]
    element_type: str  # paragraph, table, figure, equation
    source_document: str


class DocumentProcessor:
    """
    Processes PDF documents using Docling for parsing and chunking.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 1000,
        chunk_overlap: int = 150,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the document processor.

        Args:
            max_chunk_tokens: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            embedding_model: Name of the embedding model for chunker
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        # Initialize Docling converter
        self.converter = DocumentConverter()

        # Initialize tokenizer wrapper for OpenAI
        tokenizer = OpenAITokenizerWrapper(
            model_name="cl100k_base",  # Encoding for text-embedding-3-large
            max_length=max_chunk_tokens
        )

        # Initialize chunker
        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_chunk_tokens
        )

        logger.info(
            f"DocumentProcessor initialized with max_tokens={max_chunk_tokens}, "
            f"overlap={chunk_overlap}, model={embedding_model}"
        )

    def __del__(self):
        """Cleanup resources when the processor is destroyed."""
        try:
            # Close the converter if it has cleanup methods
            if hasattr(self.converter, 'close'):
                self.converter.close()
            elif hasattr(self.converter, '__del__'):
                del self.converter
        except Exception as e:
            # Suppress errors during cleanup to avoid issues at interpreter shutdown
            pass

    def process_pdf(self, pdf_path: Path) -> tuple[DoclingDocument, list[DocumentChunk]]:
        """
        Process a PDF file into a structured document and chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (DoclingDocument, list of DocumentChunks)

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If processing fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Convert PDF to Docling document
            result = self.converter.convert(str(pdf_path))
            doc = result.document

            logger.info(f"Successfully parsed PDF: {pdf_path.name}")

            # Extract chunks from the document
            chunks = self._create_chunks(doc, pdf_path.name)

            logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")

            return doc, chunks

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise

    def _create_chunks(self, doc: DoclingDocument, source_name: str) -> list[DocumentChunk]:
        """
        Create chunks from a Docling document.

        Args:
            doc: Docling document
            source_name: Name of the source document

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Use Docling's chunking iterator
        chunk_iter = self.chunker.chunk(dl_doc=doc)

        for chunk_index, chunk in enumerate(chunk_iter):
            # Extract metadata from the chunk
            section_title = self._extract_section_title(chunk, doc)
            section_hierarchy = self._extract_section_hierarchy(chunk, doc)
            page_number = self._extract_page_number(chunk)
            element_type = self._determine_element_type(chunk)

            # Create DocumentChunk object
            doc_chunk = DocumentChunk(
                text=chunk.text,
                chunk_index=chunk_index,
                section_title=section_title,
                section_hierarchy=section_hierarchy,
                page_number=page_number,
                element_type=element_type,
                source_document=source_name
            )

            chunks.append(doc_chunk)

        return chunks

    def _extract_section_title(self, chunk, doc: DoclingDocument) -> str:
        """Extract the section title for a chunk."""
        # Try to get section information from chunk metadata
        if hasattr(chunk, 'meta') and chunk.meta:
            if 'headings' in chunk.meta and chunk.meta['headings']:
                return chunk.meta['headings'][-1]

        return "Unknown Section"

    def _extract_section_hierarchy(self, chunk, doc: DoclingDocument) -> list[str]:
        """Extract the full section hierarchy for a chunk."""
        if hasattr(chunk, 'meta') and chunk.meta:
            if 'headings' in chunk.meta and chunk.meta['headings']:
                return chunk.meta['headings']

        return []

    def _extract_page_number(self, chunk) -> Optional[int]:
        """Extract page number from chunk metadata."""
        if hasattr(chunk, 'meta') and chunk.meta:
            if 'page' in chunk.meta:
                return chunk.meta['page']

        return None

    def _determine_element_type(self, chunk) -> str:
        """Determine the type of element (paragraph, table, figure, etc.)."""
        # Check chunk metadata or content for element type
        if hasattr(chunk, 'meta') and chunk.meta:
            if 'doc_items' in chunk.meta:
                doc_items = chunk.meta['doc_items']
                if any(isinstance(item, TableItem) for item in doc_items):
                    return "table"
                if any(isinstance(item, PictureItem) for item in doc_items):
                    return "figure"

        # Check if text contains equation patterns
        if any(marker in chunk.text for marker in ['\\(', '\\[', '$$', r'\begin{equation}']):
            return "equation"

        return "paragraph"

    def export_to_markdown(self, doc: DoclingDocument, output_path: Path) -> None:
        """
        Export a Docling document to Markdown format.

        Args:
            doc: Docling document
            output_path: Path to save the Markdown file
        """
        try:
            markdown_content = doc.export_to_markdown()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"Exported document to Markdown: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to Markdown: {e}")
            raise

    def extract_text_from_first_pages(self, pdf_path: Path, num_pages: int = 5) -> str:
        """
        Extract text from the first N pages of a PDF.
        Useful for DOI extraction and metadata gathering.

        Args:
            pdf_path: Path to the PDF file
            num_pages: Number of pages to extract

        Returns:
            Extracted text from first pages
        """
        try:
            result = self.converter.convert(str(pdf_path))
            doc = result.document

            # Get text from first few pages
            text_parts = []
            for item in doc.iterate_items():
                if hasattr(item, 'page') and item.page and item.page <= num_pages:
                    if hasattr(item, 'text') and item.text:
                        text_parts.append(item.text)

            return " ".join(text_parts)

        except Exception as e:
            logger.error(f"Failed to extract text from first pages: {e}")
            return ""
