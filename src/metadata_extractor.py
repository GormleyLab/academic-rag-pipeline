"""
Metadata extraction module for citation information from PDFs and APIs.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import requests
import fitz  # PyMuPDF
from pybtex.database import parse_string as parse_bibtex

try:
    import pdf2bib
    PDF2BIB_AVAILABLE = True
except ImportError:
    PDF2BIB_AVAILABLE = False

from src.utils import (
    setup_logger,
    extract_doi_from_text,
    extract_arxiv_id_from_text,
    extract_pubmed_id_from_text,
    generate_bibtex_key
)


logger = setup_logger(__name__)


class ExtractionMethod(Enum):
    """Enumeration of metadata extraction methods."""
    PDF2BIB = "pdf2bib"
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    PDF_METADATA = "pdf_metadata"
    PARSED = "parsed"
    MANUAL = "manual"


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    title: str
    authors: list[str]
    year: int
    bibtex_key: str
    bibtex_entry: str
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    publisher: Optional[str] = None
    extraction_method: ExtractionMethod = ExtractionMethod.PARSED


class MetadataExtractor:
    """
    Extracts citation metadata from PDFs using multiple strategies.
    """

    def __init__(
        self,
        crossref_email: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the metadata extractor.

        Args:
            crossref_email: Email for polite CrossRef API usage
            max_retries: Maximum number of API retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.crossref_email = crossref_email
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.session = requests.Session()
        if crossref_email:
            self.session.headers.update({
                'User-Agent': f'AcademicRAG/1.0 (mailto:{crossref_email})'
            })

        # Configure pdf2bib if available
        if PDF2BIB_AVAILABLE:
            pdf2bib.config.set('verbose', False)  # Set to True for debugging
            logger.info("MetadataExtractor initialized with pdf2bib support")
        else:
            logger.warning("pdf2bib not available - install with 'pip install pdf2bib'")
            logger.info("MetadataExtractor initialized without pdf2bib support")

    def extract_metadata(
        self,
        pdf_path: Path,
        first_pages_text: Optional[str] = None,
        existing_keys: Optional[set[str]] = None
    ) -> PaperMetadata:
        """
        Extract metadata from a PDF using multiple strategies.

        Args:
            pdf_path: Path to the PDF file
            first_pages_text: Optional pre-extracted text from first pages
            existing_keys: Set of existing BibTeX keys for collision detection

        Returns:
            PaperMetadata object

        Raises:
            Exception: If all extraction methods fail
        """
        if existing_keys is None:
            existing_keys = set()

        logger.info(f"Extracting metadata from: {pdf_path.name}")

        # Strategy 1: Try pdf2bib (extracts DOI/arXiv directly from PDF and fetches metadata)
        if PDF2BIB_AVAILABLE:
            metadata = self._get_metadata_from_pdf2bib(pdf_path, existing_keys)
            if metadata:
                return metadata

        # Strategy 2: Try DOI + CrossRef (from text extraction)
        if first_pages_text:
            doi = extract_doi_from_text(first_pages_text)
            if doi:
                logger.info(f"Found DOI: {doi}")
                metadata = self._get_metadata_from_crossref(doi, existing_keys)
                if metadata:
                    return metadata

        # Strategy 3: Try arXiv
        if first_pages_text:
            arxiv_id = extract_arxiv_id_from_text(first_pages_text)
            if arxiv_id:
                logger.info(f"Found arXiv ID: {arxiv_id}")
                metadata = self._get_metadata_from_arxiv(arxiv_id, existing_keys)
                if metadata:
                    return metadata

        # Strategy 4: Try PubMed
        if first_pages_text:
            pmid = extract_pubmed_id_from_text(first_pages_text)
            if pmid:
                logger.info(f"Found PMID: {pmid}")
                metadata = self._get_metadata_from_pubmed(pmid, existing_keys)
                if metadata:
                    return metadata

        # Strategy 5: Try PDF metadata
        metadata = self._extract_from_pdf_metadata(pdf_path, existing_keys)
        if metadata:
            return metadata

        # Strategy 6: Parse document text
        logger.warning(f"Using document parsing for {pdf_path.name} (may be incomplete)")
        return self._parse_metadata_from_text(first_pages_text or "", pdf_path, existing_keys)

    def _get_metadata_from_pdf2bib(
        self,
        pdf_path: Path,
        existing_keys: set[str]
    ) -> Optional[PaperMetadata]:
        """
        Extract metadata using pdf2bib library.

        pdf2bib extracts identifiers (DOI, arXiv, etc.) directly from the PDF
        and fetches metadata from appropriate sources.

        Args:
            pdf_path: Path to PDF file
            existing_keys: Set of existing BibTeX keys

        Returns:
            PaperMetadata if successful, None otherwise
        """
        try:
            logger.info(f"Attempting pdf2bib extraction for: {pdf_path.name}")

            # Call pdf2bib to extract identifier and fetch metadata
            result = pdf2bib.pdf2bib(str(pdf_path))

            # Handle result format (can be dict with file path as key or direct result)
            if isinstance(result, dict):
                # Check if this is a direct result (has 'identifier' key)
                # or a dict of results (file paths as keys)
                if 'identifier' in result:
                    pdf_result = result
                elif str(pdf_path) in result:
                    pdf_result = result[str(pdf_path)]
                else:
                    logger.warning(f"pdf2bib returned unexpected result format for {pdf_path.name}")
                    return None
            else:
                logger.warning(f"pdf2bib returned unexpected type: {type(result)}")
                return None

            # Check if identifier was found
            identifier = pdf_result.get('identifier')
            if not identifier:
                logger.info(f"pdf2bib could not find identifier in {pdf_path.name}")
                return None

            identifier_type = pdf_result.get('identifier_type', 'unknown')
            logger.info(f"pdf2bib found {identifier_type}: {identifier}")

            # Get BibTeX from the result
            bibtex_entry = pdf_result.get('bibtex')
            if not bibtex_entry:
                logger.warning(f"pdf2bib found identifier but no BibTeX data for {pdf_path.name}")
                return None

            # Parse BibTeX to extract fields
            parsed = self._parse_bibtex_entry(bibtex_entry)
            if not parsed:
                logger.warning(f"Failed to parse BibTeX from pdf2bib for {pdf_path.name}")
                return None

            # Generate proper BibTeX key
            bibtex_key = generate_bibtex_key(
                parsed['authors'],
                parsed['year'],
                existing_keys
            )

            # Replace the key in the BibTeX entry
            bibtex_entry = self._replace_bibtex_key(bibtex_entry, bibtex_key)

            # Determine DOI and URL
            doi = parsed.get('doi') or (identifier if identifier_type == 'doi' else None)

            # Construct URL based on identifier type
            if doi:
                url_link = f"https://doi.org/{doi}"
            elif identifier_type == 'arxiv':
                url_link = f"https://arxiv.org/abs/{identifier}"
            else:
                url_link = pdf_result.get('url')

            return PaperMetadata(
                title=parsed['title'],
                authors=parsed['authors'],
                year=parsed['year'],
                bibtex_key=bibtex_key,
                bibtex_entry=bibtex_entry,
                journal=parsed.get('journal'),
                volume=parsed.get('volume'),
                pages=parsed.get('pages'),
                doi=doi,
                url=url_link,
                publisher=parsed.get('publisher'),
                extraction_method=ExtractionMethod.PDF2BIB
            )

        except Exception as e:
            logger.warning(f"pdf2bib extraction failed for {pdf_path.name}: {e}")
            return None

    def _get_metadata_from_crossref(
        self,
        doi: str,
        existing_keys: set[str]
    ) -> Optional[PaperMetadata]:
        """
        Fetch metadata from CrossRef API using DOI.

        Args:
            doi: Digital Object Identifier
            existing_keys: Set of existing BibTeX keys

        Returns:
            PaperMetadata if successful, None otherwise
        """
        url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=10)

                if response.status_code == 200:
                    bibtex_entry = response.text

                    # Parse BibTeX to extract fields
                    parsed = self._parse_bibtex_entry(bibtex_entry)

                    if parsed:
                        # Generate proper BibTeX key
                        bibtex_key = generate_bibtex_key(
                            parsed['authors'],
                            parsed['year'],
                            existing_keys
                        )

                        # Replace the key in the BibTeX entry
                        bibtex_entry = self._replace_bibtex_key(bibtex_entry, bibtex_key)

                        # Construct URL from DOI
                        url_link = f"https://doi.org/{doi}"

                        return PaperMetadata(
                            title=parsed['title'],
                            authors=parsed['authors'],
                            year=parsed['year'],
                            bibtex_key=bibtex_key,
                            bibtex_entry=bibtex_entry,
                            journal=parsed.get('journal'),
                            volume=parsed.get('volume'),
                            pages=parsed.get('pages'),
                            doi=doi,
                            url=url_link,
                            publisher=parsed.get('publisher'),
                            extraction_method=ExtractionMethod.CROSSREF
                        )

                elif response.status_code == 404:
                    logger.warning(f"DOI not found in CrossRef: {doi}")
                    return None

                else:
                    logger.warning(f"CrossRef API returned status {response.status_code}")

            except requests.RequestException as e:
                logger.warning(f"CrossRef API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        return None

    def _get_metadata_from_arxiv(
        self,
        arxiv_id: str,
        existing_keys: set[str]
    ) -> Optional[PaperMetadata]:
        """
        Fetch metadata from arXiv API.

        Args:
            arxiv_id: arXiv identifier
            existing_keys: Set of existing BibTeX keys

        Returns:
            PaperMetadata if successful, None otherwise
        """
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                # Parse XML response (simplified - would need proper XML parsing)
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)

                # Extract metadata from XML
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                entry = root.find('atom:entry', ns)

                if entry is not None:
                    title = entry.find('atom:title', ns).text.strip()
                    authors = [
                        author.find('atom:name', ns).text
                        for author in entry.findall('atom:author', ns)
                    ]
                    published = entry.find('atom:published', ns).text
                    year = int(published[:4])
                    abstract = entry.find('atom:summary', ns).text.strip()

                    bibtex_key = generate_bibtex_key(authors, year, existing_keys)

                    # Create BibTeX entry
                    bibtex_entry = self._create_bibtex_entry(
                        entry_type="article",
                        key=bibtex_key,
                        title=title,
                        authors=authors,
                        year=year,
                        journal="arXiv preprint",
                        url=f"https://arxiv.org/abs/{arxiv_id}"
                    )

                    return PaperMetadata(
                        title=title,
                        authors=authors,
                        year=year,
                        bibtex_key=bibtex_key,
                        bibtex_entry=bibtex_entry,
                        journal="arXiv preprint",
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        abstract=abstract,
                        extraction_method=ExtractionMethod.ARXIV
                    )

        except Exception as e:
            logger.warning(f"Failed to fetch from arXiv: {e}")

        return None

    def _get_metadata_from_pubmed(
        self,
        pmid: str,
        existing_keys: set[str]
    ) -> Optional[PaperMetadata]:
        """
        Fetch metadata from PubMed.

        Args:
            pmid: PubMed ID
            existing_keys: Set of existing BibTeX keys

        Returns:
            PaperMetadata if successful, None otherwise
        """
        # This is a placeholder - full implementation would use NCBI E-utilities API
        logger.info(f"PubMed API integration not fully implemented yet for PMID: {pmid}")
        return None

    def _extract_from_pdf_metadata(
        self,
        pdf_path: Path,
        existing_keys: set[str]
    ) -> Optional[PaperMetadata]:
        """
        Extract metadata from PDF properties.

        Args:
            pdf_path: Path to PDF file
            existing_keys: Set of existing BibTeX keys

        Returns:
            PaperMetadata if successful, None otherwise
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata

            if metadata:
                title = metadata.get('title', '')
                author = metadata.get('author', '')
                subject = metadata.get('subject', '')

                # Try to extract year from creation/modification date
                year = None
                if metadata.get('creationDate'):
                    try:
                        year = int(metadata['creationDate'][2:6])  # Format: D:YYYYMMDD...
                    except (ValueError, IndexError):
                        pass

                if title and author and year:
                    authors = [author] if author else []
                    bibtex_key = generate_bibtex_key(authors, year, existing_keys)

                    bibtex_entry = self._create_bibtex_entry(
                        entry_type="article",
                        key=bibtex_key,
                        title=title,
                        authors=authors,
                        year=year
                    )

                    return PaperMetadata(
                        title=title,
                        authors=authors,
                        year=year,
                        bibtex_key=bibtex_key,
                        bibtex_entry=bibtex_entry,
                        extraction_method=ExtractionMethod.PDF_METADATA
                    )

        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")

        return None

    def _parse_metadata_from_text(
        self,
        text: str,
        pdf_path: Path,
        existing_keys: set[str]
    ) -> PaperMetadata:
        """
        Parse metadata from document text as fallback.

        Args:
            text: Document text
            pdf_path: Path to PDF (for filename fallback)
            existing_keys: Set of existing BibTeX keys

        Returns:
            PaperMetadata (may be incomplete)
        """
        # Simple heuristics - extract title (usually largest text on first page)
        lines = text.split('\n')
        title = lines[0] if lines else pdf_path.stem

        # Try to find year in text
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        year = int(year_match.group(0)) if year_match else 2024

        authors = ["Unknown"]
        bibtex_key = generate_bibtex_key(authors, year, existing_keys)

        bibtex_entry = self._create_bibtex_entry(
            entry_type="article",
            key=bibtex_key,
            title=title,
            authors=authors,
            year=year
        )

        return PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            bibtex_key=bibtex_key,
            bibtex_entry=bibtex_entry,
            extraction_method=ExtractionMethod.PARSED
        )

    def _parse_bibtex_entry(self, bibtex_str: str) -> Optional[dict]:
        """Parse a BibTeX entry string and extract fields."""
        try:
            bib_data = parse_bibtex(bibtex_str, 'bibtex')
            if not bib_data.entries:
                return None

            entry = list(bib_data.entries.values())[0]

            # Extract authors
            authors = []
            if 'author' in entry.persons:
                authors = [
                    str(person)
                    for person in entry.persons['author']
                ]

            # Extract fields
            fields = entry.fields

            return {
                'title': fields.get('title', ''),
                'authors': authors,
                'year': int(fields.get('year', 2024)),
                'journal': fields.get('journal'),
                'volume': fields.get('volume'),
                'pages': fields.get('pages'),
                'publisher': fields.get('publisher'),
                'doi': fields.get('doi')
            }

        except Exception as e:
            logger.warning(f"Failed to parse BibTeX: {e}")
            return None

    def _create_bibtex_entry(
        self,
        entry_type: str,
        key: str,
        title: str,
        authors: list[str],
        year: int,
        **kwargs
    ) -> str:
        """Create a BibTeX entry string."""
        # Format authors for BibTeX
        author_str = " and ".join(authors) if authors else "Unknown"

        entry_lines = [
            f"@{entry_type}{{{key},",
            f"  title = {{{title}}}," if title else "  title = {Untitled},",
            f"  author = {{{author_str}}}," if author_str else "  author = {Unknown},",
            f"  year = {{{year}}},"
        ]

        # Add optional fields
        for field, value in kwargs.items():
            if value:
                entry_lines.append(f"  {field} = {{{value}}},")

        entry_lines.append("}")

        return "\n".join(entry_lines)

    def _replace_bibtex_key(self, bibtex_entry: str, new_key: str) -> str:
        """Replace the citation key in a BibTeX entry."""
        import re
        # Match @type{oldkey,
        pattern = r'(@\w+\{)[^,]+,'
        replacement = r'\g<1>' + new_key + ','
        return re.sub(pattern, replacement, bibtex_entry, count=1)
