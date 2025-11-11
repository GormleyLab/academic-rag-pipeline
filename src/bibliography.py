"""
Bibliography management module for BibTeX operations.
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from pybtex.database import parse_string as parse_bibtex, BibliographyData, Entry

from src.utils import setup_logger, validate_bibtex_entry


logger = setup_logger(__name__)


@dataclass
class BibliographyEntry:
    """Represents a BibTeX bibliography entry."""
    bibtex_key: str
    bibtex_entry: str
    title: str
    authors: list[str]
    year: int
    is_valid: bool = True


class BibliographyManager:
    """
    Manages BibTeX bibliography operations.
    """

    def __init__(self):
        """Initialize the bibliography manager."""
        logger.info("BibliographyManager initialized")

    def generate_bibliography_file(
        self,
        entries: list[BibliographyEntry],
        output_path: Path,
        include_abstracts: bool = False
    ) -> dict:
        """
        Generate a .bib file from bibliography entries.

        Args:
            entries: List of BibliographyEntry objects
            output_path: Path where to save the .bib file
            include_abstracts: Whether to include abstract fields

        Returns:
            Dictionary with generation results (included, missing, errors)
        """
        logger.info(f"Generating bibliography file: {output_path}")

        included = []
        missing = []
        errors = []

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                try:
                    # Validate entry
                    if not validate_bibtex_entry(entry.bibtex_entry):
                        logger.warning(
                            f"Invalid BibTeX entry for {entry.bibtex_key}, attempting to fix"
                        )
                        errors.append(entry.bibtex_key)
                        continue

                    # Write entry to file
                    bibtex_text = entry.bibtex_entry

                    # Remove abstract if not requested
                    if not include_abstracts:
                        bibtex_text = self._remove_abstract_field(bibtex_text)

                    f.write(bibtex_text)
                    f.write('\n\n')

                    included.append(entry.bibtex_key)
                    logger.debug(f"Added entry: {entry.bibtex_key}")

                except Exception as e:
                    logger.error(f"Error processing entry {entry.bibtex_key}: {e}")
                    errors.append(entry.bibtex_key)

        result = {
            'output_path': str(output_path),
            'total_requested': len(entries),
            'included': included,
            'missing': missing,
            'errors': errors,
            'success_count': len(included)
        }

        logger.info(
            f"Bibliography generated: {len(included)} entries written to {output_path}"
        )

        return result

    def parse_bibtex_file(self, bib_path: Path) -> list[BibliographyEntry]:
        """
        Parse an existing .bib file.

        Args:
            bib_path: Path to .bib file

        Returns:
            List of BibliographyEntry objects

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not bib_path.exists():
            raise FileNotFoundError(f"BibTeX file not found: {bib_path}")

        logger.info(f"Parsing BibTeX file: {bib_path}")

        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_content = f.read()

        try:
            bib_data = parse_bibtex(bib_content, 'bibtex')

            entries = []
            for key, entry in bib_data.entries.items():
                # Extract authors
                authors = []
                if 'author' in entry.persons:
                    authors = [str(person) for person in entry.persons['author']]

                # Extract year
                year = int(entry.fields.get('year', 0))

                # Reconstruct BibTeX entry
                bibtex_entry = self._entry_to_string(key, entry)

                bib_entry = BibliographyEntry(
                    bibtex_key=key,
                    bibtex_entry=bibtex_entry,
                    title=entry.fields.get('title', ''),
                    authors=authors,
                    year=year,
                    is_valid=True
                )

                entries.append(bib_entry)

            logger.info(f"Parsed {len(entries)} entries from {bib_path}")
            return entries

        except Exception as e:
            logger.error(f"Failed to parse BibTeX file: {e}")
            raise

    def merge_bibliographies(
        self,
        bib_files: list[Path],
        output_path: Path,
        deduplicate: bool = True
    ) -> dict:
        """
        Merge multiple .bib files into one.

        Args:
            bib_files: List of paths to .bib files
            output_path: Path for merged output file
            deduplicate: Whether to remove duplicate entries

        Returns:
            Dictionary with merge results
        """
        logger.info(f"Merging {len(bib_files)} bibliography files")

        all_entries = []
        seen_keys = set()

        for bib_file in bib_files:
            try:
                entries = self.parse_bibtex_file(bib_file)

                for entry in entries:
                    if deduplicate:
                        if entry.bibtex_key in seen_keys:
                            logger.debug(
                                f"Skipping duplicate entry: {entry.bibtex_key}"
                            )
                            continue
                        seen_keys.add(entry.bibtex_key)

                    all_entries.append(entry)

            except Exception as e:
                logger.error(f"Failed to process {bib_file}: {e}")

        # Generate merged file
        result = self.generate_bibliography_file(all_entries, output_path)
        result['source_files'] = [str(f) for f in bib_files]

        logger.info(f"Merged bibliography created with {len(all_entries)} entries")
        return result

    def validate_bibliography(self, bib_path: Path) -> dict:
        """
        Validate a .bib file and report issues.

        Args:
            bib_path: Path to .bib file

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating bibliography: {bib_path}")

        entries = self.parse_bibtex_file(bib_path)

        valid = []
        invalid = []
        warnings = []

        for entry in entries:
            issues = []

            # Check for required fields
            if not entry.title:
                issues.append("Missing title")

            if not entry.authors or entry.authors == ["Unknown"]:
                issues.append("Missing or unknown authors")

            if not entry.year or entry.year == 0:
                issues.append("Missing or invalid year")

            # Check BibTeX validity
            if not validate_bibtex_entry(entry.bibtex_entry):
                issues.append("Invalid BibTeX format")

            if issues:
                invalid.append({
                    'key': entry.bibtex_key,
                    'issues': issues
                })
            else:
                valid.append(entry.bibtex_key)

        result = {
            'total_entries': len(entries),
            'valid_entries': len(valid),
            'invalid_entries': len(invalid),
            'valid_keys': valid,
            'invalid_details': invalid,
            'warnings': warnings
        }

        logger.info(
            f"Validation complete: {len(valid)}/{len(entries)} entries valid"
        )

        return result

    def format_citation(
        self,
        entry: BibliographyEntry,
        style: str = "inline"
    ) -> str:
        """
        Format a citation in different styles.

        Args:
            entry: BibliographyEntry to format
            style: Citation style ("inline", "apa", "mla")

        Returns:
            Formatted citation string
        """
        if style == "inline":
            # Format: [Smith2024]
            return f"[{entry.bibtex_key}]"

        elif style == "apa":
            # Format: Smith, J., & Jones, A. (2024). Title. Journal.
            authors_str = self._format_authors_apa(entry.authors)
            return f"{authors_str} ({entry.year}). {entry.title}."

        elif style == "mla":
            # Format: Smith, John, and Alice Jones. "Title." Year.
            authors_str = self._format_authors_mla(entry.authors)
            return f"{authors_str}. \"{entry.title}.\" {entry.year}."

        else:
            return f"[{entry.bibtex_key}]"

    def _remove_abstract_field(self, bibtex_entry: str) -> str:
        """Remove abstract field from BibTeX entry."""
        import re
        # Remove abstract = {...} or abstract = "..."
        pattern = r',?\s*abstract\s*=\s*[{"](.*?)["}]'
        return re.sub(pattern, '', bibtex_entry, flags=re.DOTALL)

    def _entry_to_string(self, key: str, entry: Entry) -> str:
        """Convert pybtex Entry to BibTeX string."""
        lines = [f"@{entry.type}{{{key},"]

        # Add fields
        for field, value in entry.fields.items():
            lines.append(f"  {field} = {{{value}}},")

        # Add persons (authors, editors, etc.)
        for role, persons in entry.persons.items():
            persons_str = " and ".join(str(person) for person in persons)
            lines.append(f"  {role} = {{{persons_str}}},")

        lines.append("}")

        return "\n".join(lines)

    def _format_authors_apa(self, authors: list[str]) -> str:
        """Format authors in APA style."""
        if not authors:
            return "Unknown"

        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]}, & {authors[1]}"
        else:
            # First author et al. for 3+ authors
            return f"{authors[0]}, et al."

    def _format_authors_mla(self, authors: list[str]) -> str:
        """Format authors in MLA style."""
        if not authors:
            return "Unknown"

        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]}, and {authors[1]}"
        else:
            return f"{authors[0]}, et al."

    def extract_keys_from_text(self, text: str) -> list[str]:
        """
        Extract citation keys from text (e.g., from LaTeX document).

        Args:
            text: Text containing citations

        Returns:
            List of unique citation keys found
        """
        import re

        # Common LaTeX citation patterns
        patterns = [
            r'\\cite\{([^}]+)\}',  # \cite{key}
            r'\\citep\{([^}]+)\}',  # \citep{key}
            r'\\citet\{([^}]+)\}',  # \citet{key}
        ]

        keys = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Handle multiple keys: \cite{key1,key2,key3}
                split_keys = [k.strip() for k in match.split(',')]
                keys.update(split_keys)

        return sorted(list(keys))
