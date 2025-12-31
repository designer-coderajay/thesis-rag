"""
QA Chain with longer, more detailed responses.
Thesis Research Assistant

FIXED: Citations now use real metadata from filenames.
LLM is instructed to ONLY use [Source N] format.
"""

import textwrap
import re
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Try to import ollama
try:
    import ollama
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class Citation:
    """Citation object with source_file and page_number attributes."""
    source_file: str
    page_number: str = "?"
    author: str = "Unknown"
    year: str = "?"
    title: str = "Unknown"

    @property
    def source(self):
        return self.source_file

    @property
    def page(self):
        return self.page_number


@dataclass
class QueryResponse:
    """Response object from RAG query."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sources: List[Citation] = field(default_factory=list)


def parse_filename_metadata(filename: str) -> dict:
    """
    Extract author, year, title from filename.

    Expected formats:
    - 019_elhage2022toy_2022.txt -> author=Elhage, year=2022, title=Toy Models
    - 018_wang2023interpretability_2023.pdf -> author=Wang, year=2023
    """
    metadata = {
        'author': 'Unknown',
        'year': '?',
        'title': 'Unknown',
        'raw_filename': filename
    }

    if not filename:
        return metadata

    # Remove path and extension
    basename = filename.split('/')[-1]
    name_part = basename.rsplit('.', 1)[0]  # Remove extension

    # Pattern 1: NNN_authorYEARtitle_YEAR
    match = re.match(r'^\d+_([a-z]+)(\d{4})([a-z]+)?', name_part, re.IGNORECASE)
    if match:
        author = match.group(1).capitalize()
        year = match.group(2)
        title_hint = match.group(3) or ''

        # Map known title hints to full titles
        title_map = {
            'toy': 'Toy Models of Superposition',
            'mathematical': 'A Mathematical Framework for Transformer Circuits',
            'interpretability': 'Interpretability in the Wild',
            'scaling': 'Scaling Monosemanticity',
            'induction': 'In-context Learning and Induction Heads',
            'superposition': 'Toy Models of Superposition',
            'attention': 'Attention Is All You Need',
            'transformer': 'Attention Is All You Need',
            'gpt': 'Language Models are Unsupervised Multitask Learners',
            'bert': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'rome': 'Locating and Editing Factual Associations in GPT',
            'acdc': 'Towards Automated Circuit Discovery',
            'ioi': 'Interpretability in the Wild (IOI)',
            'shap': 'A Unified Approach to Interpreting Model Predictions',
            'lime': 'Why Should I Trust You? Explaining the Predictions',
            'eraser': 'ERASER: A Benchmark of Rationale Extraction',
            'language': 'Language Models Can Explain Neurons',
            'unified': 'SHAP: A Unified Approach',
            'why': 'LIME: Why Should I Trust You?',
            'axiomatic': 'Integrated Gradients',
            'locating': 'ROME: Locating and Editing Facts',
            'grokking': 'Grokking: Generalization Beyond Overfitting',
            'tuned': 'Tuned Lens',
            'neural': 'Neural Machine Translation',
            'survey': 'A Survey of Transformers',
            'vision': 'Vision Transformers',
            'probing': 'Probing Classifiers',
            'emergent': 'Emergent Abilities of LLMs',
            'chain': 'Chain of Thought Prompting',
            'self': 'Self-Consistency',
            'constitutional': 'Constitutional AI',
            'towards': 'Towards Faithful Explanations',
            'rationalizing': 'Rationalizing Neural Predictions',
            'explain': 'Explaining Neural Networks',
            'faithful': 'Faithful Explanations',
            'measuring': 'Measuring Faithfulness',
            'human': 'Human-Centered XAI',
            'reliability': 'Reliability of Saliency Methods',
            'evaluating': 'Evaluating Explanations',
            'snli': 'e-SNLI Dataset',
            'sparse': 'Sparse Autoencoders',
            'representation': 'Representation Engineering',
            'millions': 'Scaling Monosemanticity: Millions of Features',
            'mib': 'MIB Benchmark',
            'sasc': 'SASC: Semantic Automaticity',
            'explaining': 'Explaining Neural Networks',
        }

        title = title_map.get(title_hint.lower(), title_hint.capitalize() if title_hint else 'Unknown')

        metadata['author'] = author
        metadata['year'] = year
        metadata['title'] = title
        return metadata

    # Pattern 2: Just try to find a year
    year_match = re.search(r'(\d{4})', name_part)
    if year_match:
        metadata['year'] = year_match.group(1)

    # Try to extract author (first alphabetic sequence)
    author_match = re.search(r'([a-zA-Z]+)', name_part)
    if author_match:
        metadata['author'] = author_match.group(1).capitalize()

    return metadata


class RetrievalQAChain:
    """Question-answering chain with RAG - Enhanced for detailed responses."""

    def __init__(self, vector_store, model: str = "llama3.2", num_chunks: int = 8, **kwargs):
        self.vector_store = vector_store
        self.model = model
        self.num_chunks = num_chunks

    def _extract_text_and_metadata(self, results: List) -> List[Tuple[str, dict]]:
        """
        Extract text and metadata from search results.
        Your format: (metadata_dict, score)
        """
        contexts = []

        for r in results:
            text = ""
            metadata = {}

            if isinstance(r, (tuple, list)) and len(r) >= 1:
                first = r[0]

                if isinstance(first, dict):
                    text = first.get('text', '')
                    source_file = first.get('source_file', 'Unknown')

                    # Parse real metadata from filename
                    parsed = parse_filename_metadata(source_file)

                    metadata = {
                        'source': source_file,
                        'page': first.get('page_number', '?'),
                        'source_type': first.get('source_type', ''),
                        'chunk_index': first.get('chunk_index', 0),
                        'author': parsed['author'],
                        'year': parsed['year'],
                        'title': parsed['title'],
                    }
                elif isinstance(first, str):
                    text = first
                elif hasattr(first, 'text'):
                    text = first.text
                    if hasattr(first, 'metadata'):
                        metadata = first.metadata if isinstance(first.metadata, dict) else {}

            if text:
                contexts.append((text, metadata))

        return contexts

    def _build_prompt(self, query: str, contexts: List[Tuple[str, dict]]) -> str:
        """Build prompt with retrieved contexts and REAL citation info."""

        # Build context with REAL metadata
        context_text = ""
        source_list = ""

        for i, (text, metadata) in enumerate(contexts, 1):
            author = metadata.get('author', 'Unknown')
            year = metadata.get('year', '?')
            title = metadata.get('title', 'Unknown')
            source_file = metadata.get('source', 'Unknown')

            context_text += f"\n[Source {i}]\n{text}\n"
            source_list += f"- [Source {i}]: {author} ({year}). {title}. File: {source_file}\n"

        # CRITICAL: Tell LLM to ONLY use [Source N] format
        prompt = f"""You are a research assistant helping write a Master's thesis on Explainable AI and Mechanistic Interpretability.

TASK: Answer the question using ONLY the provided sources.

CRITICAL CITATION RULES:
1. ONLY cite as [Source 1], [Source 2], etc.
2. DO NOT invent author names, paper titles, or journal names
3. DO NOT make up page numbers or publication details
4. If you want to mention an author, say "the authors of [Source N]"
5. At the end, I will show the real source details

AVAILABLE SOURCES:
{source_list}

SOURCE CONTENT:
{context_text}

INSTRUCTIONS:
1. Write 4-6 substantial paragraphs (5-7 sentences each)
2. Be DETAILED and THOROUGH - explain concepts fully
3. Cite as [Source N] for every claim - NO OTHER FORMAT
4. Include specific details, methods, findings, and implications
5. Use academic writing style
6. If discussing methods, explain HOW they work
7. DO NOT write a "References" section - I will add real citations

QUESTION: {query}

DETAILED RESPONSE (cite as [Source 1], [Source 2], etc. ONLY):"""

        return prompt

    def query(self, question: str, k: int = None) -> QueryResponse:
        """Query the RAG system."""
        if k is None:
            k = self.num_chunks

        results = self.vector_store.search(question, k=k)

        if not results:
            return QueryResponse(answer="No relevant sources found.", citations=[], sources=[])

        contexts = self._extract_text_and_metadata(results)

        if not contexts:
            return QueryResponse(answer="No relevant sources found.", citations=[], sources=[])

        prompt = self._build_prompt(question, contexts)

        if not OLLAMA_AVAILABLE:
            return QueryResponse(answer="Ollama not available. Run: pip install ollama", citations=[], sources=[])

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 2000,
                    "temperature": 0.3,
                }
            )
            answer = response['message']['content']

            # Append real source list to answer
            source_list = "\n\n---\nSOURCES USED:\n"
            seen = set()
            for i, (text, metadata) in enumerate(contexts, 1):
                source = metadata.get('source', 'Unknown')
                if source not in seen:
                    author = metadata.get('author', 'Unknown')
                    year = metadata.get('year', '?')
                    title = metadata.get('title', 'Unknown')
                    source_list += f"[Source {i}]: {author} ({year}). {title}\n"
                    seen.add(source)

            answer = answer + source_list

        except Exception as e:
            return QueryResponse(answer=f"Error: {e}", citations=[], sources=[])

        # Build citations from contexts
        citations = []
        seen = set()
        for text, metadata in contexts:
            source = metadata.get('source', 'Unknown')
            if source not in seen:
                citations.append(Citation(
                    source_file=source,
                    page_number=str(metadata.get('page', '?')),
                    author=metadata.get('author', 'Unknown'),
                    year=metadata.get('year', '?'),
                    title=metadata.get('title', 'Unknown'),
                ))
                seen.add(source)

        return QueryResponse(answer=answer, citations=citations, sources=citations)


def wrap_text(text: str, width: int = 50) -> str:
    """Wrap text for clean display."""
    paragraphs = text.split('\n\n')
    wrapped = []
    for p in paragraphs:
        p = ' '.join(p.split())
        if p:
            wrapped.append(textwrap.fill(p, width=width))
    return '\n\n'.join(wrapped)
