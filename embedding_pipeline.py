#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 base_url: Optional[str] = None):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
            base_url: Custom OpenAI-compatible endpoint (e.g. a classroom
                Vocareum proxy at https://openai.vocareum.com/v1). If not
                given, falls back to the OPENAI_BASE_URL or OPENAI_API_BASE
                environment variables, then the real OpenAI API.
        """
        # Resolve a custom base URL if one wasn't passed explicitly. The
        # openai SDK only auto-reads OPENAI_BASE_URL; OPENAI_API_BASE is an
        # older name some setup guides (including some classroom/Vocareum
        # instructions) still use, so we check both here.
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        # Initialize OpenAI client
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key, base_url=self.base_url)

        # Store configuration parameters
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client. The embedding function is attached to
        # the collection (not just used ad-hoc) so that any other process
        # that later opens this same collection with the same embedding
        # function (e.g. rag_client.py) will query in the same vector space.
        # CHROMA_OPENAI_API_KEY is the env var OpenAIEmbeddingFunction reads
        # the key from by default.
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_api_key
        self.embedding_function = OpenAIEmbeddingFunction(
            model_name=embedding_model,
            api_base=self.base_url
        )
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Connected to collection '{collection_name}' "
            f"({self.collection.count()} existing documents) at {chroma_persist_directory}"
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        text = text.strip()
        if not text:
            return []

        # Handle short texts that don't need chunking
        if len(text) <= self.chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = 0
            chunk_metadata['total_chunks'] = 1
            chunk_metadata['chunk_size'] = len(text)
            return [(text, chunk_metadata)]

        # Implement chunking logic with overlap. `end` is always capped at
        # chunk_size characters from `start`, so no chunk can ever exceed
        # chunk_size - the sentence-boundary search below can only move
        # `end` earlier, never later.
        raw_chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to break at sentence boundaries: search backwards from the
            # hard cutoff for the nearest sentence-ending punctuation, but
            # don't search further back than halfway through the chunk (to
            # avoid producing tiny chunks).
            if end < text_length:
                search_floor = start + max(int(self.chunk_size * 0.5), 1)
                best_boundary = -1
                for punct in ['. ', '.\n', '! ', '? ', '\n\n']:
                    idx = text.rfind(punct, search_floor, end)
                    if idx != -1:
                        candidate = idx + len(punct)
                        if candidate > best_boundary:
                            best_boundary = candidate
                if best_boundary != -1:
                    end = best_boundary

            chunk = text[start:end].strip()
            if chunk:
                raw_chunks.append(chunk)

            if end >= text_length:
                break

            # Apply chunk_overlap consistently between consecutive chunks
            next_start = end - self.chunk_overlap
            # Guard against a non-advancing loop if overlap >= chunk length
            start = next_start if next_start > start else end

        # Create metadata for each chunk
        result: List[Tuple[str, Dict[str, Any]]] = []
        for i, chunk in enumerate(raw_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(raw_chunks)
            chunk_metadata['chunk_size'] = len(chunk)
            result.append((chunk, chunk_metadata))

        return result
    
    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            # Query collection for document ID
            result = self.collection.get(ids=[doc_id])
            # Return True if exists, False otherwise
            return bool(result and result.get('ids'))
        except Exception as e:
            logger.debug(f"Error checking existence of {doc_id}: {e}")
            return False
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)
            
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Call OpenAI embeddings API
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            # Return embedding vector
            return response.data[0].embedding
        except Exception as e:
            # Add error handling
            logger.error(f"Error getting embedding from OpenAI: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed many texts in a single OpenAI API call.

        This matters a lot on rate-limited keys (e.g. classroom Vocareum
        proxy keys): calling get_embedding() once per chunk means one HTTP
        request per chunk, which quickly triggers 429 Too Many Requests on
        a corpus with thousands of chunks. Sending the whole batch as one
        `input` list cuts the number of requests by a factor of
        `len(texts)` (up to OpenAI's per-request array/token limits).

        Args:
            texts: List of chunk texts to embed, in order

        Returns:
            List of embedding vectors, in the same order as `texts`
        """
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            # OpenAI guarantees response.data is returned in the same
            # order as the input list, but sort by index defensively.
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as e:
            logger.error(f"Error getting batch embeddings from OpenAI: {e}")
            raise

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        # Create consistent ID format using mission, source, and chunk_index
        # Format: mission_source_chunk_0001
        mission = metadata.get('mission', 'unknown')
        source = metadata.get('source', file_path.stem)
        chunk_index = metadata.get('chunk_index', 0)
        return f"{mission}_{source}_chunk_{chunk_index:04d}"
    
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip', request_delay: float = 0.0) -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling
        
        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            request_delay: Seconds to sleep after each OpenAI embedding
                        call, as extra headroom against rate limits
            
        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}
        
        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        # Handle 'replace' mode up front: wipe any existing chunks that
        # belong to this file so we can re-add everything cleanly below.
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Replaced {len(existing_ids)} existing chunks for {file_path}")

        # Process documents in batches. Within each batch, texts that need
        # a *new* embedding (new adds and updates) are collected first and
        # embedded together in a single get_embeddings_batch() call, rather
        # than one OpenAI API call per chunk. This is the key fix for
        # rate-limited keys/proxies (e.g. classroom Vocareum keys) that
        # return 429 Too Many Requests when hit with one request per chunk
        # on a corpus with thousands of chunks.
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]

            # New documents to add
            new_ids: List[str] = []
            new_texts: List[str] = []
            new_metas: List[Dict[str, Any]] = []

            # Existing documents to update (update_mode == 'update')
            update_ids: List[str] = []
            update_texts: List[str] = []
            update_metas: List[Dict[str, Any]] = []

            for text, metadata in batch:
                # Generate document ID
                doc_id = self.generate_document_id(file_path, metadata)

                # Check if exists (always False right after a 'replace' wipe)
                exists = update_mode != 'replace' and self.check_document_exists(doc_id)

                if exists:
                    if update_mode == 'skip':
                        stats['skipped'] += 1
                    elif update_mode == 'update':
                        update_ids.append(doc_id)
                        update_texts.append(text)
                        update_metas.append(metadata)
                    continue

                new_ids.append(doc_id)
                new_texts.append(text)
                new_metas.append(metadata)

            # Embed and add all new documents in this batch with ONE API call
            if new_texts:
                try:
                    embeddings = self.get_embeddings_batch(new_texts)
                    if request_delay > 0:
                        time.sleep(request_delay)
                except Exception as e:
                    logger.error(f"Error embedding batch starting at {batch_start}: {e}")
                    stats['skipped'] += len(new_texts)
                    embeddings = None

                if embeddings is not None:
                    try:
                        self.collection.add(
                            ids=new_ids,
                            documents=new_texts,
                            metadatas=new_metas,
                            embeddings=embeddings
                        )
                        stats['added'] += len(new_ids)
                    except Exception as e:
                        logger.error(f"Error adding batch to collection: {e}")
                        stats['skipped'] += len(new_ids)

            # Embed and update all existing documents in this batch with
            # ONE API call (bypassing update_document()'s one-at-a-time
            # embedding calls, for the same rate-limit reason as above)
            if update_texts:
                try:
                    update_embeddings = self.get_embeddings_batch(update_texts)
                    if request_delay > 0:
                        time.sleep(request_delay)
                except Exception as e:
                    logger.error(f"Error embedding update batch starting at {batch_start}: {e}")
                    stats['skipped'] += len(update_texts)
                    update_embeddings = None

                if update_embeddings is not None:
                    try:
                        self.collection.update(
                            ids=update_ids,
                            documents=update_texts,
                            metadatas=update_metas,
                            embeddings=update_embeddings
                        )
                        stats['updated'] += len(update_ids)
                    except Exception as e:
                        logger.error(f"Error updating batch in collection: {e}")
                        stats['skipped'] += len(update_ids)

        # Return statistics
        return stats
    
    def process_all_text_data(self, base_path: str, update_mode: str = 'skip',
                               batch_size: int = 50, request_delay: float = 0.0) -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            batch_size: Number of chunks embedded per OpenAI API call. Lower
                        this if you're hitting 429 Too Many Requests on a
                        rate-limited key (e.g. a classroom Vocareum proxy).
            request_delay: Seconds to sleep between embedding batches, as
                        extra headroom against rate limits (default: none).
            
        Returns:
            Statistics about processed files
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }
        
        # Get files to process
        files_to_process = self.scan_text_files_only(base_path)

        # Loop through each file
        for file_path in files_to_process:
            try:
                logger.info(f"Processing file: {file_path}")

                # Process file: read, extract metadata, and chunk it
                documents = self.process_text_file(file_path)

                if not documents:
                    logger.warning(f"No content extracted from {file_path}, skipping")
                    continue

                mission = self.extract_mission_from_path(file_path)

                # Add to collection (embed + persist, honoring update_mode)
                file_stats = self.add_documents_to_collection(
                    documents, file_path, batch_size=batch_size,
                    update_mode=update_mode, request_delay=request_delay
                )

                # Update statistics

                stats['files_processed'] += 1
                stats['total_chunks'] += len(documents)
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']

                if mission not in stats['missions']:
                    stats['missions'][mission] = {
                        'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0
                    }
                mission_stats = stats['missions'][mission]
                mission_stats['files'] += 1
                mission_stats['chunks'] += len(documents)
                mission_stats['added'] += file_stats['added']
                mission_stats['updated'] += file_stats['updated']
                mission_stats['skipped'] += file_stats['skipped']

            except Exception as e:
                # Handle errors gracefully - one bad file shouldn't abort the run
                logger.error(f"Error processing file {file_path}: {e}")
                stats['errors'] += 1

        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        # Return collection name, document count, metadata
        try:
            return {
                'collection_name': self.collection_name,
                'document_count': self.collection.count(),
                'chroma_directory': self.chroma_persist_directory,
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 'unknown',
                'error': str(e),
            }
    
    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        # Perform test query and return results
        try:
            query_embedding = self.get_embedding(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {'error': str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get()
            
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            # Analyze metadata
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                
                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                
                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                
                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--base-url', default=None,
                        help='Custom OpenAI-compatible endpoint, e.g. https://openai.vocareum.com/v1 '
                             'for a classroom Vocareum key. Defaults to the OPENAI_BASE_URL or '
                             'OPENAI_API_BASE environment variable if set, otherwise the real OpenAI API.')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of chunks embedded per OpenAI API call. Lower this '
                             '(e.g. 10) if you hit 429 Too Many Requests on a rate-limited key.')
    parser.add_argument('--request-delay', type=float, default=0.0,
                        help='Seconds to sleep between embedding API calls, as extra headroom '
                             'against rate limits (e.g. 1.0 on a classroom/Vocareum key).')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        base_url=args.base_url
    )
    
    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    
    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return
    
    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    
    stats = pipeline.process_all_text_data(
        args.data_path, update_mode=args.update_mode,
        batch_size=args.batch_size, request_delay=args.request_delay
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
    
    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
    
    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
