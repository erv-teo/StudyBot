#!/usr/bin/env python3
"""CLI utility for chunk management in StudyBot RAG pipeline."""

import argparse
import json
import sys
from typing import Optional
from core.vectorstore.standalone_chunk_manager import StandaloneChunkManager
from core.config.settings import get_config


def format_chunk_info(chunk: dict, show_full_content: bool = False) -> str:
    """Format chunk information for display."""
    chunk_id = chunk.get('chunk_id', 'Unknown')
    source = chunk.get('metadata', {}).get('source', 'Unknown')
    section = chunk.get('metadata', {}).get('section', 'Unknown')
    content = chunk.get('content', '')
    preview = chunk.get('content_preview', content[:100] + "..." if len(content) > 100 else content)
    
    result = f"""
Chunk ID: {chunk_id}
Source: {source}
Section: {section}
Size: {len(content)} characters
Content: {content if show_full_content else preview}
"""
    
    if 'relevance_type' in chunk:
        result += f"Relevance: {chunk['relevance_type']}\n"
    
    return result


def list_chunks_command(chunk_manager: StandaloneChunkManager, args):
    """List chunks with optional filtering."""
    kwargs = {}
    if args.source:
        kwargs['source_filter'] = args.source
    if args.doc_id:
        kwargs['doc_id_filter'] = args.doc_id
    if args.section:
        kwargs['section_filter'] = args.section
    if args.limit:
        kwargs['limit'] = args.limit
    
    chunks = chunk_manager.list_chunks(**kwargs)
    
    print(f"Found {len(chunks)} chunks:")
    print("=" * 50)
    
    for chunk in chunks:
        print(format_chunk_info(chunk))
        print("-" * 30)


def search_chunks_command(chunk_manager: StandaloneChunkManager, args):
    """Search chunks by content."""
    kwargs = {
        'semantic': args.semantic,
        'limit': args.limit or 10
    }
    
    chunks = chunk_manager.search_chunks(args.query, **kwargs)
    
    print(f"Found {len(chunks)} matching chunks for query: '{args.query}'")
    print("=" * 50)
    
    for chunk in chunks:
        print(format_chunk_info(chunk))
        print("-" * 30)


def get_chunk_command(chunk_manager: StandaloneChunkManager, args):
    """Get detailed information about a specific chunk."""
    chunk = chunk_manager.get_chunk_details(args.chunk_id)
    
    if chunk:
        print("Chunk Details:")
        print("=" * 50)
        print(format_chunk_info(chunk, show_full_content=True))
        
        # Show full metadata
        print("\nFull Metadata:")
        metadata = chunk.get('metadata', {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print(f"Chunk '{args.chunk_id}' not found.")


def edit_chunk_command(chunk_manager: StandaloneChunkManager, args):
    """Edit a chunk's content."""
    # Get current chunk
    current_chunk = chunk_manager.get_chunk_details(args.chunk_id)
    if not current_chunk:
        print(f"Chunk '{args.chunk_id}' not found.")
        return
    
    print("Current content:")
    print("-" * 30)
    print(current_chunk.get('content', ''))
    print("-" * 30)
    
    if args.content:
        new_content = args.content
    else:
        print("\nEnter new content (press Ctrl+D when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            new_content = '\n'.join(lines)
    
    if new_content.strip():
        success = chunk_manager.edit_chunk(args.chunk_id, new_content.strip())
        if success:
            print(f"Successfully updated chunk '{args.chunk_id}'")
        else:
            print(f"Failed to update chunk '{args.chunk_id}'")
    else:
        print("No content provided. Chunk not updated.")


def delete_chunk_command(chunk_manager: StandaloneChunkManager, args):
    """Delete a chunk."""
    if not args.confirm:
        response = input(f"Are you sure you want to delete chunk '{args.chunk_id}'? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled.")
            return
    
    success = chunk_manager.delete_chunk(args.chunk_id)
    if success:
        print(f"Successfully deleted chunk '{args.chunk_id}'")
    else:
        print(f"Failed to delete chunk '{args.chunk_id}'")


def stats_command(chunk_manager: StandaloneChunkManager, args):
    """Show document statistics."""
    if args.source:
        stats = chunk_manager.get_document_stats(args.source)
        print(f"Statistics for document: {args.source}")
        print("=" * 50)
        print(json.dumps(stats, indent=2))
    else:
        # Show overall stats
        chunks = chunk_manager.list_chunks(limit=10000)
        sources = set(chunk.get('metadata', {}).get('source', 'Unknown') for chunk in chunks)
        
        print("Overall Statistics:")
        print("=" * 50)
        print(f"Total chunks: {len(chunks)}")
        print(f"Total documents: {len(sources)}")
        print(f"Average chunks per document: {len(chunks) // len(sources) if sources else 0}")
        
        # Section breakdown
        sections = {}
        for chunk in chunks:
            section = chunk.get('metadata', {}).get('section', 'unknown')
            sections[section] = sections.get(section, 0) + 1
        
        print("\nSection breakdown:")
        for section, count in sections.items():
            print(f"  {section}: {count}")


def validate_command(chunk_manager: StandaloneChunkManager, args):
    """Validate chunks and find problems."""
    problems = chunk_manager.validate_chunks(args.source)
    
    if problems:
        print(f"Found {len(problems)} problematic chunks:")
        print("=" * 50)
        
        for problem in problems:
            print(f"Chunk: {problem['chunk_id']}")
            print(f"Source: {problem['source']}")
            print(f"Issues: {', '.join(problem['issues'])}")
            print(f"Content length: {problem['content_length']}")
            print("-" * 30)
    else:
        scope = f" for {args.source}" if args.source else ""
        print(f"No problems found{scope}.")


def main():
    parser = argparse.ArgumentParser(description="StudyBot Chunk Management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List chunks command
    list_parser = subparsers.add_parser('list', help='List chunks')
    list_parser.add_argument('--source', help='Filter by source')
    list_parser.add_argument('--doc-id', help='Filter by document ID')
    list_parser.add_argument('--section', help='Filter by section')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum results')
    
    # Search chunks command
    search_parser = subparsers.add_parser('search', help='Search chunks')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--semantic', action='store_true', default=True, help='Use semantic search')
    search_parser.add_argument('--text', dest='semantic', action='store_false', help='Use text search')
    search_parser.add_argument('--limit', type=int, help='Maximum results')
    
    # Get chunk command
    get_parser = subparsers.add_parser('get', help='Get chunk details')
    get_parser.add_argument('chunk_id', help='Chunk ID')
    
    # Edit chunk command
    edit_parser = subparsers.add_parser('edit', help='Edit chunk content')
    edit_parser.add_argument('chunk_id', help='Chunk ID')
    edit_parser.add_argument('--content', help='New content (if not provided, will prompt)')
    
    # Delete chunk command
    delete_parser = subparsers.add_parser('delete', help='Delete chunk')
    delete_parser.add_argument('chunk_id', help='Chunk ID')
    delete_parser.add_argument('--confirm', action='store_true', help='Skip confirmation')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--source', help='Source document (if not provided, shows overall stats)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate chunks')
    validate_parser.add_argument('--source', help='Source document to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize chunk manager
    try:
        config = get_config()
        chunk_manager = StandaloneChunkManager.from_config(config)
        print("Chunk manager initialized successfully.")
        print()
    except Exception as e:
        print(f"Error initializing chunk manager: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'list':
            list_chunks_command(chunk_manager, args)
        elif args.command == 'search':
            search_chunks_command(chunk_manager, args)
        elif args.command == 'get':
            get_chunk_command(chunk_manager, args)
        elif args.command == 'edit':
            edit_chunk_command(chunk_manager, args)
        elif args.command == 'delete':
            delete_chunk_command(chunk_manager, args)
        elif args.command == 'stats':
            stats_command(chunk_manager, args)
        elif args.command == 'validate':
            validate_command(chunk_manager, args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
