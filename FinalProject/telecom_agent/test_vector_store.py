"""
Vector Store Testing Module
Tests the ChromaDB vector store for RAG pipeline functionality.
Validates document retrieval, similarity search, and RAG chain responses.
"""

import os
import time
import json
from typing import List, Dict
from pathlib import Path

from config import VECTOR_STORE_DIR, GEMINI_API_KEY
from rag_pipeline import create_vector_store, get_rag_chain, query_knowledge_base
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# ============================================================
# Test Suite: Vector Store Operations
# ============================================================

class VectorStoreTestSuite:
    """Comprehensive test suite for ChromaDB vector store."""

    def __init__(self):
        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        self.vectorstore = None
        self.rag_chain = None

    def log_test(self, name: str, passed: bool, message: str = "", duration: float = 0):
        """Log test result."""
        self.results["tests_run"] += 1
        if passed:
            self.results["tests_passed"] += 1
            status = "✅ PASS"
        else:
            self.results["tests_failed"] += 1
            status = "❌ FAIL"

        self.results["test_details"].append({
            "name": name,
            "status": status,
            "message": message,
            "duration_ms": round(duration * 1000, 2)
        })
        print(f"{status} | {name}")
        if message:
            print(f"    └─ {message}")

    def test_vector_store_exists(self):
        """Test 1: Verify vector store file exists."""
        start = time.time()
        db_path = os.path.join(VECTOR_STORE_DIR, "chroma.sqlite3")
        exists = os.path.exists(db_path)
        duration = time.time() - start

        if exists:
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            self.log_test(
                "Vector Store File Exists",
                True,
                f"Found chroma.sqlite3 ({size_mb:.2f} MB)",
                duration
            )
        else:
            self.log_test(
                "Vector Store File Exists",
                False,
                f"chroma.sqlite3 not found at {db_path}",
                duration
            )

    def test_vector_store_load(self):
        """Test 2: Load vector store from disk."""
        start = time.time()
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=GEMINI_API_KEY
            )
            self.vectorstore = Chroma(
                persist_directory=VECTOR_STORE_DIR,
                embedding_function=embeddings
            )
            duration = time.time() - start
            self.log_test(
                "Vector Store Load",
                True,
                f"Successfully loaded ChromaDB from {VECTOR_STORE_DIR}",
                duration
            )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Vector Store Load",
                False,
                f"Error loading vector store: {str(e)}",
                duration
            )

    def test_vector_store_metadata(self):
        """Test 3: Retrieve vector store metadata and statistics."""
        if not self.vectorstore:
            self.log_test("Vector Store Metadata", False, "Vector store not loaded")
            return

        start = time.time()
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            duration = time.time() - start

            self.log_test(
                "Vector Store Metadata",
                True,
                f"Total documents/chunks: {count}",
                duration
            )

            # Store for later tests
            self.doc_count = count
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Vector Store Metadata",
                False,
                f"Error retrieving metadata: {str(e)}",
                duration
            )

    def test_similarity_search_basic(self):
        """Test 4: Test basic similarity search with telecom query."""
        if not self.vectorstore:
            self.log_test("Similarity Search (Basic)", False, "Vector store not loaded")
            return

        start = time.time()
        try:
            query = "network outage troubleshooting"
            docs = self.vectorstore.similarity_search(query, k=3)
            duration = time.time() - start

            if len(docs) > 0:
                self.log_test(
                    "Similarity Search (Basic)",
                    True,
                    f"Retrieved {len(docs)} documents for '{query}'",
                    duration
                )
                print(f"    └─ Top result excerpt: {docs[0].page_content[:100]}...")
            else:
                self.log_test(
                    "Similarity Search (Basic)",
                    False,
                    f"No documents returned for '{query}'",
                    duration
                )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Similarity Search (Basic)",
                False,
                f"Error during similarity search: {str(e)}",
                duration
            )

    def test_similarity_search_multiple_queries(self):
        """Test 5: Test similarity search with multiple telecom queries."""
        if not self.vectorstore:
            self.log_test("Similarity Search (Multiple Queries)", False, "Vector store not loaded")
            return

        telecom_queries = [
            "tower down service restoration",
            "fiber optic cable damage repair",
            "signal degradation troubleshooting",
            "call drop issue resolution",
            "data speed throughput optimization"
        ]

        all_succeeded = True
        start = time.time()

        for query in telecom_queries:
            try:
                docs = self.vectorstore.similarity_search(query, k=2)
                if len(docs) == 0:
                    all_succeeded = False
                    print(f"    ⚠️  No results for: {query}")
            except Exception as e:
                all_succeeded = False
                print(f"    ⚠️  Error for query '{query}': {str(e)}")

        duration = time.time() - start
        self.log_test(
            "Similarity Search (Multiple Queries)",
            all_succeeded,
            f"Tested {len(telecom_queries)} different telecom queries",
            duration
        )

    def test_similarity_search_with_scores(self):
        """Test 6: Test similarity search with relevance scores."""
        if not self.vectorstore:
            self.log_test("Similarity Search with Scores", False, "Vector store not loaded")
            return

        start = time.time()
        try:
            query = "hardware failure diagnosis"
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=3)
            duration = time.time() - start

            if len(docs_with_scores) > 0:
                scores = [score for _, score in docs_with_scores]
                avg_score = sum(scores) / len(scores)
                self.log_test(
                    "Similarity Search with Scores",
                    True,
                    f"Retrieved {len(docs_with_scores)} docs, avg relevance score: {avg_score:.3f}",
                    duration
                )
                for i, (doc, score) in enumerate(docs_with_scores, 1):
                    print(f"    ├─ Doc {i} score: {score:.4f}")
            else:
                self.log_test(
                    "Similarity Search with Scores",
                    False,
                    "No documents retrieved",
                    duration
                )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Similarity Search with Scores",
                False,
                f"Error: {str(e)}",
                duration
            )

    def test_retriever_creation(self):
        """Test 7: Create and test a retriever from vector store."""
        if not self.vectorstore:
            self.log_test("Retriever Creation", False, "Vector store not loaded")
            return

        start = time.time()
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            test_query = "network failure recovery"
            results = retriever.invoke(test_query)
            duration = time.time() - start

            self.log_test(
                "Retriever Creation",
                len(results) > 0,
                f"Retriever returned {len(results)} documents for '{test_query}'",
                duration
            )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Retriever Creation",
                False,
                f"Error creating retriever: {str(e)}",
                duration
            )

    def test_rag_chain_initialization(self):
        """Test 8: Initialize RAG chain."""
        start = time.time()
        try:
            self.rag_chain = get_rag_chain(self.vectorstore)
            duration = time.time() - start
            self.log_test(
                "RAG Chain Initialization",
                self.rag_chain is not None,
                "RAG chain created successfully with retriever",
                duration
            )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "RAG Chain Initialization",
                False,
                f"Error initializing RAG chain: {str(e)}",
                duration
            )

    def test_rag_chain_query(self):
        """Test 9: Query RAG chain for telecom incident resolution."""
        if not self.rag_chain:
            self.log_test("RAG Chain Query", False, "RAG chain not initialized")
            return

        start = time.time()
        try:
            question = "How do I troubleshoot a network outage affecting multiple sites?"
            answer = self.rag_chain.invoke(question)
            duration = time.time() - start

            if answer and len(answer) > 50:
                self.log_test(
                    "RAG Chain Query",
                    True,
                    f"Generated {len(answer)} character response",
                    duration
                )
                print(f"    └─ Response preview: {answer[:150]}...")
            else:
                self.log_test(
                    "RAG Chain Query",
                    False,
                    "RAG chain returned empty or very short response",
                    duration
                )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "RAG Chain Query",
                False,
                f"Error querying RAG chain: {str(e)}",
                duration
            )

    def test_knowledge_base_query(self):
        """Test 10: Query knowledge base wrapper function."""
        start = time.time()
        try:
            result = query_knowledge_base(
                "What are the steps to restore service after a fiber cut?"
            )
            duration = time.time() - start

            has_answer = result.get("answer") and len(result["answer"]) > 0
            has_sources = len(result.get("sources", [])) > 0

            self.log_test(
                "Knowledge Base Query Wrapper",
                has_answer and has_sources,
                f"Answer: {len(result.get('answer', ''))} chars, Sources: {len(result.get('sources', []))}",
                duration
            )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Knowledge Base Query Wrapper",
                False,
                f"Error: {str(e)}",
                duration
            )

    def test_empty_query_handling(self):
        """Test 11: Handle empty or minimal queries gracefully."""
        if not self.vectorstore:
            self.log_test("Empty Query Handling", False, "Vector store not loaded")
            return

        start = time.time()
        try:
            # Test with empty string
            docs = self.vectorstore.similarity_search("", k=1)
            # Test with very short query
            docs2 = self.vectorstore.similarity_search("fix", k=1)
            duration = time.time() - start

            # Both should either work or fail gracefully
            self.log_test(
                "Empty Query Handling",
                True,
                "Handled edge cases without crashing",
                duration
            )
        except Exception as e:
            duration = time.time() - start
            self.log_test(
                "Empty Query Handling",
                False,
                f"Error handling edge cases: {str(e)}",
                duration
            )

    def test_document_metadata(self):
        """Test 12: Verify document metadata is properly stored."""
        if not self.vectorstore:
            self.log_test("Document Metadata", False, "Vector store not loaded")
            return

        start = time.time()
        try:
            docs = self.vectorstore.similarity_search("troubleshooting", k=1)
            if docs:
                metadata = docs[0].metadata
                has_source = "source" in metadata
                has_content = len(docs[0].page_content) > 0
                duration = time.time() - start

                self.log_test(
                    "Document Metadata",
                    has_source and has_content,
                    f"Metadata keys: {list(metadata.keys())}, Content length: {len(docs[0].page_content)}",
                    duration
                )
            else:
                duration = time.time() - start
                self.log_test("Document Metadata", False, "No documents retrieved", duration)
        except Exception as e:
            duration = time.time() - start
            self.log_test("Document Metadata", False, f"Error: {str(e)}", duration)

    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "=" * 70)
        print("  VECTOR STORE TEST SUITE")
        print("=" * 70 + "\n")

        self.test_vector_store_exists()
        self.test_vector_store_load()
        self.test_vector_store_metadata()
        self.test_similarity_search_basic()
        self.test_similarity_search_multiple_queries()
        self.test_similarity_search_with_scores()
        self.test_retriever_creation()
        self.test_rag_chain_initialization()
        self.test_rag_chain_query()
        self.test_knowledge_base_query()
        self.test_empty_query_handling()
        self.test_document_metadata()

        self.print_summary()

    def print_summary(self):
        """Print test summary report."""
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        print(f"Tests Run:     {self.results['tests_run']}")
        print(f"Tests Passed:  {self.results['tests_passed']} ✅")
        print(f"Tests Failed:  {self.results['tests_failed']} ❌")

        pass_rate = (self.results['tests_passed'] / self.results['tests_run'] * 100) if self.results['tests_run'] > 0 else 0
        print(f"Pass Rate:     {pass_rate:.1f}%")
        print("=" * 70 + "\n")

        return {
            "total_tests": self.results["tests_run"],
            "passed": self.results["tests_passed"],
            "failed": self.results["tests_failed"],
            "pass_rate": pass_rate,
            "details": self.results["test_details"]
        }


# ============================================================
# Performance Benchmarks
# ============================================================

def run_performance_benchmark():
    """Run performance benchmarks on vector store operations."""
    print("\n" + "=" * 70)
    print("  PERFORMANCE BENCHMARKS")
    print("=" * 70 + "\n")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vectorstore = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings
        )

        benchmarks = []

        # Benchmark 1: Similarity search performance
        queries = [
            "network outage",
            "fiber cut",
            "tower failure",
            "call drop",
            "data throughput"
        ]

        print("Benchmarking Similarity Search Performance...")
        for query in queries:
            start = time.time()
            docs = vectorstore.similarity_search(query, k=4)
            duration = time.time() - start
            benchmarks.append({
                "operation": "similarity_search",
                "query": query,
                "results_count": len(docs),
                "duration_ms": round(duration * 1000, 2)
            })
            print(f"  {query:20} | {duration*1000:7.2f}ms | {len(docs)} docs")

        # Benchmark 2: Retriever performance
        print("\nBenchmarking Retriever Performance...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        for query in queries[:3]:
            start = time.time()
            docs = retriever.invoke(query)
            duration = time.time() - start
            benchmarks.append({
                "operation": "retriever",
                "query": query,
                "results_count": len(docs),
                "duration_ms": round(duration * 1000, 2)
            })
            print(f"  {query:20} | {duration*1000:7.2f}ms | {len(docs)} docs")

        print("\n" + "=" * 70 + "\n")
        return benchmarks

    except Exception as e:
        print(f"❌ Benchmark Error: {str(e)}\n")
        return []


# ============================================================
# Export Results
# ============================================================

def export_results(test_summary: dict, benchmarks: list, output_file: str = None):
    """Export test results to JSON file."""
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(__file__),
            "vector_store_test_results.json"
        )

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_summary": test_summary,
        "performance_benchmarks": benchmarks
    }

    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to {output_file}")
    except Exception as e:
        print(f"❌ Error exporting results: {str(e)}")


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    # Run test suite
    tester = VectorStoreTestSuite()
    summary = tester.run_all_tests()

    # Run performance benchmarks
    benchmarks = run_performance_benchmark()

    # Export results
    export_results(summary, benchmarks)
