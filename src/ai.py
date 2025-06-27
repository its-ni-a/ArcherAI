import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
import re

# ----------------------------- Environment variables -----------------------------
load_dotenv(dotenv_path="C:/Users/ndebo/OneDrive/Documents/Programming Projects/JWChatbot/api-key.env")

# ----------------------------- Vector Database -----------------------------
class WOLVectorDatabase:
    """Vector database specifically designed for WOL content"""
    
    def __init__(self, db_file='wol_vectors.pkl', model_name='all-MiniLM-L6-v2'):
        self.db_file = db_file
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.vectors = None
        self.metadata = {}
        
    def _load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove invisible characters
        invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in invisible_chars:
            text = text.replace(char, '')
        return text.strip()
    
    def _extract_page_info(self, page_text: str) -> Dict:
        """Extract metadata from page text"""
        lines = page_text.split('\n')
        url = ""
        title = ""
        
        # Look for URL in first few lines
        for line in lines[:3]:
            if 'wol.jw.org' in line or 'https://' in line:
                url = line.strip()
                break
        
        # Look for title (usually first meaningful line)
        for line in lines:
            clean_line = line.strip()
            if clean_line and len(clean_line) > 10 and not clean_line.startswith('---'):
                title = clean_line[:100]  # First 100 chars as title
                break
        
        return {'url': url, 'title': title}
    
    def build_database(self, content_file='wol_content.txt', chunk_size=500, overlap=50):
        """Build the vector database from content file"""
        if not os.path.exists(content_file):
            print(f"Content file {content_file} not found!")
            return False
        
        print("Building WOL vector database...")
        self._load_model()
        
        # Read content
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by pages
        pages = content.split('--- PAGE:')
        print(f"Found {len(pages)} pages")
        
        self.chunks = []
        processed_pages = 0
        
        for page_idx, page in enumerate(pages):
            if len(page.strip()) < 100:  # Skip very short pages
                continue
            
            page_text = self._clean_text(page)
            page_info = self._extract_page_info(page_text)
            
            # Split page into overlapping chunks
            words = page_text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_data = {
                    'text': chunk_text,
                    'page_idx': page_idx,
                    'chunk_idx': i // (chunk_size - overlap),
                    'url': page_info['url'],
                    'title': page_info['title'],
                    'word_count': len(chunk_words),
                    'preview': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text
                }
                
                self.chunks.append(chunk_data)
            
            processed_pages += 1
            if processed_pages % 50 == 0:
                print(f"Processed {processed_pages} pages, created {len(self.chunks)} chunks so far...")
        
        print(f"Created {len(self.chunks)} chunks from {processed_pages} pages")
        
        # Generate embeddings in batches
        print("Generating embeddings...")
        batch_size = 100
        all_vectors = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch_texts = [chunk['text'] for chunk in self.chunks[i:i + batch_size]]
            batch_vectors = self.model.encode(batch_texts, show_progress_bar=True)
            all_vectors.append(batch_vectors)
            print(f"Processed {min(i + batch_size, len(self.chunks))}/{len(self.chunks)} chunks")
        
        self.vectors = np.vstack(all_vectors)
        
        # Save metadata
        self.metadata = {
            'total_chunks': len(self.chunks),
            'total_pages': processed_pages,
            'model_name': self.model_name,
            'chunk_size': chunk_size,
            'overlap': overlap
        }
        
        # Save to disk
        self._save_database()
        print("Vector database built successfully!")
        return True
    
    def _save_database(self):
        """Save database to disk"""
        db_data = {
            'chunks': self.chunks,
            'vectors': self.vectors,
            'metadata': self.metadata
        }
        
        with open(self.db_file, 'wb') as f:
            pickle.dump(db_data, f)
        
        print(f"Database saved to {self.db_file}")
        print(f"Database size: {os.path.getsize(self.db_file) / (1024*1024):.2f} MB")
    
    def load_database(self) -> bool:
        """Load existing database"""
        if not os.path.exists(self.db_file):
            print(f"Database file {self.db_file} not found")
            return False
        
        print("Loading vector database...")
        with open(self.db_file, 'rb') as f:
            db_data = pickle.load(f)
        
        self.chunks = db_data['chunks']
        self.vectors = db_data['vectors']
        self.metadata = db_data.get('metadata', {})
        
        # Load model for searching
        self._load_model()
        
        print(f"Database loaded: {self.metadata.get('total_chunks', len(self.chunks))} chunks from {self.metadata.get('total_pages', 'unknown')} pages")
        return True
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Search for relevant content"""
        if self.vectors is None or self.model is None:
            print("Database not loaded!")
            return []
        
        # Encode query
        query_vector = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get indices sorted by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter by minimum similarity and get top_k
        results = []
        for idx in sorted_indices:
            if similarities[idx] < min_similarity:
                break
            if len(results) >= top_k:
                break
                
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        if not self.chunks:
            return {"status": "Database not loaded"}
        
        return {
            "total_chunks": len(self.chunks),
            "total_pages": self.metadata.get('total_pages', 'unknown'),
            "model_name": self.metadata.get('model_name', 'unknown'),
            "chunk_size": self.metadata.get('chunk_size', 'unknown'),
            "database_file_size_mb": os.path.getsize(self.db_file) / (1024*1024) if os.path.exists(self.db_file) else 0
        }
    
# ----------------------------- AI integration with vectors -----------------------------
class WOLResearchAssistantWithVectors:
    """WOL Research Assistant powered by vector search"""
    
    def __init__(self, vector_db: WOLVectorDatabase):
        self.vector_db = vector_db
        self.llm = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup the language model"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found!")
        
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.3,
            openai_api_key=api_key
        )
    
    def _format_context(self, relevant_chunks: List[Dict]) -> str:
        """Format relevant chunks into context"""
        if not relevant_chunks:
            return "Based on the Watchtower Online Library content:"
        
        context = "Based on the following relevant content from the Watchtower Online Library:\n\n"
        
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"Reference {i} (Relevance: {chunk['similarity']:.3f}):\n"
            if chunk.get('title'):
                context += f"Title: {chunk['title']}\n"
            if chunk.get('url'):
                context += f"Source: {chunk['url']}\n"
            context += f"Content: {chunk['text']}\n\n"
        
        return context
    
    def query(self, question: str, max_context_chunks: int = 3) -> str:
        """Answer a question using vector search + LLM"""
        # Search for relevant content
        relevant_chunks = self.vector_db.search(question, top_k=max_context_chunks)
        
        if not relevant_chunks:
            return "I cannot find relevant information in the Watchtower Online Library content to answer your question. Please try rephrasing your question or ask about topics related to Jehovah's Witnesses."
        
        # Format context
        context = self._format_context(relevant_chunks)
        
        # Create prompt
        template = f"""{context}

Purpose:
    You are a Watchtower Online Library research assistant chatbot, please respond as if you are one of Jehovah's Witnesses. 
    Your expertise is exclusively in providing factual answers to questions about Jehovah's Witnesses, specifically using information from the Watchtower Online Library. This includes general JW-related queries, Bible questions, and organizational information.

Bias:
    Do not say "according to Jehovah's Witnesses", nor "they believe", nor "they say" nor "their publications". 
    All the information from Watchtower Online Library is a fact, treat it as such.

Source Citations:
    Provide accurate and helpful responses based on official Bible teachings according to Jehovah's Witnesses and publications from the Watchtower Online Library. State all information as fact without referencing beliefs or interpretations.
    For every response please include the scripture that supports the response in the format: (Bible book Chapter Number:Verse Number)
    If possible also include the publication that supports the response in the format: References are shown as the mnemonic followed by page number(s).

Other Question Cases:
    Greetings: If the user greets you or says a similar conversational response you may answer accordingly in a kind manner
    Follow-up: If a follow up question is asked regarding a previous question, please answer that accordingly
    Not Relevant: If a question is not about jw.org, Jehovah's Witnesses, or related topics, respond with "I cannot assist with that, sorry. I can only help with questions about Jehovah's Witnesses and jw.org content."

Question: {question}
Answer: """
        
        # Get response from LLM
        try:
            response = self.llm.invoke(template)
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
        except Exception as e:
            return f"Error generating response: {e}"

# Main functions for easy setup
def setup_vector_database(rebuild=False):
    """Setup or load the vector database"""
    vector_db = WOLVectorDatabase()
    
    if rebuild or not vector_db.load_database():
        print("Building new vector database...")
        success = vector_db.build_database('wol_content.txt')
        if not success:
            return None
    else:
        print("Loaded existing vector database")
    
    return vector_db

def interactive_chat_with_vectors():
    """Interactive chat using vector search"""
    print("Setting up WOL Vector Research Assistant...")
    
    # Setup vector database
    vector_db = setup_vector_database()
    if not vector_db:
        print("Failed to setup vector database!")
        return
    
    # Setup assistant
    assistant = WOLResearchAssistantWithVectors(vector_db)
    
    # Print database stats
    stats = vector_db.get_database_stats()
    print(f"Database loaded: {stats['total_chunks']} chunks from {stats['total_pages']} pages")
    print("-" * 50)
    print("WOL Vector Research Assistant ready! Type 'quit' to exit, 'stats' for database info.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = vector_db.get_database_stats()
                for key, value in stats.items():
                    print(f"{key}: {value}")
                continue
            
            if not user_input:
                print("Please enter a question.")
                continue
            
            print("WOL Research Assistant: ", end="")
            response = assistant.query(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def test_vector_search():
    """Test the vector search functionality"""
    vector_db = setup_vector_database()
    if not vector_db:
        return
    
    test_queries = [
        "What is the Kingdom of God?",
        "Who is Jehovah?",
        "What happens when we die?",
        "How should Christians worship?",
        "What is the truth about the Trinity?"
    ]
    
    print("Testing vector search...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_db.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  Result {i} (similarity: {result['similarity']:.3f}):")
            print(f"    Preview: {result['preview']}")
            print(f"    Title: {result.get('title', 'N/A')}")

if __name__ == "__main__":
    # Choose what to run:
    
    # Option 1: Interactive chat (recommended)
    interactive_chat_with_vectors()
    
    # Option 2: Test vector search only
    # test_vector_search()
    
    # Option 3: Rebuild database (if needed)
    # setup_vector_database(rebuild=True)