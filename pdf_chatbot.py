import os
import tempfile
from typing import List, Sequence, Dict, Any
from typing_extensions import Annotated, TypedDict
import getpass

# Core LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Document processing imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store imports
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Embeddings and LLM imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import trim_messages

class PDFChatbotState(TypedDict):
    """State for our PDF chatbot"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_processed: bool
    collection_name: str

class PDFChatbot:
    def __init__(self, 
                 google_api_key: str = None,
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: str = None,
                 collection_name: str = "pdf_documents"):
        """
        Initialize the PDF chatbot
        
        Args:
            google_api_key: Google API key for Gemini
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key (if using cloud)
            collection_name: Name for the vector collection
        """
        # Set up API keys
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        elif not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
        
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        if qdrant_api_key:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Initialize embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant that answers questions based on the provided PDF document context. 
                Use the following pieces of context to answer the user's question. If you don't know the answer 
                based on the context, just say that you don't know - don't try to make up an answer.
                
                Context: {context}
                
                Always cite the relevant sections when providing answers."""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Set up message trimmer
        self.trimmer = trim_messages(
            max_tokens=4000,
            strategy="last",
            token_counter=self.llm,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        
        # Build the workflow
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(state_schema=PDFChatbotState)
        
        # Add nodes
        workflow.add_node("chat", self._chat_node)
        
        # Add edges
        workflow.add_edge(START, "chat")
        
        # Compile with memory
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
    
    def _setup_vector_store(self):
        """Set up the vector store with the processed documents"""
        # Create collection if it doesn't exist
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # Dimension for Google's embedding model
                    distance=Distance.COSINE
                )
            )
        except Exception as e:
            # Collection might already exist
            print(f"Collection setup info: {e}")
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
    
    def upload_pdf(self, pdf_path: str) -> str:
        """
        Upload and process a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Success message
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            
            # Set up vector store
            self._setup_vector_store()
            
            # Add documents to vector store
            self.vector_store.add_documents(splits)
            
            return f"Successfully processed and uploaded {len(splits)} chunks from the PDF!"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> str:
        """
        Upload and process a PDF from bytes
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Name for the temporary file
            
        Returns:
            Success message
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            # Process the PDF
            result = self.upload_pdf(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return result
            
        except Exception as e:
            return f"Error processing PDF bytes: {str(e)}"
    
    def _retrieve_context(self, query: str, k: int = 4) -> str:
        """Retrieve relevant context from the vector store"""
        if not self.vector_store:
            return "No PDF has been uploaded yet. Please upload a PDF first."
        
        try:
            # Retrieve relevant documents
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            docs = retriever.invoke(query)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
            
        except Exception as e:
            return f"Error retrieving context: {str(e)}"
    
    def _chat_node(self, state: PDFChatbotState) -> Dict[str, Any]:
        """Main chat node that handles the conversation"""
        # Get the last user message
        last_message = state["messages"][-1]
        
        if isinstance(last_message, HumanMessage):
            query = last_message.content
            
            # Retrieve context from PDF
            context = self._retrieve_context(query)
            
            # Trim messages to fit context window
            trimmed_messages = self.trimmer.invoke(state["messages"])
            
            # Create prompt with context
            prompt = self.prompt_template.invoke({
                "context": context,
                "messages": trimmed_messages
            })
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return {"messages": [AIMessage(content=response)]}
        
        return {"messages": []}
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Chat with the PDF
        
        Args:
            message: User message
            thread_id: Thread ID for conversation history
            
        Returns:
            Bot response
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        input_messages = [HumanMessage(content=message)]
        
        try:
            output = self.app.invoke({
                "messages": input_messages,
                "pdf_processed": self.vector_store is not None,
                "collection_name": self.collection_name
            }, config)
            
            return output["messages"][-1].content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_chat(self, message: str, thread_id: str = "default"):
        """
        Stream chat response
        
        Args:
            message: User message
            thread_id: Thread ID for conversation history
            
        Yields:
            Response chunks
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        input_messages = [HumanMessage(content=message)]
        
        try:
            for chunk, metadata in self.app.stream({
                "messages": input_messages,
                "pdf_processed": self.vector_store is not None,
                "collection_name": self.collection_name
            }, config, stream_mode="messages"):
                
                if isinstance(chunk, AIMessage):
                    yield chunk.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def clear_collection(self):
        """Clear the vector store collection"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self.vector_store = None
            return "Collection cleared successfully!"
        except Exception as e:
            return f"Error clearing collection: {str(e)}"

# Example usage
def main():
    """Example usage of the PDF chatbot"""
    
    # Initialize the chatbot
    chatbot = PDFChatbot(
        collection_name="my_pdf_docs"
    )
    
    # Example 1: Upload a PDF file
    # pdf_path = "your_document.pdf"
    # result = chatbot.upload_pdf(pdf_path)
    # print(result)
    
    # Example 2: Chat with the PDF
    # response = chatbot.chat("What is the main topic of this document?")
    # print(response)
    
    # Example 3: Stream response
    # for chunk in chatbot.stream_chat("Summarize the key points"):
    #     print(chunk, end="", flush=True)
    
    print("PDF Chatbot initialized successfully!")
    print("To use:")
    print("1. Upload a PDF: chatbot.upload_pdf('path/to/your/file.pdf')")
    print("2. Chat: chatbot.chat('Your question here')")
    print("3. Stream: for chunk in chatbot.stream_chat('Your question'): print(chunk, end='')")
    
    return chatbot

if __name__ == "__main__":
    # You'll need to install the required packages:
    """
    pip install langchain-core langgraph langchain-google-genai langchain-qdrant
    pip install pypdf qdrant-client langchain-community
    """
    
    chatbot = main()