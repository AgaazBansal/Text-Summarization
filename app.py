import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document
import re
import time
import os
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Streamlit App Configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stMarkdown h1 {
        color: #1E3D59;
        text-align: center;
    }
    .stMarkdown h3 {
        color: #1E3D59;
    }
    .stSidebar {
        background-color: #F0F2F6;
        padding: 2rem;
    }
    .stSidebar .stTextInput>div>div>input {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Main Container
with st.container():
    # Header with Logo
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("📝 AI Content Summarizer")
        st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app helps you summarize content from:
    - YouTube videos
    - Web articles
    - Blog posts
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - Maximum video duration: 30 minutes
    - For longer videos, please use shorter segments
    - Processing time may vary based on content length
    """)
    
    st.markdown("---")
    st.markdown("### 📌 Instructions")
    st.markdown("""
    1. Paste a YouTube or website URL
    2. Click 'Summarize' to get started
    """)

# Main Content Area
with st.container():
    # URL Input with better styling
    generic_url = st.text_input(
        "🔗 Enter URL (YouTube or Website)",
        placeholder="https://www.youtube.com/watch?v=... or https://example.com",
        label_visibility="visible"
    )

def extract_video_id(url):
    """Extracts video ID from various YouTube URL formats."""
    try:
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"youtu\.be\/([0-9A-Za-z_-]{11})",
            r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        st.error(f"Error extracting video ID: {str(e)}")
        return None

def get_video_duration(video_id):
    """Get video duration in minutes."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_parts = transcript.fetch()
        if transcript_parts:
            # Get the last timestamp to determine total duration
            last_part = transcript_parts[-1]
            duration_minutes = last_part.start / 60
            return duration_minutes
        return 0
    except:
        return 0

def get_youtube_transcript(video_url):
    """Fetches the transcript of a YouTube video with improved language handling."""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return "Invalid YouTube URL"

        # Check video duration
        duration_minutes = get_video_duration(video_id)
        if duration_minutes > 30:
            return f"Video duration ({duration_minutes:.1f} minutes) exceeds the 30-minute limit. Please use a shorter video or segment."

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(["en"])
        except:
            # If English not available, try to get any available transcript
            transcript = next(iter(transcript_list), None)

        if not transcript:
            return "No transcript found for this video."

        # Fetch transcript
        transcript_parts = transcript.fetch()
        
        # Format transcript with timestamps and chunk the content
        formatted_text = ""
        current_chunk = ""
        chunk_size = 0
        max_chunk_size = 4000  # Adjust this based on model's context window
        
        for part in transcript_parts:
            text_with_timestamp = f"[{int(part.start//60):02d}:{int(part.start%60):02d}] {part.text}\n"
            
            # If adding this part would exceed chunk size, start a new chunk
            if chunk_size + len(text_with_timestamp) > max_chunk_size:
                formatted_text += current_chunk + "\n---\n"  # Add separator between chunks
                current_chunk = text_with_timestamp
                chunk_size = len(text_with_timestamp)
            else:
                current_chunk += text_with_timestamp
                chunk_size += len(text_with_timestamp)
        
        # Add the last chunk
        formatted_text += current_chunk
        
        return formatted_text

    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcripts available for this video."
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"

def process_website(url):
    """Process website content with improved error handling."""
    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        return loader.load()
    except Exception as e:
        return f"Error processing website: {str(e)}"

def generate_summary(docs, llm, prompt):
    """Generate summary with improved handling of long content."""
    try:
        # Split content into chunks if needed
        all_summaries = []
        for doc in docs:
            content = doc.page_content
            chunks = content.split("\n---\n")
            
            for chunk in chunks:
                if chunk.strip():
                    chunk_doc = Document(page_content=chunk)
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    chunk_summary = chain.run([chunk_doc])
                    all_summaries.append(chunk_summary)
        
        # Combine all chunk summaries
        if len(all_summaries) > 1:
            # Create a final summary of all chunk summaries
            final_prompt = PromptTemplate(
                template="""Create a coherent and concise final summary of the following content in 300 words. 
                Combine the key points and main ideas from all sections:
                Content: {text}
                """,
                input_variables=["text"]
            )
            final_chain = load_summarize_chain(llm, chain_type="stuff", prompt=final_prompt)
            final_summary = final_chain.run([Document(page_content="\n\n".join(all_summaries))])
            return final_summary
        else:
            return all_summaries[0]
            
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Update the prompt template for better summarization
prompt_template = """
Please provide a detailed and comprehensive summary of the following content in 800-1000 words. 
Include:
1. Main points and key ideas
2. Important details and examples
3. Technical concepts (if any)
4. Key conclusions or takeaways
5. Supporting arguments or evidence

Do not add information that is not present in the original text.
If the content is technical, maintain the technical accuracy. If it's a conversation, preserve the main discussion points.

Content: {text}
"""

# Initialize docs variable
docs = []

# Summarization Button with better styling
if st.button("✨ Summarize Content", key="summarize_button", disabled=st.session_state.processing):
    if not generic_url.strip():
        st.error("⚠️ Please provide a URL to get started.")
    elif not validators.url(generic_url):
        st.error("⚠️ Please enter a valid URL (YouTube or Website).")
    else:
        try:
            st.session_state.processing = True
            with st.spinner("🔄 Processing your request..."):
                time.sleep(0.5)
                
                # LLM Configuration
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
                
                # Prompt Template
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    transcript_text = get_youtube_transcript(generic_url)
                    if "Error" in transcript_text or "No transcript" in transcript_text:
                        st.error(transcript_text)
                    else:
                        docs = [Document(page_content=transcript_text)]
                else:
                    docs = process_website(generic_url)
                    if isinstance(docs, str):
                        st.error(docs)
                        docs = []

                if docs:
                    output_summary = generate_summary(docs, llm, prompt)
                    
                    if "Error" in output_summary:
                        st.error(output_summary)
                    else:
                        # Display the summary in a nice format
                        st.markdown("### 📋 Detailed Summary")
                        st.markdown(output_summary)
                        
                        # Create a container for download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Text download button
                            st.download_button(
                                label="📥 Download as Text",
                                data=output_summary,
                                file_name=f"detailed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        with col2:
                            # Create PDF
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            
                            # Add title
                            pdf.set_font("Arial", 'B', 16)
                            pdf.cell(200, 10, txt="Detailed Content Summary", ln=True, align='C')
                            pdf.ln(10)
                            
                            # Add content
                            pdf.set_font("Arial", size=12)
                            pdf.multi_cell(0, 10, txt=output_summary)
                            
                            # Add metadata
                            pdf.ln(10)
                            pdf.set_font("Arial", 'I', 8)
                            pdf.cell(0, 5, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                            pdf.cell(0, 5, txt=f"Source: {generic_url}", ln=True)
                            
                            # Get PDF bytes
                            pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            
                            # PDF download button
                            st.download_button(
                                label="📄 Download as PDF",
                                data=pdf_bytes,
                                file_name=f"detailed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.error("❌ Could not process the content. Please try again.")
                    
        except Exception as e:
            st.exception(f"❌ An error occurred: {str(e)}")
        finally:
            st.session_state.processing = False

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>Built Different. Engineered to Impress. 💡</p>
</div>
""", unsafe_allow_html=True)
