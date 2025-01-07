import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.docstore.document import Document
import time

# Page configuration 
st.set_page_config(
    page_title="Smart Content Summarizer",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #4CAF50;
        border-left: 5px solid  #00802b;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
col1, col2 = st.columns([1, 4])
# with col1:
    # st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.governanceinstitute.com.au%2Fthought-leadership%2Fai-ethics-and-governance-white-paper-launch%2F&psig=AOvVaw1mx4cATLtV4U-FHLlFGH5w&ust=1736324164727000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCKC1mNyV44oDFQAAAAAdAAAAABAE")
with col2:
    st.title("🌟 Smart Content Summarizer")
    st.markdown("*Transform long content into concise summaries powered by AI*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input with validation
    groq_api_key = st.text_input("GROQ API Key", type="password", value="")
    if groq_api_key:
        st.success("API Key provided! ✅")
    
    # Model selection
    model_option = st.selectbox(
        "Select AI Model",
        ["gemma2-9b-it", "mixtral-8x7b-32768", "llama2-70b-4096"],
        help="Choose the AI model for summarization"
    )
    
    # Summary options
    st.subheader("Summary Options")
    summary_length = st.select_slider(
        "Summary Length",
        options=["Very Short", "Short", "Medium", "Long", "Very Long"],
        value="Medium"
    )
    
    style_option = st.selectbox(
        "Summary Style",
        ["Professional", "Casual", "Academic", "Creative"],
        help="Choose the tone of the summary"
    )

# Main content 
st.markdown("### 🔗 Enter Content URL")
url_placeholder = "Enter YouTube URL or website link (e.g., https://example.com)"
generic_URL = st.text_input("URL", placeholder=url_placeholder, label_visibility="collapsed")

# text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    length_function=len
)

def get_youtube_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be',):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None

# Adjust prompt
def get_custom_prompts(length, style):
    length_words = {
        "Very Short": "50-75",
        "Short": "100-150",
        "Medium": "200-250",
        "Long": "300-350",
        "Very Long": "400-500"
    }
    
    style_instructions = {
        "Professional": "in a clear, formal, and business-appropriate tone",
        "Casual": "in a conversational and easy-to-understand way",
        "Academic": "with academic rigor and scholarly language",
        "Creative": "in an engaging and creative style"
    }
    
    map_template = f"""
    Write a {style_instructions[style]} summary of the following content:
    {{text}}
    CONCISE SUMMARY:
    """
    
    combine_template = f"""
    Create a {length_words[length]}-word summary {style_instructions[style]} by combining these summaries:
    {{text}}
    FINAL SUMMARY:
    """
    
    return (
        PromptTemplate(template=map_template, input_variables=["text"]),
        PromptTemplate(template=combine_template, input_variables=["text"])
    )

if st.button("🚀 Generate Summary", help="Click to generate summary"):
    # Validation
    if not groq_api_key.strip() or not generic_URL.strip():
        st.error("🚫 Please provide both API key and URL to proceed!")
    elif not validators.url(generic_URL):
        st.error("🔗 Please enter a valid URL!")
    else:
        try:
            # Create progress elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize LLM
            llm = ChatGroq(model=model_option, groq_api_key=groq_api_key)
            
            # Process content
            with st.spinner("🔄 Processing content..."):
                # Update progress
                status_text.text("Fetching content...")
                progress_bar.progress(25)
                
                # Handle different content types
                if "youtube.com" in generic_URL or "youtu.be" in generic_URL:
                    video_id = get_youtube_id(generic_URL)
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = ' '.join([t['text'] for t in transcript_list])
                    docs = [Document(page_content=transcript_text, metadata={"source": generic_URL})]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_URL],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()
                
                # Update progress
                status_text.text("Processing content...")
                progress_bar.progress(50)
                
                # Split and summarize
                splits = text_splitter.split_documents(docs)
                map_prompt, combine_prompt = get_custom_prompts(summary_length, style_option)
                
                # Update progress
                status_text.text("Generating summary...")
                progress_bar.progress(75)
                
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                    verbose=True
                )
                
                output_summary = chain.run(splits)
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("Summary generated!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Display summary
                st.markdown("### 📝 Generated Summary")
                st.markdown(
                    f"""<div class="success-box">
                    {output_summary}
                    </div>""",
                    unsafe_allow_html=True
                )
                
                #utility buttons
                col1, col2 = st.columns([1,6])
                with col1:
                    st.download_button(
                        "💾 Save",
                        output_summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            if "youtube" in str(e).lower():
                st.info("ℹ️ For YouTube videos, please ensure the video has closed captions available.")