import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
import pypdf as PyPDF2
import os
from dotenv import load_dotenv

# --- 1. ENVIRONMENT SETUP ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Using the high-performance Llama 3.3 model for enterprise-grade reasoning
my_llm = LLM(
    model="groq/llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# --- 2. HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF for the agents to read."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def load_business_profile():
    """Loads your company strengths from a local text file."""
    try:
        with open("my_business_profile.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Standard IT services and software engineering expertise."

# --- 3. STREAMLIT UI SETUP ---
st.set_page_config(page_title="BidFlow AI | Enterprise RFP", layout="wide", page_icon="🚀")

# Custom CSS for a more professional "Saas" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 BidFlow AI: Strategic RFP Response Engine")
st.markdown("Automating the bridge between Client Requirements and Your Technical Solutions.")
st.divider()

# --- 4. SIDEBAR: THE BUSINESS IDENTITY (YOUR SIDE) ---
with st.sidebar:
    st.header("🏢 Your Business Identity")
    st.markdown("This info is used to prove why you are the best fit.")
    profile = load_business_profile()
    
    if st.checkbox("🔍 View My Business Profile", help="Show the facts the AI uses to represent YOU."):
        st.info(profile)
    
    st.divider()
    st.caption("v2.1 | Powered by Agentic RAG Architecture")

# --- 5. MAIN AREA: THE CLIENT OPPORTUNITY (THEIR SIDE) ---
uploaded_file = st.file_uploader("Upload Client Tender/RFP (PDF)", type="pdf")

if uploaded_file:
    # We use session state to keep the data persistent
    if 'tender_text' not in st.session_state:
        with st.spinner("Analyzing document structure..."):
            st.session_state.tender_text = extract_text_from_pdf(uploaded_file)
            st.success("Tender Content Successfully Ingested!")

    # NEW SECTION: SHOW THE EXTRACTED REQUIREMENTS
    st.subheader("📋 Client Requirements Analysis")
    st.markdown("The AI has identified the following core needs from the uploaded document:")
    
    # Checkbox to reveal the "Customer Side"
    if st.checkbox("👀 Show Extracted Tender Requirements", help="Show what the AI understood from the CLIENT'S document."):
        # We run a quick mini-task to extract just the requirements for display
        if 'extracted_reqs' not in st.session_state:
            with st.spinner("Agents are extracting mandatory points..."):
                extraction_agent = Agent(
                    role='Requirement Analyst',
                    goal='Extract 5 mandatory requirements from the text.',
                    backstory='Specialist in identifying project scope and deliverables.',
                    llm=my_llm,
                    allow_delegation=False
                )
                extraction_task = Task(
                    description=f"Summarize the 5 most important technical requirements from this text: {st.session_state.tender_text[:5000]}",
                    expected_output="A bulleted list of 5 requirements.",
                    agent=extraction_agent
                )
                crew_extract = Crew(agents=[extraction_agent], tasks=[extraction_task])
                st.session_state.extracted_reqs = crew_extract.kickoff()
        
        st.warning(st.session_state.extracted_reqs)

    st.divider()

    # --- 6. AGENTIC WORKFLOW: GENERATING THE PROPOSAL ---
    if st.button("✨ Generate Strategic Business Proposal"):
        
        auditor = Agent(
            role='Compliance Auditor',
            goal='Ensure our response addresses every technical requirement found.',
            backstory='Expert in matching corporate capabilities to project needs.',
            llm=my_llm,
            verbose=True,
            allow_delegation=False
        )

        strategist = Agent(
            role='Senior Proposal Writer',
            goal='Write a persuasive, high-value proposal based on our profile and the tender.',
            backstory='Master of technical sales and executive communication.',
            llm=my_llm,
            verbose=True,
            allow_delegation=False
        )

        task_matching = Task(
            description=f"Compare these tender requirements: {st.session_state.tender_text[:6000]} against our company profile: {profile}. Find the 3 strongest 'selling points' we have for this client.",
            expected_output="An internal analysis of why we are the right choice.",
            agent=auditor
        )

        task_final_proposal = Task(
            description="Using the audit results, write a formal 4-section proposal: Executive Summary, Proposed Solution, Past Experience, and Call to Action.",
            expected_output="A professional, markdown-formatted business proposal document.",
            agent=strategist
        )

        crew = Crew(
            agents=[auditor, strategist],
            tasks=[task_matching, task_final_proposal],
            process=Process.sequential
        )

        with st.status("Agents are collaborating on your proposal...", expanded=True) as status:
            final_result = crew.kickoff()
            status.update(label="Proposal Generation Complete!", state="complete", expanded=False)

        st.subheader("📄 Final Strategic Proposal")
        st.markdown(final_result)
        
        st.download_button(
            label="Download Proposal as Text",
            data=str(final_result),
            file_name="Strategic_Proposal.txt",
            mime="text/plain"
        )