import streamlit as st
import openai
import anthropic
import pandas as pd
import re
import json
import logging
from stqdm import stqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

Opus = "claude-3-opus-20240229"
Sonnet = "claude-3-sonnet-20240229"
Haiku = "claude-3-haiku-20240307"

# Set up API keys and models
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai = openai.Client(api_key=OPENAI_API_KEY)
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
gpt_model = "ft:gpt-3.5-turbo-0125:personal:idea-generator:9DgQ5nsD"




# App title and description
st.set_page_config(page_title="Idea Generation and Refinement", layout="wide")

# Set up logging
logging.basicConfig(filename="app.log", level=logging.INFO)

def log_usage(message):
    logging.info(f"Usage: {message}")

log_usage("App started")

custom_css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: bold;
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p:hover {
        color: #ff7f0e;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #f8f8f8;
        padding: 20px;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-panel"] h2 {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-panel"] p {
        font-size: 16px;
        color: #333;
    }
</style>"""


st.markdown(custom_css, unsafe_allow_html=True)

# Define colors and styles
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"
background_color = "#f8f8f8"

st.markdown(f"""
    <style>
    body {{
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1 {{
        font-size: 2.5rem;
    }}
    h2 {{
        font-size: 2rem;
    }}
    p {{
        font-size: 1.1rem;
    }}
    .reportview-container {{
        background-color: {background_color};
    }}
    .stButton > button {{
        background-color: {primary_color};
        color: white;
    }}
    .stTextInput > div > div > input {{
        border-color: {secondary_color};
    }}
    .icon {{
        font-size: 2rem;
        margin-right: 0.5rem;
    }}
    .reportview-container .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Fractl Trained Ideation Pipeline")
st.markdown("""
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
<p style="font-size: 18px;">
This tool harnesses the power of advanced AI models to help you generate creative and unique ideas for your projects. Here's how it works:
</p>
<ol style="font-size: 16px;">
<li>Select the desired number of finalized ideas you need.</li>
<li>Our fine-tuned GPT-3.5 model, trained on thousands of past Fractl ideas, will generate a pool of initial ideas based on your input.</li>
<li>The Claude-3 model will then evaluate and curate the best ideas from the generated pool.</li>
<li>For each idea you like best, the tool will develop detailed briefs to help you understand and implement them effectively.</li>
</ol>
<p style="font-size: 18px;">
Get ready to explore a world of creative possibilities and take your projects to the next level!
</p>
</div>
""", unsafe_allow_html=True)




@st.cache_data
def get_ideas(topic: str, num_ideas: int, temperature: float, model: str) -> list:
    """
    Call GPT-3 API to generate ideas.

    Args:
        topic (str): The topic for idea generation.
        num_ideas (int): The number of ideas to generate.
        temperature (float): The temperature value for controlling creativity.
        model (str): The GPT-3 model to use for idea generation.

    Returns:
        list: The generated ideas.
    """
    num_ideas_mult = (num_ideas*10)
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Generate the most newsworthy possible idea for the given topic. Start your answer with Title:"},
                {"role": "user", "content": f"Topic: {topic} \n Your idea: \n"}
            ],
            max_tokens=800,
            n=num_ideas,
            stop=None,
            temperature=temperature,
        )
        ideas = [choice.message.content.strip() for choice in response.choices]
        return ideas

    except Exception as e:
        st.error(f"Error: {str(e)}")
        raise

@st.cache_data
def evaluate_ideas(generated_ideas: list, num_ideas: int, model: str) -> list:
    try:
        system_prompt = "You are an expert editor at a major news publication. Your task is to select the most newsworthy and interesting story ideas from a list of brainstormed ideas. For your selection, propose an optimized title, description, justification, specific step by step methodology, and names with links if possible to required datasets/sources."

        def get_idea_summary(idea):
            prompt = f"""Here is a brainstormed idea:

            {idea}

            Please provide concrete details for your chosen idea in the following format:

            Title: [Enhanced title for the idea]
            Description: [Detailed description of the idea, including the lede, newsworthy hooks, target audience, and why they should care]
            Justification: [Justification for selecting this idea]
            Methodology: [Methodology for producing this idea, including feasibility within a 2-week timeline]
            Datasets/Sources: [Datasets, sources, technologies, and tools needed to accomplish this idea]"""

            response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                system=system_prompt,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return response.content[0].text

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_idea_summary, idea) for idea in generated_ideas[:num_ideas]]
            idea_summaries = [future.result() for future in stqdm(as_completed(futures), total=len(futures))]

        return idea_summaries[:num_ideas]

    except anthropic.APIError as e:
        st.error(f"Anthropic API Error: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Error: {str(e)}")
        raise




@st.cache_data
def fix_json_with_gpt(json_str, expected_format):
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at fixing JSON strings to match the expected format."
            },
            {
                "role": "user",
                "content": f"Fix the following JSON string to match the expected format:\n\nExpected format:\n{expected_format}\n\nJSON string to fix:\n{json_str}"
            }
        ],
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    fixed_json = response.choices[0].message.content.strip()
    print(fixed_json)
    return fixed_json

@st.cache_data
def generate_briefs(selected_ideas: list, model: str) -> list:
    try:
        def generate_idea_brief(idea):
            prompt = f"""Enhance and make more specific each part of the brief, especially providing sources for datasets needed for the idea and methodological guidance to help prevent roadblocks or timesucks.
            
            Idea: {idea}"""
            
            response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            content = response.content[0].text
            
            expected_format = """
            {
                "title": "Enhanced title for the idea",
                "description": "Detailed description of the idea, including the lede, newsworthy hooks, target audience, and why they should care",
                "justification": "Justification for selecting this idea",
                "methodology": "Methodology for producing this idea, including feasibility within a 2-week timeline",
                "datasets_sources": "Datasets, sources, technologies, and tools needed to accomplish this idea"
            }
            """
            
            fixed_json = fix_json_with_gpt(content, expected_format)
            idea_brief = json.loads(fixed_json)
            return idea_brief

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_idea_brief, idea) for idea in selected_ideas]
            idea_briefs = [future.result() for future in stqdm(as_completed(futures), total=len(futures))]

        return idea_briefs

    except anthropic.APIError as e:
        st.error(f"Anthropic API Error: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Error: {str(e)}")
        raise



def export_briefs(idea_briefs: list) -> None:
    """
    Export idea briefs to a PDF document.
    Args:
        idea_briefs (list): The list of idea briefs to export.
    """
    try:
        import tempfile
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.colors import HexColor

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            textColor=HexColor('#1f77b4'),
            spaceBefore=12,
            spaceAfter=6,
        )
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            textColor=HexColor('#ff7f0e'),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            spaceBefore=6,
            spaceAfter=6,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            doc = SimpleDocTemplate(tmp_file.name, pagesize=letter)
            elements = []
            for brief in idea_briefs:
                elements.append(Paragraph(brief["title"], title_style))
                elements.append(Paragraph("Description", section_style))
                elements.append(Paragraph(brief["description"], body_style))
                elements.append(Paragraph("Justification", section_style))
                elements.append(Paragraph(brief["justification"], body_style))
                elements.append(Paragraph("Methodology", section_style))
                elements.append(Paragraph(brief["methodology"], body_style))
                elements.append(Paragraph("Datasets/Sources", section_style))
                elements.append(Paragraph(brief["datasets_sources"], body_style))
                elements.append(Spacer(1, 24))
            doc.build(elements)
            st.download_button(
                label="Download Idea Briefs",
                data=tmp_file.read(),
                file_name="idea_briefs.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.error(f"Error exporting idea briefs: {str(e)}")

tab1, tab2, tab3, tab4 = st.tabs(["Idea Brainstorming with Fractl Finetuned Model", "Idea Evaluation with Claude", "Idea Selection", "Download Final Results"])
with tab1:
    st.subheader("Idea Generation")
    topic = st.text_input("Enter a topic", help="Provide a topic for idea generation")
    num_ideas = st.number_input("Number of ideas to generate", min_value=1, max_value=1000, value=10, help="Select the number of ideas to generate (1-1000)")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Adjust the creativity level (0.0-1.0)")
    
    generate_button = st.button("Brainstorm Ideas")
    if generate_button:
        if not topic:
            st.warning("Please enter a topic.")
        else:
            try:
                progress_text = "Generating ideas..."
                my_bar = st.progress(0, text=progress_text)
                
                generated_ideas = get_ideas(topic, num_ideas, temperature, gpt_model)
                
                for i in range(num_ideas):
                    progress = (i + 1) / num_ideas
                    my_bar.progress(progress, text=progress_text)
                
                st.session_state.generated_ideas = generated_ideas
                st.write("Generated Ideas:")
                
                ideas_df = pd.DataFrame(generated_ideas, columns=["Ideas"])
                with st.expander("View Generated Ideas", expanded=True):
                    st.dataframe(ideas_df, use_container_width=True, hide_index=True)  # Display ideas without index column
            except Exception as e:
                st.error(f"An error occurred during idea generation for topic '{topic}' with {num_ideas} ideas: {str(e)}")

with tab2:
    st.subheader("Idea Evaluation")
    if not st.session_state.get("generated_ideas"):
        st.warning("Please brainstorm ideas first.")
    else:
        st.write("Fractl Finetuned Model's Brainstorming Ideas:")
        
        ideas_df = pd.DataFrame(st.session_state.generated_ideas, columns=["Generated Ideas"])
        with st.expander("View Generated Ideas", expanded=True):
            st.dataframe(ideas_df, use_container_width=True,  hide_index=True)  # Display ideas without index column
        
        claude_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        selected_model = st.selectbox("Select Claude Model", options=claude_models)
        
        evaluate_button = st.button("Evaluate and Refine Ideas")
        if evaluate_button:
            with st.spinner("Evaluating and refining ideas..."):
                refined_ideas = evaluate_ideas(st.session_state.generated_ideas, num_ideas, selected_model)
            st.session_state.refined_ideas = refined_ideas
            st.write("Refined Ideas:")
            
            refined_ideas_df = pd.DataFrame({'Refined Idea': refined_ideas})
            st.dataframe(refined_ideas_df, use_container_width=True,  hide_index=True)  # Display refined ideas without index column
with tab3:
    st.subheader("Idea Selection")
    if not st.session_state.get("refined_ideas"):
        st.warning("Please evaluate ideas first.")
    else:
        st.write("Refined Ideas:")
        
        refined_ideas_df = pd.DataFrame({'Refined Ideas': st.session_state.refined_ideas})
        refined_ideas_titles = refined_ideas_df['Refined Ideas'].apply(lambda x: x.split("\n")[0].replace("Title: ", ""))
        
        selected_titles = st.multiselect("Select ideas for brief generation", options=refined_ideas_titles)
        brief_button = st.button("Generate Idea Briefs")
        
        if brief_button:
            if not selected_titles:
                st.warning("Please select at least one idea.")
            else:
                selected_idea_blocks = [idea for title, idea in zip(refined_ideas_titles, st.session_state.refined_ideas) if title in selected_titles]
                with st.spinner("Generating idea briefs..."):
                    idea_briefs = generate_briefs(selected_idea_blocks, selected_model)
                st.session_state.idea_briefs = idea_briefs
                st.write("Idea Briefs:")
                for brief in idea_briefs:
                    st.write(brief)

with tab4:
    st.subheader("Final Results")
    if not st.session_state.get("idea_briefs"):
        st.warning("Please generate idea briefs first.")
    else:
        idea_briefs_df = pd.DataFrame.from_records(st.session_state.idea_briefs)
        st.dataframe(idea_briefs_df[['title', 'description', 'methodology']], use_container_width=True,  hide_index=True)  # Display idea briefs without index column
        
        for i, brief in enumerate(st.session_state.idea_briefs, start=1):
            with st.expander(f"{i}. {brief['title']}"):
                st.write(f"**Description:** {brief['description']}")
                st.write(f"**Justification:** {brief['justification']}")
                st.write(f"**Methodology:** {brief['methodology']}")
                st.write(f"**Datasets/Sources:** {brief['datasets_sources']}")
        
        export_button = st.button("Export All Idea Briefs as PDF")
        if export_button:
            export_briefs(st.session_state.idea_briefs)
