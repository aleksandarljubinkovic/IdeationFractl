import streamlit as st
import openai
import anthropic
import pandas as pd
import re
import json

Opus = "claude-3-opus-20240229"
Sonnet = "claude-3-sonnet-20240229"
Haiku = "claude-3-haiku-20240307"


# Set up API keys and models
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai = openai.Client(api_key=OPENAI_API_KEY)

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
gpt_model = "ft:gpt-3.5-turbo-0125:personal:idea-generator:9DgQ5nsD"
claude_model = Haiku

# App title and description
st.set_page_config(page_title="Idea Generation and Refinement", layout="wide")
st.title("Idea Generation and Refinement")
st.write("Generate, evaluate, and refine ideas for your topic.")

# Sidebar navigation
current_step = st.sidebar.radio("Navigation", ["Idea Generation", "Idea Evaluation", "Idea Selection", "Final Results"])

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
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates creative ideas."},
                {"role": "user", "content": f"Generate {num_ideas} ideas for the topic: {topic}"}
            ],
            max_tokens=400,
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
def evaluate_ideas(generated_ideas: list, model: str) -> list:
    try:
        system_prompt = "As the NYTimes data journalism editor, your job is to find the 20 most newsworthy and interesting story ideas that are also viable to create without large technical hurdles from a possible from a corpus of brainstorm ideas, and then improve them, make them more concrete, and suggest methodologies that would make them practical. For each selection you also provide justification for your answer and why that idea deserves that position in your top ten. THe rubric you think about for evaluating ideas most frequently is the SUCCESs model of content stickiness made famous in Made to Stick by Chip and Dan Heath"
        prompt = f"""Select only 20 of the best and most viable ideas out of a corpus of ideas of varying quality. 
        Here are the brainstorm ideas to choose from: \n {generated_ideas} \n
        
        For each of the 20 selected ideas, please provide the following information in a structured format:
        
        Title: [Enhanced title for the idea]
        Description: [Detailed description of the idea, including the lede, newsworthy hooks, target audience, and why they should care]
        Justification: [Justification for selecting this idea]
        Methodology: [Methodology for producing this idea, including feasibility within a 2-week timeline]
        Datasets/Sources: [Datasets, sources, technologies, and tools needed to accomplish this idea]
        
        Please ensure that all the information for each idea is provided in a single block, separated by a blank line."""
        response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
            system=system_prompt,
            model=model,
            messages = [{"role": "user", "content": prompt }],
            max_tokens=4000,
        )
        refined_ideas_text = response.content[0].text
        refined_ideas = refined_ideas_text.split("\n\n")
        return refined_ideas
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
        idea_briefs = []
        for idea in selected_ideas:
            prompt = f"""Enhance and make more specific each part of each brief, especially providing sources for datasets needed for the idea and methodological guidance to help prevent roadblocks or timesucks.
            
            Idea: {idea}"""
            
            response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                model=model,
                messages = [{"role": "user", "content": prompt }],
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
            idea_briefs.append(idea_brief)
        
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
        from reportlab.lib.styles import getSampleStyleSheet

        styles = getSampleStyleSheet()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            doc = SimpleDocTemplate(tmp_file.name, pagesize=letter)
            elements = []

            for brief in idea_briefs:
                elements.append(Paragraph(brief["title"], styles["Heading1"]))
                elements.append(Paragraph(brief["description"], styles["BodyText"]))
                elements.append(Paragraph("Justification: " + brief["justification"], styles["BodyText"]))
                elements.append(Paragraph("Methodology: " + brief["methodology"], styles["BodyText"]))
                elements.append(Paragraph("Datasets/Sources: " + brief["datasets_sources"], styles["BodyText"]))
                elements.append(Spacer(1, 12))

            doc.build(elements)
            st.download_button(
                label="Download Idea Briefs",
                data=tmp_file.read(),
                file_name="idea_briefs.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.error(f"Error exporting idea briefs: {str(e)}")

if current_step == "Idea Generation":
    st.subheader("Idea Generation")
    topic = st.text_input("Enter a topic")
    num_ideas = st.number_input("Number of ideas to generate", min_value=1, max_value=100, value=10)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    generate_button = st.button("Generate Ideas")

    if generate_button:
        if not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating ideas..."):
                generated_ideas = get_ideas(topic, num_ideas, temperature, gpt_model)
                st.session_state.generated_ideas = generated_ideas
                st.write("Generated Ideas:")
                ideas_df = pd.DataFrame({"Idea": generated_ideas})
                st.table(ideas_df)

elif current_step == "Idea Evaluation":
    st.subheader("Idea Evaluation")
    if "generated_ideas" not in st.session_state:
        st.warning("Please generate ideas first.")
    else:
        generated_ideas = st.session_state.generated_ideas
        st.write("Generated Ideas:")
        ideas_df = pd.DataFrame({"Idea": generated_ideas})
        st.table(ideas_df)
        evaluate_button = st.button("Evaluate and Refine Ideas")
        if evaluate_button:
            with st.spinner("Evaluating and refining ideas..."):
                refined_ideas = evaluate_ideas(generated_ideas, claude_model)
                st.session_state.refined_ideas = refined_ideas
                st.write("Refined Ideas:")
                refined_ideas_data = [{'Refined Idea': idea} for idea in refined_ideas]
                refined_ideas_df = pd.DataFrame(refined_ideas_data)
                st.table(refined_ideas_df)


elif current_step == "Idea Selection":
    st.subheader("Idea Selection")
    if "refined_ideas" not in st.session_state:
        st.warning("Please evaluate ideas first.")
    else:
        refined_ideas = st.session_state.refined_ideas
        idea_titles = [idea.split("\n")[0].replace("Title: ", "") for idea in refined_ideas]
        selected_ideas = st.multiselect("Select ideas for brief generation", idea_titles)
        brief_button = st.button("Generate Idea Briefs")
        if brief_button:
            if not selected_ideas:
                st.warning("Please select at least one idea.")
            else:
                selected_idea_blocks = [idea for idea in refined_ideas if idea.split("\n")[0].replace("Title: ", "") in selected_ideas]
                with st.spinner("Generating idea briefs..."):
                    idea_briefs = generate_briefs(selected_idea_blocks, claude_model)
                    st.session_state.idea_briefs = idea_briefs
                    st.write("Idea Briefs:")
                    for brief in idea_briefs:
                        st.write(brief)


elif current_step == "Final Results":
    st.subheader("Final Results")
    if "idea_briefs" not in st.session_state:
        st.warning("Please generate idea briefs first.")
    else:
        idea_briefs = st.session_state.idea_briefs
        for brief in idea_briefs:
            with st.expander(brief["title"]):
                st.write(brief["description"])
                st.write("Justification: " + brief["justification"])
                st.write("Methodology: " + brief["methodology"])
                st.write("Datasets/Sources: " + brief["datasets_sources"])
        export_button = st.button("Export Idea Briefs")

        if export_button:
            export_briefs(idea_briefs)
