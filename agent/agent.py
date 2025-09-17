import os
import operator
import streamlit as st
from typing_extensions import Annotated
from pydantic import BaseModel, field_validator
from typing import List, Optional, Literal, TypedDict
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.types import Send

## GROQ
GROQ_MODEL = st.secrets.get("GROQ_MODEL")
GROQ_KEY = st.secrets.get("GROQ_KEY")
llm = ChatGroq(api_key=GROQ_KEY,model_name=GROQ_MODEL,temperature=1)

## OPENAI
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL")
OPENAIKEY = st.secrets.get("OPENAI_KEY")
llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAIKEY, temperature=0.7)

class UserStoryAnalysis(BaseModel):
    """
    Data model for storing user story analysis information.
    Attributes:
        clarity (str): Feedback on the clarity of the user story.
        completeness (str): Feedback on the completeness of the user story.
        improvements (str): Suggestions for improving the user story.
    """
    clarity: str = Field(..., description="Feedback on the clarity of the user story.")
    completeness: str = Field(..., description="Feedback on the completeness of the user story.")
    improvements: str = Field(..., description="Suggestions for improving the user story.")
    user_story_text: str = Field(..., description="An improved version of the user story.")
    user_story_index: str = Field(..., description="Index of the user story that was fixed.")

class GeneralUserStoriesAnalysis(BaseModel):
    overall_consistency: str = Field(
        ..., 
        description="A general evaluation of how well the user stories align and maintain coherence across the set."
    )
    contradictions: str = Field(
        description="Detected conflicts between user stories, indicating where requirements may clash."
    )
    contradictions_indexes: List[str] = Field(
        default_factory=list,
        description="Indexes of the user stories that contain contradictions. Examples: ['u_s_001', 'u_s_005']",
    )
    contextual_issues: str = Field(
        description="Stories that seem out of place or do not fit the overall context or product vision."
    )
    contextual_issues_indexes: List[str] = Field(
        default_factory=list,
        description="Indexes of the user stories that have contextual issues. Examples: ['u_s_002', 'u_s_007']",
    )
    value_assessment: str = Field(
        ..., 
        description="Evaluation of whether the stories provide clear and meaningful value to users or the business."
    )
    clarity_assessment: str = Field(
        ..., 
        description="Assessment of the clarity and understandability of the stories, highlighting ambiguous terms."
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Positive aspects of the stories as a whole, such as coverage of roles or user focus."
    )
    general_recommendations: Optional[List[str]] = Field(
        default_factory=list,
        description="High-level suggestions for improvement to enhance consistency, clarity, or value."
    )


class UserStory(BaseModel):
    text: str = Field(..., description="An improved version of the user story.")
    index: str = Field(..., description="Index of the user story that was fixed.")

class FixUserStoriesResponse(BaseModel):
    """
    Data model for storing fixed user stories.
    Attributes:
        fixed_user_stories (list): List of improved user stories.
    """
    fixed_user_stories: list[UserStory] = Field(..., description="List of improved user stories.")

class PlantUMLResponse(BaseModel):
    """
    Data model for storing PlantUML code.
    """
    plantuml_code: str = Field(..., description="The generated PlantUML code.")

    @field_validator("plantuml_code")
    def must_be_valid_plantuml(cls, v):
        if not v.strip().startswith("@startuml") or not v.strip().endswith("@enduml"):
            raise ValueError("PlantUML code must start with @startuml and end with @enduml")
        return v

class State(TypedDict):
    action: str 
    user_stories: dict = Field(default=[], description="User stories.")
    user_stories_analysis: Annotated[list[str], operator.add]
    general_user_stories_analysis: GeneralUserStoriesAnalysis = Field(default=None, description="Analysis of multiple user stories.")
    plantuml_code: str = Field(default="", description="The PlantUML code representing the flowchart.")

class UserStoryState(TypedDict):
    user_story: str


# Define allowed actions as a Literal type
actions = Literal[
    "execute_parallel_user_stories_analysis",
    "execute_general_user_stories_analysis",
    "fix_contradictions_and_contextual_issues",
    "create_plantuml_code"
]
def query_router(state: State) -> actions:
    return state["action"]

def execute_parallel_user_stories_analysis(state: State):
    return {}

def analyze_user_stories_in_parallel(state: State):
    user_stories = state["user_stories"]
    return [Send("analyze_user_story",  { "user_story": f"{index}: {value}"} ) for index, value in user_stories.items()]

def analyze_user_story(state: UserStoryState):
    system_msg = """You are an expert in software development and user stories.
    Analyze the following user story and provide feedback on its clarity, completeness, and potential improvements.
    Return your answer as a valid JSON object matching the following schema, with all fields properly quoted and closed:
    {{
        "clarity": "...",
        "completeness": "...",
        "improvements": "...",
        "user_story_text": "...",
        "user_story_index": "..."
    }}
    User story:
    """
    response = llm.with_structured_output(UserStoryAnalysis).invoke([SystemMessage(content=system_msg)]+
                                                                    [HumanMessage(content=state["user_story"])])
    user_stories_analysis = response.model_dump()
    return {"user_stories_analysis": [user_stories_analysis]}

def execute_general_user_stories_analysis(state: State):
    user_stories = state["user_stories"]
    system_msg = """
    You are an expert in software development and user stories.
    Analyze the provided user stories as a whole and provide a general assessment.

    Focus on:
    1. Overall consistency: Do the stories align with a common goal or vision?
    2. Contradictions: Are there any conflicts between the stories?
    3. Contextual coherence: Does any story feel out of place compared to the others?
    4. Value: Do all stories contribute clear value to the user or business?
    5. Clarity: Are the stories understandable and free of major ambiguities?

    Give a concise summary highlighting strengths and potential issues of the set of stories.

    User Stories:
    """
    user_stories_str = ""
    for index, value in user_stories.items():
        user_stories_str += f"{index}: {value} \n"

    response = llm.with_structured_output(GeneralUserStoriesAnalysis).invoke([SystemMessage(content=system_msg)]+
                                                                             [HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return {"general_user_stories_analysis": response_dict}

def fix_contradictions_and_contextual_issues(state: State):
    system_msg = """
    You are an expert in software development and user stories.
    You will receive a set of user stories that contain contradictions or contextual inconsistencies.
    Here are the details identified from analysis:
    - Contradictions: {contradictions}
    - Contextual issues: {contextual_issues}
    "Your task:
    1. Carefully review the user stories listed below.
    2. Resolve any contradictions and contextual issues, keeping the overall product vision in mind.
    3. Rewrite and improve the user stories so they are clear, coherent, and consistent with each other.
    4. Maintain the original intent of each story while making them actionable.
    5. Return only the improved user stories as a clean list.

    User Stories to fix:
    """
    user_stories = state["user_stories"]
    contradictions = state["general_user_stories_analysis"]["contradictions"]
    contradictions_indexes = state["general_user_stories_analysis"]["contradictions_indexes"]
    contextual_issues = state["general_user_stories_analysis"]["contextual_issues"]
    contextual_issues_indexes = state["general_user_stories_analysis"]["contextual_issues_indexes"]
    
    indexes = contradictions_indexes + contextual_issues_indexes
    indexes = set(indexes)  # Remove duplicates
    
    user_stories_to_fix_str = ""
    for index, value in user_stories.items():
        if index in indexes:
            user_stories_to_fix_str += f"{index}: {value} \n"
    
    system_msg = system_msg.format(contradictions=contradictions, contextual_issues=contextual_issues)
    response = llm.with_structured_output(FixUserStoriesResponse).invoke([SystemMessage(content=system_msg)]+
                                                                         [HumanMessage(content=user_stories_to_fix_str)])
    response_dict = response.model_dump()
    fixed_user_stories = response_dict["fixed_user_stories"]
    fixed_user_stories_dict = {fixed_user_story_data["index"]:fixed_user_story_data["text"] for fixed_user_story_data in fixed_user_stories}

    # update user stories:
    for index, value in user_stories.items():
        if index in fixed_user_stories_dict:
            user_stories[index] = value
    return {"user_stories":user_stories}


def create_plantuml_code(state: State):
    """
    Generate PlantUML class diagram code based on user stories.
    """
    system_msg = """
    You are an expert in software architecture and UML. 
    Your task is to generate a **PlantUML Class Diagram** from the following user stories.

    Rules:
    1. **Diagram wrapper**: Always wrap the output between @startuml ... @enduml.
    2. **Classes**:
    - Represent each main actor, role, system, or entity as a class.
    - Extract **attributes** (as fields) and **operations** (as methods) when possible from the stories.
    - Use **<<stereotypes>>** to indicate the type:
        - <<actor>> : human role (green)
        - <<system>> : IT system / software (yellow)
        - <<data>> : database, file, or record (orange)
        - <<external>> : external system or regulation (red)
    3. **Relationships**:
    - Use **generalization (<|--)** for inheritance/specialization.
    - Use **associations (-- or -->)** for interactions, labeling with the action/verb if possible (e.g., "creates", "views", "updates").
    - Indicate **multiplicity/cardinality** (e.g., 1, 0..*, 1..*) if suggested by the stories.
    4. **Styling**:
    - Apply stereotype-based colors:
        - <<actor>> : green
        - <<system>> : yellow
        - <<data>> : orange
        - <<external>> : red
    - Ensure all text is readable with proper spacing.
    5. **Validation**:
    - Ensure the diagram is **syntactically valid PlantUML**.
    - The diagram must be complete enough to visualize the system as described.

    Goal: The output should be close to a professional UML class diagram (like one used in software design docs), not just a list of classes.
    User Stories:
    """
    user_stories = state["user_stories"]
    user_stories_str = ""
    for index, value in user_stories.items():
        user_stories_str += f"{index}: {value} \n"
    response = llm.with_structured_output(PlantUMLResponse).invoke([SystemMessage(content=system_msg)]+
                                                                   [HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    plantuml_code = response_dict["plantuml_code"]
    return {"plantuml_code":plantuml_code}

def invoke(state, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = graph.invoke(state, config=config)
    return response


# Build the graph directly
builder = StateGraph(State)
builder.add_node("execute_parallel_user_stories_analysis",execute_parallel_user_stories_analysis)
builder.add_node("analyze_user_story", analyze_user_story)
builder.add_node("execute_general_user_stories_analysis", execute_general_user_stories_analysis)
builder.add_node("fix_contradictions_and_contextual_issues", fix_contradictions_and_contextual_issues)
builder.add_node("create_plantuml_code",create_plantuml_code)

builder.add_conditional_edges(START,
                               query_router,
                               {
                                   "execute_parallel_user_stories_analysis": "execute_parallel_user_stories_analysis",
                                   "execute_general_user_stories_analysis":"execute_general_user_stories_analysis",
                                   "fix_contradictions_and_contextual_issues":"fix_contradictions_and_contextual_issues",
                                   "create_plantuml_code":"create_plantuml_code"
                               }
                               )

builder.add_conditional_edges("execute_parallel_user_stories_analysis", analyze_user_stories_in_parallel, ["analyze_user_story"])
# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()
# Compile the graph with the checkpointer 
graph = builder.compile(checkpointer=within_thread_memory)

#graph_image = graph.get_graph(xray=True).draw_mermaid_png()
#with open("graph.png", "wb") as f:
#    f.write(graph_image)

if __name__ == "__main__":
    from plantuml import PlantUML # https://github.com/vgilabert94/streamlit-image-zoom/tree/main
    plantuml_server = PlantUML(url="http://www.plantuml.com/plantuml/img/") # Public PlantUML server (you can also host your own)
    out_folder = "out"
    # Create folder if it does not exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    diagram_path = f"{out_folder}/flow_chart.png"
    thread_id = "123"

    file_path = "dataset/HOS/g01-us/g01-us.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    user_stories = [f"As {parte.strip()}" for parte in content.split("As") if parte.strip()]
    user_stories = [user_story.replace('\n', ' ') for user_story in user_stories]
    user_stories = { f"u_s_{str(i).zfill(3)}": user_story  for i, user_story in enumerate(user_stories)}
    user_stories = dict(list(user_stories.items())[:10])

    action = "execute_parallel_user_stories_analysis"
    state = {"action":action,
            "user_stories": user_stories}

    response = invoke(state=state, thread_id=thread_id)
    #print(response)
    user_stories_analysis = response["user_stories_analysis"]
    user_stories_improved = {}
    for user_story_analysis in user_stories_analysis:
        clarity = user_story_analysis["clarity"]
        completeness = user_story_analysis["completeness"]
        improvements = user_story_analysis["improvements"]
        user_story_text = user_story_analysis["user_story_text"]
        user_story_index = user_story_analysis["user_story_index"]
        user_stories_improved[user_story_index] = user_story_text
    print(user_stories_analysis)
    user_message = input("Accept suggested user stories? ")
    if user_message == "yes":
        user_stories = user_stories_improved

    action = "execute_general_user_stories_analysis"
    state = {"action":action,
            "user_stories": user_stories}
    response = invoke(state=state, thread_id=thread_id)
    general_user_stories_analysis = response["general_user_stories_analysis"]
    print(general_user_stories_analysis)

    user_message = input("Fix user stories? ")
    if user_message == "yes":
        action = "fix_contradictions_and_contextual_issues"
        state = {"action":action}
        response = invoke(state=state, thread_id=thread_id)
        print(response["user_stories"])


    action = "create_plantuml_code"
    state = {"action":action}
    response = invoke(state=state, thread_id=thread_id)
    plantuml_code = response["plantuml_code"]

    with open(diagram_path, "wb") as f:
        f.write(plantuml_server.processes(plantuml_code))
