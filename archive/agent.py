import operator
import streamlit as st
from typing_extensions import Annotated
from pydantic import BaseModel
from typing import List, Optional, Literal, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from plantuml import PlantUML # https://github.com/vgilabert94/streamlit-image-zoom/tree/main

GROQ_MODEL = st.secrets.get("GROQ_MODEL")
GROQ_KEY = st.secrets.get("GROQ_KEY")


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
    user_story_improved: str = Field(..., description="An improved version of the user story.")

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
    Attributes:
        plantuml_code (str): The generated PlantUML code.
    """
    plantuml_code: str = Field(..., description="The generated PlantUML code.")

class State(TypedDict):
    user_stories: list[str] = Field(default=[], description="List of user stories.")
    user_stories_analysis: Annotated[list[str], operator.add]
    general_user_stories_analysis: GeneralUserStoriesAnalysis = Field(default=None, description="Analysis of multiple user stories.")
    user_stories_analyze_decision : str
    plantuml_code: str = Field(default="", description="The PlantUML code representing the flowchart.")

class UserStoryState(TypedDict):
    user_story: str


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
    for user_story in user_stories:
        for index, value in user_story.items():
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
    for user_story in user_stories:
        for index, value in user_story.items():
            if index in indexes:
                user_stories_to_fix_str += f"{index}: {value} \n"
    
    system_msg = system_msg.format(contradictions=contradictions, contextual_issues=contextual_issues)
    response = llm.with_structured_output(FixUserStoriesResponse).invoke([SystemMessage(content=system_msg)]+
                                                                         [HumanMessage(content=user_stories_to_fix_str)])
    response_dict = response.model_dump()
    fixed_user_stories = response_dict["fixed_user_stories"]
    fixed_user_stories_dict = {fixed_user_story_data["index"]:fixed_user_story_data["text"] for fixed_user_story_data in fixed_user_stories}

    # update user stories:
    for user_story in user_stories:
        for index, value in user_story.items():
            if index in indexes:
                user_stories[index] = value

    return {"user_stories":user_stories}


def user_stories_analyze_human_approval(state: State) -> Command[Literal["execute_parallel_user_stories_analysis",
                                                                          "create_plantuml_code"]] :
    response = interrupt(
        {
            "question": {
                "text": "Do you want yo analyze individually user stories?",
                "options": ("YES", "NO")
            }
        }
    )
    if user_stories_analyze_decision == "YES":
        return Command(goto="execute_parallel_user_stories_analysis",
                       update={"user_stories_analyze_decision":user_stories_analyze_decision})
    elif user_stories_analyze_decision == "NO":
        return Command(goto="create_plantuml_code",
                        update={"user_stories_analyze_decision":user_stories_analyze_decision})
    else:
        raise ValueError(f"Unknown response VALUE: {user_stories_analyze_decision}")

def execute_parallel_user_stories_analysis(state:State):
    return {}

def analyze_user_story(state: UserStoryState):
    system_msg = "You are an expert in software development and user stories. Analyze the following user story and provide feedback on its clarity, completeness, and potential improvements."
    response = llm.with_structured_output(UserStoryAnalysis).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=state["user_story"])])
    user_stories_analysis = response.model_dump()
    return {"user_stories_analysis": [user_stories_analysis]}

def analyze_user_stories_in_parallel(state: State):
    return [Send("analyze_user_story", {"user_story": s}) for s in state["user_stories"]]

def create_plantuml_code(state: State):
    system_msg = "You are an expert in software development and user stories. Create a PlantUML diagram that represents the relationships and interactions between the following user stories."
    user_stories_str = ""
    for u_i, user_story in enumerate(state["user_stories"]):
        user_stories_str += f"- {u_i+1}: {user_story} \n"
    response = llm.with_structured_output(PlantUMLResponse).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return response_dict

def invoke(state):
    response = graph.invoke(state)
    return response


# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name=GROQ_MODEL,
    temperature=0.7
    )
plantuml_server = PlantUML(url="http://www.plantuml.com/plantuml/img/") # Public PlantUML server (you can also host your own)


# Build the graph directly
builder = StateGraph(State)
builder.add_node("execute_general_user_stories_analysis", execute_general_user_stories_analysis)
builder.add_node("fix_contradictions_and_contextual_issues", fix_contradictions_and_contextual_issues)
builder.add_node("execute_individual_user_stories_analysis",user_stories_analyze_human_approval)
builder.add_node("execute_parallel_user_stories_analysis",execute_parallel_user_stories_analysis)
builder.add_node("analyze_user_story", analyze_user_story)
builder.add_node("create_plantuml_code",create_plantuml_code)

builder.add_edge(START, "execute_general_user_stories_analysis")
builder.add_edge("execute_general_user_stories_analysis", "fix_contradictions_and_contextual_issues")
builder.add_edge("fix_contradictions_and_contextual_issues", "execute_individual_user_stories_analysis")
builder.add_conditional_edges("execute_parallel_user_stories_analysis", analyze_user_stories_in_parallel, ["analyze_user_story"])
builder.add_edge("analyze_user_story","create_plantuml_code")


graph = builder.compile()
graph_image = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)

file_path = "dataset/HOS/g01-us/g01-us.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

user_stories = [f"As {parte.strip()}" for parte in content.split("As") if parte.strip()]
user_stories = [user_story.replace('\n', ' ') for user_story in user_stories]
user_stories = [{f"u_s_{str(i).zfill(3)}": user_story}  for i, user_story in enumerate(user_stories)]
user_stories = user_stories[:10]  

"""
state = {"user_stories": user_stories}
response = invoke(state)
print("\ngeneral_user_stories_analysis:")
for key, value in response["general_user_stories_analysis"].items():
    print(f"{key}: {value}")
"""