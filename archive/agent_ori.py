import operator
import streamlit as st
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from plantuml import PlantUML # https://github.com/vgilabert94/streamlit-image-zoom/tree/main

MODEL = st.secrets.get("OPENAI_MODEL")
API_KEY = st.secrets.get("OPENAI_KEY")


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
    """
    Data model for storing analysis of multiple user stories.
    Attributes:
        contradictions (str): Feedback on any contradictions between the user stories.
        inconsistencies (str): Feedback on any inconsistencies among the user stories.
        suggestions (str): Suggestions for improving the coherence of the user stories.
    """
    contradictions: str = Field(..., description="Feedback on any contradictions between the user stories.")
    inconsistencies: str = Field(..., description="Feedback on any inconsistencies among the user stories.")
    suggestions: str = Field(..., description="Suggestions for improving the coherence of the user stories.")
    user_stories_indexes_to_fix: str = Field(..., description="Indexes of user stories that need to be fixed.")

class FixedUserStory(BaseModel):
    fixed_user_story: str = Field(..., description="An improved version of the user story.")
    fixed_user_story_index: int = Field(..., description="Index of the user story that was fixed.")

class FixUserStoriesResponse(BaseModel):
    """
    Data model for storing fixed user stories.
    Attributes:
        fixed_user_stories (list): List of improved user stories.
    """
    fixed_user_stories: list[FixedUserStory] = Field(..., description="List of improved user stories.")

class PlantUMLResponse(BaseModel):
    """
    Data model for storing PlantUML code.
    Attributes:
        plantuml_code (str): The generated PlantUML code.
    """
    plantuml_code: str = Field(..., description="The generated PlantUML code.")

class State(MessagesState):
    action: str = Field(default="", description="The action to be performed.")
    user_stories: list[str] = Field(default=[], description="List of user stories.")
    user_stories_analysis: Annotated[list[str], operator.add]
    general_user_stories_analysis: GeneralUserStoriesAnalysis = Field(default=None, description="Analysis of multiple user stories.")
    plantuml_code: str = Field(default="", description="The PlantUML code representing the flowchart.")
    

class UserStoryState(TypedDict):
    user_story: str

def router(state: State):
    return state["action"]

def preprocess_user_stories(state: State):
    user_stories = state["user_stories"]
    user_stories = [s for s in user_stories if s.strip()]
    return {"user_stories": user_stories}

def analyze_user_story(state: UserStoryState):
    system_msg = "You are an expert in software development and user stories. Analyze the following user story and provide feedback on its clarity, completeness, and potential improvements."
    response = llm.with_structured_output(UserStoryAnalysis).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=state["user_story"])])
    user_stories_analysis = response.model_dump()
    return {"user_stories_analysis": [user_stories_analysis]}

def analyze_user_stories_in_parallel(state: State):
    return [Send("analyze_user_story", {"user_story": s}) for s in state["user_stories"]]

def execute_user_stories_analysis(state: State):
    user_stories = state["user_stories"]
    system_msg = """ You are an expert in software development and user stories.
                     Analyze the following user stories and revisa if no se contradicen entre ellas. o si hay alguna que no tenga sentido en el contexto de las otras."
    """
    user_stories_str = ""
    for u_i, user_story in enumerate(user_stories):
        user_stories_str += f"- {u_i+1}: {user_story} \n"
    response = llm.with_structured_output(GeneralUserStoriesAnalysis).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return {"general_user_stories_analysis": response_dict}

def fix_user_stories_contradictions(state: State):
    user_stories_to_fix = state["user_stories"]
    contradictions = state["general_user_stories_analysis"]["contradictions"]
    inconsistencies = state["general_user_stories_analysis"]["inconsistencies"]
    suggestions = state["general_user_stories_analysis"]["suggestions"]

    system_msg = (
        "You are an expert in software development and user stories.\n"
        "You will receive a set of user stories that have been found to contain contradictions or inconsistencies.\n"
        "Below are the details:\n"
        f"- Contradictions: {contradictions}\n"
        f"- Inconsistencies: {inconsistencies}\n"
        f"- Suggestions: {suggestions}\n"
        "Your task:\n"
        "1. Carefully review the user stories listed below.\n"
        "2. Fix any contradictions and inconsistencies, using the context above.\n"
        "3. Provide improved versions of the user stories that are coherent and consistent with each other.\n"
        "4. Return only the improved user stories in a list."
    )
    user_stories_str = "\n".join(user_stories_to_fix)
    response = llm.with_structured_output(FixUserStoriesResponse).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return response_dict

def create_plantuml_code(state: State):
    system_msg = "You are an expert in software development and user stories. Create a PlantUML diagram that represents the relationships and interactions between the following user stories."
    user_stories_str = ""
    for u_i, user_story in enumerate(state["user_stories"]):
        user_stories_str += f"- {u_i+1}: {user_story} \n"
    response = llm.with_structured_output(PlantUMLResponse).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return response_dict

def invoke(state,thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = graph.invoke(state, config=config)
    return response


llm = ChatOpenAI(model=MODEL, api_key=API_KEY, temperature=1)
plantuml_server = PlantUML(url="http://www.plantuml.com/plantuml/img/") # Public PlantUML server (you can also host your own)


# Build the graph directly
builder = StateGraph(State)
builder.add_node("preprocess_user_stories", preprocess_user_stories)
builder.add_node("analyze_user_story", analyze_user_story)
builder.add_node("execute_user_stories_analysis", execute_user_stories_analysis)
builder.add_node("fix_user_stories_contradictions", fix_user_stories_contradictions)
builder.add_node("create_plantuml_code", create_plantuml_code)

builder.add_conditional_edges(START, router , {"analyze_user_stories_in_parallel": "preprocess_user_stories",
                                               "execute_user_stories_analysis": "execute_user_stories_analysis",
                                               "fix_user_stories_contradictions": "fix_user_stories_contradictions",
                                               "create_plantuml_code": "create_plantuml_code"})

builder.add_conditional_edges("preprocess_user_stories", analyze_user_stories_in_parallel, ["analyze_user_story"])

graph = builder.compile()
graph_image = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)

file_path = "dataset/HOS/g01-us/g01-us.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

user_stories = [f"As {parte.strip()}" for parte in content.split("As") if parte.strip()]
user_stories = [user_story.replace('\n', ' ') for user_story in user_stories]
user_stories = user_stories[:3]  

state = {"action": "analyze_user_stories_in_parallel", "user_stories": user_stories}
response = invoke(state)
user_stories_analysis = response["user_stories_analysis"]
print("User Stories Analysis Results:")
for user_story_analysis in user_stories_analysis:
    for key, value in user_story_analysis.items():
        print(f"{key}: {value}")

user_stories = []
for user_story_analysis in user_stories_analysis:
    user_stories.append(user_story_analysis["user_story_improved"])
state = {"action": "execute_user_stories_analysis", "user_stories": user_stories}
response = invoke(state)
general_user_stories_analysis = response["general_user_stories_analysis"]
print("\nGeneral User Stories Analysis Results:")
for key, value in general_user_stories_analysis.items():
    print(f"{key}: {value}")