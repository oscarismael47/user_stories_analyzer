import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

class UserStoriesAnalysis(BaseModel):
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

def analyze_user_story(user_story):
    system_msg = "You are an expert in software development and user stories. Analyze the following user story and provide feedback on its clarity, completeness, and potential improvements."
    response = llm.with_structured_output(UserStoryAnalysis).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_story)])
    response_dict = response.model_dump()
    return response_dict

def analyze_user_stories(user_stories):
    system_msg = """ You are an expert in software development and user stories.
                     Analyze the following user stories and revisa if no se contradicen entre ellas. o si hay alguna que no tenga sentido en el contexto de las otras."
    """
    user_stories_str = ""
    for u_i, user_story in enumerate(user_stories):
        user_stories_str += f"- {u_i+1}: {user_story} \n"
    response = llm.with_structured_output(UserStoriesAnalysis).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return response_dict

def fix_user_stories_contradictions(user_stories_to_fix, contradictions, inconsistencies, suggestions):
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

def create_plantuml_code(user_stories):
    system_msg = "You are an expert in software development and user stories. Create a PlantUML diagram that represents the relationships and interactions between the following user stories."
    user_stories_str = ""
    for u_i, user_story in enumerate(user_stories):
        user_stories_str += f"- {u_i+1}: {user_story} \n"
    response = llm.with_structured_output(PlantUMLResponse).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=user_stories_str)])
    response_dict = response.model_dump()
    return response_dict

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, temperature=1)
plantuml_server = PlantUML(url="http://www.plantuml.com/plantuml/img/") # Public PlantUML server (you can also host your own)

file_path = "dataset/HOS/g01-us/g01-us.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

user_stories = [f"As {parte.strip()}" for parte in content.split("As") if parte.strip()]
user_stories = [user_story.replace('\n', ' ') for user_story in user_stories]

# Save to txt file
with open("output.txt", "w", encoding="utf-8") as f:
    for line in user_stories:
        f.write(line + "\n")


"""

user_stories = user_stories[:10]  

#for user_story in user_stories:
#    print(f"{user_story}")
#    user_story_analysis = analyze_user_story(user_story)
#    print(user_story_analysis)

user_stories_str = "\n".join(user_stories)
user_stories_analysis = analyze_user_stories(user_stories)
contradictions = user_stories_analysis.get("contradictions", "")
inconsistencies = user_stories_analysis.get("inconsistencies", "")
suggestions = user_stories_analysis.get("suggestions", "")
user_stories_to_fix_indexes = user_stories_analysis.get("user_stories_indexes_to_fix", "")

user_stories_to_fix = []
for index in user_stories_to_fix_indexes.split(","):
    try:
        idx = int(index.strip()) - 1
        if 0 <= idx < len(user_stories):
            user_stories_to_fix.append(f"- {idx+1}: {user_stories[idx]}")
    except ValueError:
        continue

response = fix_user_stories_contradictions(user_stories_to_fix, contradictions, inconsistencies, suggestions)
user_stories_final = user_stories.copy()
for fixed_user_story in response.get("fixed_user_stories", []):
    user_stories_final[fixed_user_story['fixed_user_story_index']] = fixed_user_story['fixed_user_story']

print(user_stories_final)

plantuml_response = create_plantuml_code(user_stories_final)
plantuml_code = plantuml_response.get("plantuml_code", "")
print(plantuml_code)

with open("diagram.png", "wb") as f:
    f.write(plantuml_server.processes(plantuml_code))
    st.session_state.image_path = "diagram.png"

"""