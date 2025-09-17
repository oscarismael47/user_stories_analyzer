import os 
import time
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd
import uuid
from copy import copy
from agent.agent import invoke 
from file_helper import generate_user_stories_report
from plantuml import PlantUML # https://github.com/vgilabert94/streamlit-image-zoom/tree/main

def highlight_rows(row):
    if row.name in  st.session_state.contradictions_indexes:
        return ['background-color: lightblue'] * len(row)
    elif row.name in  st.session_state.contextual_issues_indexes:
        return ['background-color: yellow'] * len(row)
    else:
        return [''] * len(row)

# Set page configuration
st.set_page_config(
    page_title="My App",
    page_icon="📊",
    layout="wide",   # options: "centered" (default) or "wide"
    initial_sidebar_state="expanded"  # or "collapsed"
)

if "plantuml_server" not in st.session_state:
    st.session_state.plantuml_server = PlantUML(url="http://www.plantuml.com/plantuml/img/") # Public PlantUML server (you can also host your own)

if 'step' not in st.session_state:
    st.session_state.step = 0

if 'user_stories_dict' not in st.session_state:
    st.session_state.user_stories_dict = {}

if 'user_stories_df' not in st.session_state:
    st.session_state.user_stories_df = pd.DataFrame(columns=["User Story"])

if 'user_stories_processed_dict' not in st.session_state:
    st.session_state.user_stories_processed_dict = {}

if 'user_stories_processed_df' not in st.session_state:
    st.session_state.user_stories_processed_df = pd.DataFrame(columns=["Original User Story", "New User Story", "Explanation"])

if "proccess_id" not in st.session_state:
    st.session_state.proccess_id = "123" #str(uuid.uuid4())

out_folder = "out"
diagram_path = f"{out_folder}/flow_chart.png"
report_path = f"{out_folder}/user_stories_report.pdf"
 

# Create folder if it does not exist
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


with st.sidebar:
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "xlsx"])

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # Read TXT file
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            user_stories = [f"As {parte.strip()}" for parte in content.split("As") if parte.strip()]
            user_stories = [user_story.replace('\n', ' ') for user_story in user_stories]
            st.session_state.user_stories_dict  = { f"u_s_{str(i).zfill(3)}": user_story  for i, user_story in enumerate(user_stories)}
            st.session_state.original_user_stories_dict = copy(st.session_state.user_stories_dict)
            st.session_state.user_stories_df = pd.DataFrame.from_dict(st.session_state.user_stories_dict, 
                                                                      orient="index", columns=["User Story"])
            
if st.session_state.step == 0:
    st.subheader("**User Stories**")
    st.dataframe(st.session_state.user_stories_df)

    with st.container(horizontal=True, horizontal_alignment="right"):
        analyze_user_stories_btn = st.button("Analyze User Stories Individually")
        analyze_user_stories_in_group_btn = st.button("Analyze User Stories Together (Detect Issues)")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if analyze_user_stories_btn:
            if st.session_state.user_stories_df.empty:
                st.error("Please upload a file with user stories first.")
            else:
                with st.spinner("Analyzing User Stories Individually ...", show_time=True):
                    action = "execute_parallel_user_stories_analysis"
                    state = {"action":action,
                            "user_stories": st.session_state.user_stories_dict}
                    response = invoke(state=state, thread_id=st.session_state.proccess_id)
                    st.session_state.user_stories_analysis = response["user_stories_analysis"]            
                    st.session_state.step = 1
                    st.rerun()
        
        if analyze_user_stories_in_group_btn:
            if st.session_state.user_stories_df.empty:
                st.error("Please upload a file with user stories first.")
            else:
                with st.spinner("Executing General Analyzis...", show_time=True):
                    action = "execute_general_user_stories_analysis"
                    state = {"action":action,
                            "user_stories": st.session_state.user_stories_dict}
                    response = invoke(state=state, thread_id=st.session_state.proccess_id)
                    st.session_state.general_user_stories_analysis = response["general_user_stories_analysis"]
                    st.session_state.step = 3
                    st.rerun()

            
if st.session_state.step == 1:
    
    for user_story_analysis in st.session_state.user_stories_analysis:
        clarity = user_story_analysis["clarity"]
        completeness = user_story_analysis["completeness"]
        improvements = user_story_analysis["improvements"]
        explanation = f"Clarity:{clarity}\n\nCompleteness:{completeness}\n\nImprovements:{improvements}"

        user_story_text = user_story_analysis["user_story_text"]
        user_story_index = user_story_analysis["user_story_index"]
        user_story_text_ori = st.session_state.user_stories_dict[user_story_index]
        st.session_state.user_stories_processed_dict[user_story_index] = {"original_user_story":user_story_text_ori,
                                                                        "new_user_story":user_story_text,
                                                                          "explanation":explanation}
    st.session_state.user_stories_processed_df = pd.DataFrame.from_dict(st.session_state.user_stories_processed_dict,
                                                                        orient="index")
    st.session_state.user_stories_processed_df = st.session_state.user_stories_processed_df.rename(columns={
    "original_user_story": "Original User Story",
    "new_user_story": "New User Story",
    "explanation": "Explanation"
})
    st.subheader("**User Stories**")  
    st.dataframe(st.session_state.user_stories_processed_df)

    with st.container(horizontal=True, horizontal_alignment="right"):
        use_new_user_stories_btn = st.button("Apply Improved User Stories")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if use_new_user_stories_btn:
            with st.spinner("Updating user stories.", show_time=True):
                for index, row in st.session_state.user_stories_processed_df.iterrows():
                    st.session_state.user_stories_dict[index] = row["New User Story"]
                st.session_state.step = 2
                st.rerun()

if st.session_state.step == 2:
    st.subheader("**User Stories**")
    st.dataframe(st.session_state.user_stories_df)

    with st.container(horizontal=True, horizontal_alignment="right"):
        analyze_user_stories_in_group_btn = st.button("Analyze User Stories Together (Detect Issues)")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if analyze_user_stories_in_group_btn:
            with st.spinner("Running General Analysis...", show_time=True):
                action = "execute_general_user_stories_analysis"
                state = {"action":action,
                        "user_stories": st.session_state.user_stories_dict}
                response = invoke(state=state, thread_id=st.session_state.proccess_id)
                st.session_state.general_user_stories_analysis = response["general_user_stories_analysis"]
                st.session_state.step = 3
                st.rerun()

if st.session_state.step == 3:
    
    st.session_state.overall_consistency = st.session_state.general_user_stories_analysis["overall_consistency"]
    st.session_state.contradictions = st.session_state.general_user_stories_analysis["contradictions"]
    st.session_state.contradictions_indexes = st.session_state.general_user_stories_analysis["contradictions_indexes"]
    st.session_state.contextual_issues = st.session_state.general_user_stories_analysis["contextual_issues"]
    st.session_state.contextual_issues_indexes = st.session_state.general_user_stories_analysis["contextual_issues_indexes"]   
    st.session_state.value_assessment = st.session_state.general_user_stories_analysis["value_assessment"]   
    st.session_state.clarity_assessment = st.session_state.general_user_stories_analysis["clarity_assessment"]   
    st.session_state.strengths = st.session_state.general_user_stories_analysis["strengths"]
    st.session_state.general_recommendations = st.session_state.general_user_stories_analysis["general_recommendations"]

    st.subheader("**User Stories**")
    st.dataframe(st.session_state.user_stories_df.style.apply(highlight_rows, axis=1))

    st.subheader("**User Stories Analysis**")    
    with st.container(border=True):
        
        st.markdown("**Overall Consistency**")
        st.markdown(st.session_state.overall_consistency)
        st.markdown(f"<span style='background-color: lightblue; font-weight: bold;'>Contradictions</span> [{len(st.session_state.contradictions_indexes)}]",
                    unsafe_allow_html=True)
        st.markdown(st.session_state.contradictions)
        st.markdown(f"<span style='background-color: yellow; font-weight: bold;'>Contextual Issues</span> [{len(st.session_state.contextual_issues_indexes)}]",
                    unsafe_allow_html=True)
        st.markdown(st.session_state.contextual_issues)
        st.markdown("**Value Assessment**")
        st.markdown(st.session_state.value_assessment)
        st.markdown("**Clarity Assessment**")
        st.markdown(st.session_state.clarity_assessment)
        st.markdown("**Strengths**")
        for strength in st.session_state.strengths:
            st.markdown(f"- ● {strength}")

        st.markdown("**General Recommendations**")
        for general_recommendation in st.session_state.general_recommendations:
            st.markdown(f"- ● {general_recommendation}")


    with st.container(horizontal=True, horizontal_alignment="right"):
        fix_issues_btn = st.button("Fix Contradictions and Contextual Issues")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if fix_issues_btn:
            with st.spinner("Fixing issues...", show_time=True):
                action = "fix_contradictions_and_contextual_issues"
                state = {"action":action}
                response = invoke(state=state, thread_id=st.session_state.proccess_id)
                st.session_state.user_stories_dict = response["user_stories"]
                st.session_state.user_stories_df = pd.DataFrame.from_dict(st.session_state.user_stories_dict, 
                                                                        orient="index", columns=["User Story"])
                st.session_state.step = 4
                st.rerun()

if st.session_state.step == 4:
    st.subheader("**Final User Stories**")
    st.dataframe(st.session_state.user_stories_df)

    with st.container(horizontal=True, horizontal_alignment="right"):
        generate_class_diagram_btn = st.button("Generate Class Diagram")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if generate_class_diagram_btn:
            with st.spinner("Generating Class Diagram...", show_time=True):
                action = "create_plantuml_code"
                state = {"action":action,
                        "user_stories": st.session_state.user_stories_dict}
                response = invoke(state=state, thread_id=st.session_state.proccess_id)
                plantuml_code = response["plantuml_code"]
                with open(diagram_path, "wb") as f:
                    f.write(st.session_state.plantuml_server.processes(plantuml_code))
                    st.session_state.diagram_path =  diagram_path
                st.session_state.step = 5
                st.rerun()

if st.session_state.step == 5:
    st.subheader("**Final User Stories**")
    st.dataframe(st.session_state.user_stories_df)
    st.subheader("**Class Diagram**")
    st.image(st.session_state.diagram_path)

    with st.container(horizontal=True, horizontal_alignment="right"):
        generate_results_report_btn = st.button("Generate Results Report")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if generate_results_report_btn:
            with st.spinner("Generating Results Report...", show_time=True): 
                generate_user_stories_report(st.session_state.original_user_stories_dict,
                                            st.session_state.user_stories_processed_dict,
                                            st.session_state.overall_consistency,
                                            st.session_state.contradictions,
                                            st.session_state.contradictions_indexes,
                                            st.session_state.contextual_issues,
                                            st.session_state.contextual_issues_indexes,
                                            st.session_state.value_assessment,
                                            st.session_state.clarity_assessment,
                                            st.session_state.strengths,
                                            st.session_state.general_recommendations,
                                            st.session_state.user_stories_dict,
                                            st.session_state.diagram_path,
                                            filename=report_path
                                            )
                st.session_state.step = 6
                st.rerun()


if st.session_state.step == 6:
    st.subheader("**Results Report**")
    with st.container(border=True,width="stretch"):
        pdf_viewer(
        report_path,
        width=1000,
        height=400,
        zoom_level=1.5,                    # 120% zoom
        viewer_align="center",             # Center alignment
        show_page_separator=True           # Show separators between pages
    )
    
    with st.container(horizontal=True, horizontal_alignment="right"):
        # Add download button for the PDF
        with open(report_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="user_stories_report.pdf",
                mime="application/pdf"
            )