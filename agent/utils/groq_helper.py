import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel
from typing import List

GROQ_MODEL = st.secrets.get("GROQ_MODEL")
GROQ_KEY = st.secrets.get("GROQ_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name=GROQ_MODEL,
    temperature=0.7
)

class Response(BaseModel):
    name: str
    price: float
    features: List[str]
        
# Example usage
description = """The Kees Van Der Westen Speedster is a high-end, single-group espresso machine known for its precision, performance, 
and industrial design. Handcrafted in the Netherlands, it features dual boilers for brewing and steaming, PID temperature control for 
consistency, and a unique pre-infusion system to enhance flavor extraction. Designed for enthusiasts and professionals, it offers 
customizable aesthetics, exceptional thermal stability, and intuitive operation via a lever system. The pricing is approximatelyt $14,499 
depending on the retailer and customization options."""


system_msg = "Extract product details"
response = llm.with_structured_output(Response).invoke([SystemMessage(content=system_msg)]+
                                                       [HumanMessage(content=description)])
response_dict = response.model_dump()
print(json.dumps(response_dict, indent=2))  
