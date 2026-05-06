from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re
import streamlit as st

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

st.title("Structured Output with Json Schema")

json_schema = {
    "title": "movie",
    "description": "Extract structured movie information from text.",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The movie title."},
        "year": {"type": "integer", "description": "The movie release year."},
        "genre": {"type": "string", "description": "The movie genre."},
        "rating": {"type": "number", "description": "The movie rating as a number."}
    },
    "required": ["title", "year", "genre", "rating"],
    "additionalProperties": False
}

structured_model = model.with_structured_output(json_schema)

text1 = """The movie Inception, released in 2010, is a mind-bending thriller that falls under the science fiction genre. It was directed by Christopher Nolan and has received critical acclaim for its complex narrative and stunning visuals. The film has a rating of 8.8 on IMDb."""

if st.button("Generate Structured Output"):
    result = structured_model.invoke(text1)

    if result is None:
        st.error("The model did not return a structured output. Try again or check the model.")
    else:
        st.write("Title:", result.get("title", "N/A"))
        st.write("Year:", result.get("year", "N/A"))
        st.write("Genre:", result.get("genre", "N/A"))
        st.write("Rating:", result.get("rating", "N/A"))