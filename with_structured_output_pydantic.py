from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
import os
import re
import streamlit as st

load_dotenv()

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ['HF_HOME'] = './hf_cache'

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

st.title("Structured Output with Pydantic")

class Movie(BaseModel):
    title: str = Field(...,description="The title of the movie")
    year: int = Field(...,description="The release year of the movie")
    genre: str = Field(...,description="The genre of the movie")
    rating: Optional[float] = Field(default=None, description="The IMDb rating of the movie, if available")

structured_model = model.with_structured_output(Movie)

text1 = """The movie Inception, released in 2010, is a mind-bending thriller that falls under the science fiction genre. It was directed by Christopher Nolan and has received critical acclaim for its complex narrative and stunning visuals. The film has a rating of 8.8 on IMDb."""

text2 = """The movie The Godfather, released in 1972, is a classic crime drama that falls under the genre of mafia films. It was directed by Francis Ford Coppola and is widely regarded as one of the greatest films in world cinema. The film has a rating of 9.2 on IMDb.""" 

text3="""The movie The Room, released in 2003, is a cult classic that falls under the genre of drama. It was directed by Tommy Wiseau and is often cited as one of the worst films ever made, which has ironically contributed to its cult status. The film has a rating of 3.7 on IMDb."""

if st.button("Generate Structured Output"):
    result = structured_model.invoke(text2)

    if result is None:
        st.error("The model did not return a structured output. Try again or check the model.")
    else:
        st.write("Title:", result.title)
        st.write("Year:", result.year)
        st.write("Genre:", result.genre)
        st.write("Rating:", result.rating if result.rating is not None else "N/A")
