from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os
import re
import streamlit as st

load_dotenv()

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ['HF_HOME'] = './hf_cache'

llm = HuggingFacePipeline.from_model_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation", pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 2048})

model = ChatHuggingFace(llm=llm)

st.title("Structured Output with TypedDict")


# schema
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model = model.with_structured_output(Review)

if st.button("Generate Structured Output"):

    result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

    Pros:
    Insanely powerful processor (great for gaming and productivity)
    Stunning 200MP camera with incredible zoom capabilities
    Long battery life with fast charging
    S-Pen support is unique and useful
                                    
    Review by Nitish Singh
    """)

    if result is None:
        st.error("The model did not return a structured output. Try again or check the model.")
    else:
        st.write(result['name'])
        st.write(result['key_themes'])
        st.write(result['summary'])
        st.write(result['sentiment'])
        st.write(result['pros'])
        st.write(result['cons'])

