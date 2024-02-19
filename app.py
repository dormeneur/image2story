from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

load_dotenv(find_dotenv())

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=50)
#image to text
def img2text(url):
    try:
        image_to_text = pipe
        text = image_to_text(url)[0]["generated_text"]
        print(text)
        return text
    except Exception as e:
        print(f"Error during image-to-text generation: {e}")
        return None
img2text("IMAGE2SPEECH\\basketballdunk.jpg")


#llm
def generate_story(scenario):
    template = """ 
    You are a storyteller;
    You can generate a short story based on a simple narrative, The story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temprature=1), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


generate_story(img2text("IMAGE2SPEECH\\basketballdunk.jpg"))
#text to speech


#i can make any changes in this file