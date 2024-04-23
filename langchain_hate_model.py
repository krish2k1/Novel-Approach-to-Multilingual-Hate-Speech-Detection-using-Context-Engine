

!pip install itranslate

# Commented out IPython magic to ensure Python compatibility.
# #v1

%%writefile app.py
import os, streamlit as st
from langchain.llms.openai import OpenAI
import json
from translation import translate, detect_language

# Streamlit app
st.title('Sentiment Analysis Tool')

# Get OpenAI API key and source text input
openai_api_key = st.text_input("OpenAI API Key", type="password")
source_text = st.text_area("Source Text", height=200)

def extract_indices_from_text(text):
    # Define regular expressions to extract indices
    severity_pattern = r'Severity of Statement \(S\): (\d+(\.\d+)?)'
    derogatory_pattern = r'Derogatory Language Used \(D\): (\d+(\.\d+)?)'
    harm_pattern = r'Potential Harm or Offense \(H\): (\d+(\.\d+)?)'

    # Search for patterns in the text
    severity_match = re.search(severity_pattern, text)
    derogatory_match = re.search(derogatory_pattern, text)
    harm_match = re.search(harm_pattern, text)

    # Extract severity index, derogatory index, and harm index
    severity_index = float(severity_match.group(1)) if severity_match else None
    derogatory_index = float(derogatory_match.group(1)) if derogatory_match else None
    harm_index = float(harm_match.group(1)) if harm_match else None

    return severity_index, derogatory_index, harm_index


def calculate_hate_score(severity_index, derogatory_index, harm_index):
    if severity_index is None or derogatory_index is None or harm_index is None:
        return None

    # Define weights
    weight_severity = 0.4
    weight_derogatory = 0.3
    weight_harm = 0.3

    # Calculate Hate Score
    hate_score = weight_severity * severity_index + weight_derogatory * derogatory_index + weight_harm * harm_index
    return hate_score

def predict_sentiment(text, api_key):
    # Detect the language of the input text and translate to English if necessary
    detected_lang = detect_language(text)
    if detected_lang != 'en':
        text = translate_text(text, src=detected_lang, dest='en')

    llm = OpenAI(temperature=0, openai_api_key=api_key)
    response = llm.generate(
        prompts=[f"From the given text, recognize Hate Speech (Hate speech typically involves language that directly attacks or discriminates against a particular group based on attributes such as race, religion, ethnicity, sexual orientation, etc.) and add it to a list. Sentences in descriptions of violence and destruction in a war setting must also be considered. If this is hate speech give 1 else give 0, do not give anything else other than 1 or 0. If it is hate speech give a hate score between 1 - 10, do not give anything else and also mention towards which group the hate is targeted. Give the response in JSON. {text}"],
        max_tokens=60,
        model="gpt-3.5-turbo-instruct"
    )

    output = response.generations[0][0].text.strip()
    lines = output.split('\n')
    hate_speech = lines[0]
    hate_score = None
    targeted_group = None

    if hate_speech.lower() == "1":
        hate_score = lines[1].split(":")[1].strip()
        targeted_group = lines[2].split(":")[1].strip()

    sentiment = {"Hate Speech": hate_speech, "Hate Score": hate_score, "Targeted Group": targeted_group}

    return sentiment

def rewrite_to_positive(hate_speech, hate_score, targeted_group, api_key):
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    # Generate a response using the appropriate model
    response = llm.generate(
        prompts=[f"Transform the hate_speech into a positive message while retaining the original context. You are provided with the hate_score, and it's important that the transformed text doesn't target the specified targeted_group negatively. The output should only contain positive language and must include the targeted audience.{hate_speech, hate_score, targeted_group}"],
        max_tokens=60,
        model="gpt-3.5-turbo-instruct"  # Make sure to replace with the correct model name you have access to
    )

    # Extract the generated text from the response
    output = response.generations[0][0].text.strip()
    return output


if st.button("Check Sentiment"):
    # Validate inputs
    if not openai_api_key.strip() or not source_text.strip():
        st.write("Please complete the missing fields.")
    else:
        try:
            # Predict sentiment
            sentiment = predict_sentiment(source_text, openai_api_key)
            st.write(f"Sentiment: {sentiment}")
            if sentiment["Hate Speech"].lower() == "yes":
                positive_message = rewrite_to_positive(sentiment["Hate Speech"], sentiment["Hate Score"], sentiment["Targeted Group"], openai_api_key)
                st.write(f"Positive Message: {positive_message}")
        except Exception as e:
            st.write(f"An error occurred: {e}")


"""Install localtunnel to serve the Streamlit app"""

!npm install localtunnel

"""Run the Streamlit app in the background"""

!streamlit run app.py

!streamlit run app.py &>/content/logs.txt &

"""Expose the Streamlit app on port 8501"""

!npx localtunnel --port 8501