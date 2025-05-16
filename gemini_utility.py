# import os
# import json
# from PIL import Image

# import google.generativeai as genai

# # working directory path
# working_dir = os.path.dirname(os.path.abspath(__file__))

# # path of config_data file
# config_file_path = f"{working_dir}/config.json"
# config_data = json.load(open("config.json"))

# # loading the GOOGLE_API_KEY
# GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# # configuring google.generativeai with API key
# genai.configure(api_key=GOOGLE_API_KEY)


# def load_gemini_pro_model():
#     gemini_pro_model = genai.GenerativeModel("gemini-1.5-flash-001")
#     return gemini_pro_model


# # get response from Gemini-Pro-Vision model - image/text to text
# def gemini_pro_vision_response(prompt, image):
#     gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash-001")
#     response = gemini_pro_vision_model.generate_content([prompt, image])
#     result = response.text
#     return result


# # get response from embeddings model - text to embeddings
# def embeddings_model_response(input_text):
#     embedding_model = "models/embedding-001"
#     embedding = genai.embed_content(model=embedding_model,
#                                     content=input_text,
#                                     task_type="retrieval_document")
#     embedding_list = embedding["embedding"]
#     return embedding_list


# # get response from Gemini-Pro model - text to text
# def gemini_pro_response(user_prompt):
#     gemini_pro_model = genai.GenerativeModel("gemini-1.5-flash-001")
#     response = gemini_pro_model.generate_content(user_prompt)
#     result = response.text
#     return result


# # result = gemini_pro_response("What is Machine Learning")
# # print(result)
# # print("-"*50)
# #
# #
# # image = Image.open("test_image.png")
# # result = gemini_pro_vision_response("Write a short caption for this image", image)
# # print(result)
# # print("-"*50)
# #
# #
# # result = embeddings_model_response("Machine Learning is a subset of Artificial Intelligence")
# # print(result)

import os
import json
from PIL import Image

import google.generativeai as genai

# working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# path of config_data file
config_file_path = f"{working_dir}/config.json"
config_data = json.load(open("config.json"))

# loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)


def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-1.5-flash-001")
    return gemini_pro_model


# get response from Gemini-Pro-Vision model - image/text to text
def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash-001")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result


# get response from embeddings model - text to embeddings
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list


# get response from Gemini-Pro model - text to text
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-1.5-flash-001")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result


# result = gemini_pro_response("What is Machine Learning")
# print(result)
# print("-"*50)
#
#
# image = Image.open("test_image.png")
# result = gemini_pro_vision_response("Write a short caption for this image", image)
# print(result)
# print("-"*50)
#
#
# result = embeddings_model_response("Machine Learning is a subset of Artificial Intelligence")
# print(result)



# get response from Gemini-Pro-Vision model - for medical image analysis
def medical_image_analysis_response(image):
    system_prompt = """ 
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, disease, or health issues that may be present in the image.

Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Finding Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. Recommendations and Next steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are 'unable to be determined based on the provided image'.
3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."

4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above.

Please provide me an output response with these 4 headings: Detailed Analysis, Finding Report, Recommendations and Next steps, Treatment Suggestions
"""
    gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash-001")
    response = gemini_pro_vision_model.generate_content([system_prompt, image])
    return response.text
