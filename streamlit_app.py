import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
import streamlit as st

load_dotenv()

# Azure Form Recognizer Configuration
azure_form_recognizer_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
azure_form_recognizer_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
fileModelId = "prebuilt-layout"

# Azure OpenAI Configuration
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")

# Connect to Azure Form Recognizer
document_analysis_client = DocumentAnalysisClient(
    endpoint=azure_form_recognizer_endpoint, 
    credential=AzureKeyCredential(azure_form_recognizer_key)
)

# Ensure the temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")

# Streamlit App
st.title("Nutrition Label Analysis")

uploaded_ingredients_file = st.file_uploader("Upload Ingredients Image", type=["jpg", "jpeg", "png"])
uploaded_nutrition_file = st.file_uploader("Upload Nutrition Table Image", type=["jpg", "jpeg", "png"])

if uploaded_ingredients_file is not None and uploaded_nutrition_file is not None:
    with st.spinner('Analyzing images...'):
        # Save uploaded files temporarily
        ingredients_file_path = os.path.join("temp", uploaded_ingredients_file.name)
        nutrition_file_path = os.path.join("temp", uploaded_nutrition_file.name)

        with open(ingredients_file_path, "wb") as f:
            f.write(uploaded_ingredients_file.getbuffer())
        with open(nutrition_file_path, "wb") as f:
            f.write(uploaded_nutrition_file.getbuffer())

        # Analyze Ingredients Image
        with open(ingredients_file_path, "rb") as f:
            poller_ingredients = document_analysis_client.begin_analyze_document(
                model_id=fileModelId,
                document=f
            )
        result_ingredients = poller_ingredients.result()
        ingredients_content = ""
        if result_ingredients.pages:
            for idx, page in enumerate(result_ingredients.pages):
                for line in page.lines:
                    ingredients_content += f"{line.content}\n"

        # Analyze Nutrition Table Image
        with open(nutrition_file_path, "rb") as f:
            poller_nutrition_table = document_analysis_client.begin_analyze_document(
                model_id=fileModelId,
                document=f
            )
        result_nutrition_table = poller_nutrition_table.result()
        nutrition_table_content = ""
        if result_nutrition_table.tables:
            for table_idx, table in enumerate(result_nutrition_table.tables):
                table_content = []
                for row_idx in range(table.row_count):
                    row_content = [""] * table.column_count
                    table_content.append(row_content)
                for cell in table.cells:
                    table_content[cell.row_index][cell.column_index] = cell.content

                nutrition_table_content += f"\nTable #{table_idx + 1}:\n"
                for row in table_content:
                    nutrition_table_content += "\t".join(row) + "\n"

        combined_content = f"Ingredients:\n{ingredients_content}\nNutrition Table:\n{nutrition_table_content}"

        # Connect to Azure OpenAI
        client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint, 
            api_key=azure_oai_key, 
            api_version="2024-02-15-preview"
        )

        # Create a system message
        system_message = """
        You are a smug, funny nutritionist who provides health advice based on ingredients and nutrition tables. 
        Provide advice on what is safe to consume based on the ingredients and nutrition table.
        Discuss the ingredients as a whole but single out scientifically named ingredients so the user can understand them better.
        Mention the adequate consumption or potential harm based on excessive amounts of substances.
        Identify any potential allergies. Output a general summary first before giving further details. Here are a few examples:
        - "Example:
        Ingredients: Potatoes, Vegetable Oils, Salt, Potassium phosphates
        Nutrition Table:
        - Energy: 532 kcal per 100g
        - Fat: 31.5g per 100g
        - Sodium: 1.28g per 100g

        Summary: The ingredients are pretty standard for potato crisps. Potatoes and vegetable oils provide the base, while salt adds flavor. Watch out for the high fat and sodium content if you're trying to watch your heart health or blood pressure. As for allergies, you're mostly safe unless you're allergic to potatoes or sunflower/rapeseed oil. Potassium phosphates? Just some friendly muscle helpers, but keep it moderate!

        Potassium phosphates: Ah, the magical salts that help keep your muscles happy. Just don't overdo it!"
        """

        messages_array = [{"role": "system", "content": system_message}]
        messages_array.append({"role": "user", "content": f"Please analyze the following nutrition label content:\n{combined_content}"})

        # Send request to Azure OpenAI model
        response = client.chat.completions.create(
            model=azure_oai_deployment,
            temperature=0.6,
            max_tokens=1200,
            messages=messages_array
        )

        generated_text = response.choices[0].message.content

        # Display the summary generated by OpenAI
        st.subheader("Generated Summary:")
        st.write(generated_text)

        # Clean up temporary files
        os.remove(ingredients_file_path)
        os.remove(nutrition_file_path)
