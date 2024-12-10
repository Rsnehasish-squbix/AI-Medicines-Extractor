from flask import Flask, request, render_template
import pandas as pd
import json
import ast
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Groq API key
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# Initialize ChatGroq
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def extract_json(content):
    start = content.find('```json\n') + 7
    end = content.find('```', start)
    json_str = content[start:end].strip()
    return ast.literal_eval(json_str)

def extract_json_manual(content):
    start = content.find('```json\n') + 7
    end = content.find('```', start)
    json_str = content[start:end].strip()
    return json.loads(json_str)

def get_ai_msg(patient_prompt):
    messages = messages = [  
    {
        "role": "system",
        "content": """
        You are a medical assistant capable of extracting structured details from unstructured clinical text provided by a doctor. Your task is to identify specific categories of information from the provided text precisely. Additionally, you need to identify and flag any unknown or unfamiliar terms that may need further clarification or special handling.

        The categories you need to extract from the text are:
        1. **Status**: Identify the patient's health status, such as "stable," "critical," "recovering," "improving," etc.
        2. **Pharmacy**: Extract references to medication, prescriptions, or pharmacy-related details (e.g., medication names, dosages, directions).
        3. **Services**: Extract any references to medical services, tests, or procedures mentioned (e.g., lab tests, imaging, surgeries).
        4. **Unknown Words**: Flag any terms or phrases that are unfamiliar or do not fit within the recognized categories, indicating the need for further clarification or special handling.

        Please note that further clarification or special handling may be needed for the patient's health status and the interpretation of their symptoms.
        """
    },
    {
        "role": "system",
        "content": """
        Based on the provided information, respond strictly in valid JSON format enclosed in triple backticks (`json`). Use the following schema:

        ```json
        {
            "status": "stable",  // Replace with identified status.
            "pharmacy": {
                "medications": [
                    {
                        "name": "medicine name",  // Replace with actual medication name.
                        "dosage": "dosage amount",  // Replace with dosage.
                        "unit": "unit",  // Replace with unit (e.g., mg, ml).
                        "ICD_code": "ICD code",  // Search and include the ICD code.
                        "frequency": "frequency details"  // Replace with specific time periods (e.g., daily, twice a day).
                    }
                ]
            },
            "services": {  
                "tests": [
                    // Include all prescribed tests, if any.
                ]           
            },
            "unknown_words": [ 
                // Include any unfamiliar or unclear terms.
            ]
        }
        ```

        Ensure the response strictly adheres to this JSON schema.
        """
    },
    {
        "role": "user",
        "content": patient_prompt
    }
]


    ai_msg = llm.invoke(messages)
    ai_msg_json = extract_json(ai_msg.content)
    ai_msg_json = extract_json_manual(ai_msg.content)
    return ai_msg_json

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    patient_prompt = request.form['patient_prompt']
    response = get_ai_msg(patient_prompt)
    
    # Convert response to a DataFrame for tabular representation
    status = response.get("status", "Not Available")
    pharmacy = response.get("pharmacy", {})
    services = response.get("services", {})
    unknown_words = response.get("unknown_words", [])

    pharmacy_data = []
    for medication in pharmacy.get("medications", []):
        pharmacy_data.append({
            "Medication": medication.get("name", "N/A"),
            "Dosage": medication.get("dosage", "N/A"),
            "Unit": medication.get("unit", "N/A"),
            "ICD Code": medication.get("ICD_code", "N/A"),
            "Frequency": medication.get("frequency", "N/A")
        })

    # Create DataFrame for Pharmacy
    pharmacy_df = pd.DataFrame(pharmacy_data)

    # Convert unknown words into a DataFrame
    unknown_df = pd.DataFrame(unknown_words, columns=["Unknown Words"])

    return render_template('result.html', status=status, pharmacy=pharmacy_df.to_html(classes='table'), services=services, unknown_words=unknown_df.to_html(classes='table'))

if __name__ == '__main__':
    app.run(debug=True)
