import os
from dotenv import load_dotenv
from typing import Dict, Any
from openai import OpenAI
from schema import PatientDetails # Assuming schema.py is in the parent directory

load_dotenv()

class PatientExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_patient_details(self, text_content: str) -> PatientDetails:
        """
        Extracts patient details from the given text content using an LLM.
        This simulates langextract's capabilities.
        """
        # Define the prompt for extracting patient details
        prompt = f"""
        Extract the following patient details from the text below.
        If a field is not present, mark it as null or 'None'.

        Patient's Name:
        Date of Birth (YYYY-MM-DD):
        Gender:
        Address:
        Contact Number:

        Text:
        ---
        {text_content}
        ---

        Provide the output as a JSON object strictly adhering to the PatientDetails Pydantic model.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125", # Or another suitable model
                messages=[
                    {"role": "system", "content": "You are an expert medical data extractor. Extract information accurately and strictly follow the output format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            extracted_data = response.choices[0].message.content
            # Validate and parse with Pydantic
            patient_details = PatientDetails.model_validate_json(extracted_data)
            return patient_details
        except Exception as e:
            print(f"Error during patient details extraction: {e}")
            # Return a default/empty PatientDetails object in case of error
            return PatientDetails(name="Unknown", date_of_birth=None, gender=None, address=None, contact_number=None)


if __name__ == "__main__":
    extractor = PatientExtractor()
    sample_text = """
    Patient Name: John Doe
    DOB: 1980-05-15
    Gender: Male
    Address: 123 Main St, Anytown, USA
    Phone: 555-123-4567
    This is a consultation note for Mr. Doe regarding his recent visit.
    """
    patient_info = extractor.extract_patient_details(sample_text)
    print("Extracted Patient Details:")
    print(patient_info.model_dump_json(indent=2))
