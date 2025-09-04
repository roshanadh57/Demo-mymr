import os
from dotenv import load_dotenv
from langchain.chains import create_tagging_chain_pydantic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from schema import DocumentSummary, DocumentClassification # Assuming schema.py is in the parent directory

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", api_key=os.getenv("OPENAI_API_KEY"))

        # Prompt for summarization
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert medical document summarizer. Provide a concise and accurate summary, and identify key findings."),
            ("human", "Summarize the following medical document and list 3-5 key findings:\n\n{document_content}\n\nStrictly output as JSON adhering to the DocumentSummary Pydantic model.")
        ])

        # Chain for summarization
        self.summary_chain = self.summary_prompt | self.llm.with_structured_output(DocumentSummary)

        # Chain for classification using Pydantic tagging
        self.classification_chain = create_tagging_chain_pydantic(DocumentClassification, self.llm)


    def summarize_document(self, text_content: str) -> DocumentSummary:
        """Summarizes the given document content."""
        try:
            summary = self.summary_chain.invoke({"document_content": text_content})
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return DocumentSummary(summary_text="Failed to summarize document.", key_findings=[])

    def classify_document(self, text_content: str) -> DocumentClassification:
        """Classifies the given document content."""
        try:
            classification = self.classification_chain.invoke({"input": text_content})
            # create_tagging_chain_pydantic returns a dict, so we validate it
            return DocumentClassification.model_validate(classification)
        except Exception as e:
            print(f"Error during classification: {e}")
            return DocumentClassification(category="Unknown", is_sensitive=True)


if __name__ == "__main__":
    processor = DocumentProcessor()
    sample_text = """
    **LAB REPORT**

    Patient: Jane Smith
    DOB: 1992-11-23
    Lab ID: LR-2023-001
    Date: 2023-10-26

    Test: Complete Blood Count (CBC)
    Results:
    - Hemoglobin: 12.5 g/dL (Normal: 12-15)
    - White Blood Cell Count: 11.2 x 10^9/L (High, Normal: 4.5-11.0)
    - Platelets: 280 x 10^9/L (Normal: 150-400)

    Impression: Mild leukocytosis, likely due to a recent infection. Recommend follow-up with GP.
    """
    summary = processor.summarize_document(sample_text)
    classification = processor.classify_document(sample_text)

    print("\n--- Summarization Result ---")
    print(summary.model_dump_json(indent=2))

    print("\n--- Classification Result ---")
    print(classification.model_dump_json(indent=2))
