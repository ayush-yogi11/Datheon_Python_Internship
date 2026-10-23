import os
from typing import List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service purchased.")
    quantity: int = Field(description="Quantity purchased.")
    unit_price: float = Field(description="Price per individual unit.")
    total_price: float = Field(description="Total price for this line item.")

class InvoiceSchema(BaseModel):
    invoice_number: str = Field(description="The unique invoice identifier or number.")
    vendor_name: str = Field(description="Name of the company or vendor issuing the invoice.")
    client_name: Optional[str] = Field(description="Name of the client or customer being billed.")
    invoice_date: str = Field(description="Date the invoice was issued (YYYY-MM-DD format if possible).")
    line_items: List[LineItem] = Field(description="List of purchased items or services.")
    subtotal: float = Field(description="Subtotal before taxes and discounts.")
    tax_amount: float = Field(description="Total tax applied.")
    total_amount: float = Field(description="Final total amount due.")



def extract_invoice_data(unstructured_text: str) -> InvoiceSchema:
    # Initialize the Gemini Client
    # Client automatically picks up GEMINI_API_KEY from the environment
    client = genai.Client()

    # Construct the prompt with instructions
    prompt = f"""
    You are an expert financial document parser. Extract structured information from the following unstructured invoice text accurately. 
    Ensure all fields adhere strictly to the requested schema.

    Invoice Text:
    """ + unstructured_text

    # Call the model with structured output configuration
    response = client.models.generate_content(
        model='gemini-3.8-flash',  
        contents=prompt,
        config=types.GenerateContentConfig(
            # Force JSON output and constrain it to match the Pydantic schema
            response_mime_type="application/json",
            response_schema=InvoiceSchema,
            temperature=0.1,  # Low temperature for precise, deterministic extraction
        ),
    )

    # Parse and validate the response back into the Pydantic model
    # The SDK automatically parses the JSON text into the Pydantic instance when response_schema is provided.
    return response.parsed


if __name__ == "__main__":
    # Sample unstructured invoice text
    sample_invoice = """
    INVOICE #INV-2026-9981
    From: CloudScale Technologies Inc.
    Billed To: Acme Corp
    Date: October 23, 2026

    Items:
    - Enterprise Cloud Hosting (12 months): $1,200.00 total (Qty: 1, Unit Price: $1,200.00)
    - Premium Support Tier (Monthly): $150.00 total (Qty: 1, Unit Price: $150.00)

    Subtotal: $1,350.00
    Tax (10%): $135.00
    Total Due: $1,485.00
    """

    print("--- Parsing Unstructured Invoice ---")
    structured_data = extract_invoice_data(sample_invoice)
    
    # Print the validated Pydantic model output nicely formatted as JSON
    print(structured_data.model_dump_json(indent=2))