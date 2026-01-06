from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_pdf(filename, customer_name, revenue, cashflow, risk_type):
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "HDFC BANK - CORPORATE STATEMENT")
    c.line(50, 740, 550, 740)
    
    # Customer Details
    c.setFont("Helvetica", 12)
    c.drawString(50, 710, f"Customer Name: {customer_name}")
    c.drawString(50, 690, f"Account Number: XXXX-XXXX-8821")
    c.drawString(50, 670, f"Statement Period: Jan 2025 - Dec 2025")
    
    # Financials (The Data your AI reads)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 630, "Financial Summary:")
    
    c.setFont("Helvetica", 12)
    # Note: We formatting numbers with commas for realism
    c.drawString(50, 610, f"Annual Revenue: {revenue}") 
    c.drawString(50, 590, f"Avg Monthly Cashflow: {cashflow}")
    
    # Footer / Watermark logic to identify the test case visually
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 100, f"Internal Test Case: {risk_type}")
    
    c.save()
    print(f"✅ Created: {filename}")

# --- DATA FOR THE 5 TEST CASES ---
test_cases = [
    {
        "filename": "Case1_Golden_Applicant.pdf",
        "name": "Alpha Tech Solutions",
        "revenue": "85,00,000",
        "cashflow": "2,50,000",
        "type": "High Revenue / High Cashflow (Should APPROVE)"
    },
    {
        "filename": "Case2_HighRisk_LowCash.pdf",
        "name": "Beta Traders Pvt Ltd",
        "revenue": "90,00,000", 
        "cashflow": "15,000",
        "type": "High Revenue / LOW Cashflow (Should REJECT or FLAG)"
    },
    {
        "filename": "Case3_SmallBiz_Healthy.pdf",
        "name": "Gamma Retailers",
        "revenue": "12,00,000",
        "cashflow": "60,000",
        "type": "Low Revenue / Healthy Cashflow (Marginal/Safe)"
    },
    {
        "filename": "Case4_The_Defaulter.pdf",
        "name": "Delta Manufacturing",
        "revenue": "5,00,000",
        "cashflow": "5,000",
        "type": "Low Revenue / Low Cashflow (Definite REJECT)"
    },
    {
        "filename": "Case5_Whale_Client.pdf",
        "name": "Omega Global Exports",
        "revenue": "5,00,00,000",
        "cashflow": "10,00,000",
        "type": "Massive Revenue (Instant APPROVE)"
    }
]

# Create them all
if __name__ == "__main__":
    print("--- Generating Test PDFs ---")
    for case in test_cases:
        create_pdf(case["filename"], case["name"], case["revenue"], case["cashflow"], case["type"])
    print("--- Done! Check your folder. ---")