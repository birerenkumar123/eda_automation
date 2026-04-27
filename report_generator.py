from fpdf import FPDF
import os


def create_pdf_report(insights, filename="eda_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(
        200,
        10,
        txt="EDA Analysis Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    for item in insights:
        pdf.multi_cell(
            0,
            10,
            txt=f"- {item}"
        )

    file_path = os.path.abspath(filename)
    pdf.output(file_path)

    return file_path