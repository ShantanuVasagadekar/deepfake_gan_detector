import os
from fpdf import FPDF
import re

class ProjectReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, 'DeepShield: Final Project Report - Shantanu Vasagadekar', 0, 0, 'R')
            self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('helvetica', 'B', 16)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 12, label, 0, 1, 'L', fill=True)
        self.ln(4)

    def section_title(self, label):
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(2)

    def subsection_title(self, label):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 8, label, 0, 1, 'L')
        self.ln(1)

    def text_body(self, text):
        self.set_font('helvetica', '', 11)
        # Handle mathematical symbols and other non-latin1 characters
        text = text.replace('Σ', 'Sum').replace('β', 'beta').replace('±', '+/-')
        text = text.replace('≥', '>=').replace('≤', '<=').replace('→', '->')
        # Encode to latin-1 and ignore anything else to prevent crashes
        clean_text = text.encode('latin-1', 'ignore').decode('latin-1')
        self.multi_cell(0, 7, clean_text)
        self.ln(2)

    def list_item(self, text):
        self.set_font('helvetica', '', 11)
        self.set_x(20)
        # Handle mathematical symbols in list items too
        text = text.replace('Σ', 'Sum').replace('β', 'beta').replace('±', '+/-')
        text = text.replace('≥', '>=').replace('≤', '<=').replace('→', '->')
        clean_text = text.encode('latin-1', 'ignore').decode('latin-1')
        self.cell(5, 7, '-', 0, 0)
        self.multi_cell(0, 7, clean_text)
        self.ln(1)

def generate_report():
    pdf = ProjectReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- COVER PAGE ---
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 28)
    pdf.ln(40)
    pdf.multi_cell(0, 15, 'FINAL PROJECT REPORT', 0, 'C')
    pdf.ln(10)
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.multi_cell(0, 12, 'DeepShield: A Multi-Modal Forensic Ensemble for Robust Deepfake and AI-Generated Image Detection', 0, 'C')
    
    pdf.ln(50)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Student Name: Shantanu Vasagadekar', 0, 1, 'C')
    pdf.set_font('helvetica', '', 14)
    pdf.cell(0, 10, 'Academic Year: 2025-2026', 0, 1, 'C')
    pdf.cell(0, 10, 'Department: Computer Science and Engineering', 0, 1, 'C')
    pdf.cell(0, 10, 'University: [Insert University Name]', 0, 1, 'C')
    
    pdf.set_y(-40)
    pdf.set_font('helvetica', 'I', 10)
    pdf.cell(0, 10, 'Generated on April 23, 2026', 0, 1, 'C')

    # --- CONTENT ---
    with open('FINAL_PROJECT_REPORT.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pdf.add_page()
    
    in_list = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip title page content if it's already in the cover
        if line.startswith('# FINAL PROJECT REPORT') or line.startswith('**Student Name') or line.startswith('**Academic'):
            continue
        if line.startswith('## Project Title'):
            continue
        
        # Handle Headers
        if line.startswith('### '):
            pdf.ln(5)
            pdf.chapter_title(line.replace('### ', ''))
        elif line.startswith('## '):
            pdf.ln(5)
            pdf.chapter_title(line.replace('## ', ''))
        elif line.startswith('# '):
            pdf.ln(5)
            pdf.chapter_title(line.replace('# ', ''))
        elif re.match(r'^\d+\.\s', line): # Section numbers like 1. Introduction
            pdf.ln(5)
            pdf.chapter_title(line)
        elif re.match(r'^\d+\.\d+\s', line): # Subsection numbers like 1.1 Background
            pdf.section_title(line)
        
        # Handle Lists
        elif line.startswith('* ') or line.startswith('- '):
            pdf.list_item(line[2:])
        elif re.match(r'^\d+\.\s+', line) and not line.endswith('Introduction') and len(line) > 20: # Numbered list
            pdf.list_item(line)
            
        # Handle Horizontal Rules
        elif line == '---':
            pdf.ln(5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
        # Handle Text
        else:
            # Clean bold/italic markdown
            clean_text = line.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
            pdf.text_body(clean_text)

    output_path = 'PRN_DeepShield_FinalReport.pdf'
    pdf.output(output_path)
    print(f"PDF Report generated: {output_path}")

if __name__ == "__main__":
    generate_report()
