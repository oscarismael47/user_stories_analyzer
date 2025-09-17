from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Image

def generate_user_stories_report(
    original_stories,
    processed_stories,
    overall_consistency,
    contradictions,
    contradictions_indexes,
    contextual_issues,
    contextual_issues_indexes,
    value_assessment,
    clarity_assessment,
    strengths,
    general_recommendations,
    final_stories,
    diagram_path,
    filename="user_stories_report.pdf"
):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1Center", parent=styles['Heading1'], alignment=1))
    styles.add(ParagraphStyle(name="Heading2Bold", parent=styles['Heading2'], textColor=colors.darkblue))

    # Title
    elements.append(Paragraph("User Stories Analysis Report", styles["Heading1Center"]))
    elements.append(Spacer(1, 20))

    # Section 1: Original Stories
    elements.append(Paragraph("1. Original User Stories", styles["Heading2Bold"]))
    for k, v in original_stories.items():
        elements.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        elements.append(Spacer(1, 5))
    elements.append(Spacer(1, 15))

    # Section 2: Processed Stories
    elements.append(Paragraph("2. Processed User Stories", styles["Heading2Bold"]))
    for k, v in processed_stories.items():
        elements.append(Paragraph(f"<b>{k}:</b>", styles["Normal"]))
        elements.append(Paragraph(f"Original: {v['original_user_story']}", styles["Normal"]))
        elements.append(Paragraph(f"Processed: {v['new_user_story']}", styles["Normal"]))
        elements.append(Paragraph(f"Explanation: {v['explanation']}", styles["Normal"]))
        elements.append(Spacer(1, 10))
    elements.append(Spacer(1, 15))

    # Section 3: General Analysis
    elements.append(Paragraph("3. General User Stories Analysis", styles["Heading2Bold"]))
    elements.append(Paragraph(f"<b>Overall consistency:</b> {overall_consistency}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Contradictions:</b> {contradictions}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Contradictions Indexes:</b> {contradictions_indexes}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Contextual Issues:</b> {contextual_issues}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Contextual Issues Indexes:</b> {contextual_issues_indexes}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Value Assessment:</b> {value_assessment}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Clarity Assessment:</b> {clarity_assessment}", styles["Normal"]))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Strengths:</b>", styles["Normal"]))
    for s in strengths:
        elements.append(Paragraph(f"- {s}", styles["Normal"]))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>General Recommendations:</b>", styles["Normal"]))
    for r in general_recommendations:
        elements.append(Paragraph(f"- {r}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    # Section 4: Final User Stories
    elements.append(Paragraph("4. Final User Stories", styles["Heading2Bold"]))
    for k, v in final_stories.items():
        elements.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        elements.append(Spacer(1, 5))

    # Section 5: Diagram (if provided)
    if diagram_path:
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("5. User Stories Diagram", styles["Heading2Bold"]))
        try:
            img = Image(diagram_path, width=400, height=300)
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Could not load diagram: {e}", styles["Normal"]))


    # Build PDF
    doc.build(elements)
    print(f"✅ Report generated: {filename}")


# Ejemplo de uso (pones tus datos aquí):
if __name__ == "__main__":
    original_stories = {...}  # tu dict original
    processed_stories = {...} # tu dict procesado
    overall_consistency = "..."
    contradictions = []
    contradictions_indexes = []
    contextual_issues = []
    contextual_issues_indexes = []
    value_assessment = "..."
    clarity_assessment = "..."
    strengths = ["...", "..."]
    general_recommendations = ["...", "..."]
    final_stories = {...}

    generate_user_stories_report(
        original_stories,
        processed_stories,
        overall_consistency,
        contradictions,
        contradictions_indexes,
        contextual_issues,
        contextual_issues_indexes,
        value_assessment,
        clarity_assessment,
        strengths,
        general_recommendations,
        final_stories
    )
