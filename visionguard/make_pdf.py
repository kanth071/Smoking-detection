import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

pdf_path = 'VisionGuard_Technical_Architecture_Report.pdf'

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8.5)
        self.setFillColor(colors.HexColor('#64748b'))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, 'VisionGuard AI — Easy Technical Guide & System Explanation')
            self.setStrokeColor(colors.HexColor('#cbd5e1'))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer (all pages)
        page_text = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, 'VisionGuard Surveillance System — Easy Guide (kanth071)')
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=54,
    rightMargin=54,
    topMargin=54,
    bottomMargin=54
)

styles = getSampleStyleSheet()

# Colors
PRIMARY = colors.HexColor('#0f172a')
ACCENT = colors.HexColor('#2563eb')
GREEN = colors.HexColor('#059669')
RED = colors.HexColor('#dc2626')
TEXT_DARK = colors.HexColor('#334155')

# Styles
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=PRIMARY,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=15,
    textColor=ACCENT,
    spaceAfter=12
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=16,
    textColor=PRIMARY,
    spaceBefore=11,
    spaceAfter=5,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'SectionH2',
    parent=styles['Heading3'],
    fontName='Helvetica-Bold',
    fontSize=10,
    leading=13,
    textColor=ACCENT,
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'BodyDark',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=13,
    textColor=TEXT_DARK,
    spaceAfter=4
)

bullet_style = ParagraphStyle(
    'BulletDark',
    parent=body_style,
    leftIndent=12,
    bulletIndent=4,
    spaceAfter=3
)

story = []

# ==================== HEADER ====================
story.append(Paragraph('VisionGuard — Easy Technical Guide', title_style))
story.append(Paragraph('Simple Step-by-Step Explanation of How VisionGuard AI Works', subtitle_style))
story.append(HRFlowable(width='100%', thickness=1.5, color=ACCENT, spaceBefore=0, spaceAfter=10))

# Quick Overview Box
summary_box = [
    [Paragraph('<b>💡 What is VisionGuard in Simple Words?</b>', h2_style)],
    [Paragraph('VisionGuard is like a <b>smart AI security guard inside a camera</b>. It automatically watches live video, finds people, and checks if anyone is smoking in no-smoking areas. When it catches someone smoking, it automatically takes photos, saves evidence, and alerts the dashboard instantly.', body_style)]
]
t_summary = Table(summary_box, colWidths=[504])
t_summary.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#93c5fd')),
    ('PADDING', (0,0), (-1,-1), 7),
]))
story.append(t_summary)
story.append(Spacer(1, 10))

# ==================== 1. SIMPLE TECH STACK ====================
story.append(Paragraph('1. Simple Technology Stack (What Powers VisionGuard?)', h1_style))

stack_data = [
    [Paragraph('<b>Component</b>', body_style), Paragraph('<b>What We Used</b>', body_style), Paragraph('<b>Simple Explanation</b>', body_style)],
    [Paragraph('<b>AI Hardware</b>', body_style), Paragraph('NVIDIA RTX 4050 GPU', body_style), Paragraph('Fast graphics card used to run AI models at 50+ frames per second.', body_style)],
    [Paragraph('<b>Person AI Model</b>', body_style), Paragraph('YOLO11n (Nano)', body_style), Paragraph('Small AI model that detects human bodies and gives each person a ID number.', body_style)],
    [Paragraph('<b>Cigarette AI Model</b>', body_style), Paragraph('Custom YOLO26s (STAL)', body_style), Paragraph('Trained for <b>60 to 150 rounds (epochs)</b> on GPU to spot small cigarettes.', body_style)],
    [Paragraph('<b>Backend Server</b>', body_style), Paragraph('FastAPI & Python', body_style), Paragraph('Main server program that handles camera streams, alerts, and settings.', body_style)],
    [Paragraph('<b>Database</b>', body_style), Paragraph('SQLite (visionguard.db)', body_style), Paragraph('Database file that stores violation photo evidence, dates, and times.', body_style)],
    [Paragraph('<b>Web Dashboard</b>', body_style), Paragraph('HTML & JavaScript', body_style), Paragraph('Live dashboard website where users view live feed, alerts, and photos.', body_style)]
]
stack_table = Table(stack_data, colWidths=[110, 140, 254])
stack_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('PADDING', (0,0), (-1,-1), 5),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
]))
story.append(stack_table)
story.append(Spacer(1, 10))

# ==================== 2. THE 2 AI BRAINS ====================
story.append(Paragraph('2. How the 2 AI Models Work Together', h1_style))
story.append(Paragraph('VisionGuard uses <b>two AI brains</b> working at the exact same time:', body_style))

story.append(Paragraph('<b>Brain #1 — Person Finder (YOLO11n):</b>', h2_style))
story.append(Paragraph('• Scans the camera feed to find human bodies.<br/>'
                       '• Draws a <b>green box</b> around each person and tracks them with a number (e.g. Person #1).', bullet_style))

story.append(Paragraph('<b>Brain #2 — Cigarette Finder (Custom YOLO26s):</b>', h2_style))
story.append(Paragraph('• Trained on an NVIDIA RTX 4050 GPU for <b>60 to 150 epochs (training cycles)</b>.<br/>'
                       '• Specifically trained to detect small, thin cigarettes, lit tips, and smoke.<br/>'
                       '• Draws an <b>orange box</b> around cigarettes when found.', bullet_style))

story.append(Spacer(1, 10))

# ==================== 3. HOW FALSE POSITIVES ARE BLOCKED ====================
story.append(Paragraph('3. How VisionGuard Prevents False Alarms (Spectacles, Pens & Folds)', h1_style))
story.append(Paragraph('Standard AI models sometimes mistake spectacle frames, glasses stems, pens, or neck shadows for cigarettes. VisionGuard uses <b>4 Smart Rules</b> to stop false alarms:', body_style))

rules_data = [
    [Paragraph('<b>False Alarm Cause</b>', body_style), Paragraph('<b>Smart Rule Solution</b>', body_style), Paragraph('<b>Why It Works Simply</b>', body_style)],
    [Paragraph('Spectacle frames, glasses arms, eyes, ears', body_style), Paragraph('<b>No Glasses/Eyes Rule</b><br/>(Top 47% head block)', body_style), Paragraph('Blocks the upper 47% of the head. A cigarette is never placed on someone\'s eyes or glasses stem!', body_style)],
    [Paragraph('Neck shadow folds & shirt collar lines', body_style), Paragraph('<b>No Neck Shadow Rule</b><br/>(Lower neck block)', body_style), Paragraph('Rejects square shadow folds under the mouth.', body_style)],
    [Paragraph('Pens, cables, lollipop sticks', body_style), Paragraph('<b>Shape & Confidence Rule</b><br/>(Conf >= 50%)', body_style), Paragraph('Rejects weak guesses and thin non-cigarette shapes.', body_style)],
    [Paragraph('Cigarette picture on a poster or chair', body_style), Paragraph('<b>Person Overlap Rule</b><br/>(Containment >= 10%)', body_style), Paragraph('Cigarette MUST belong to a real person standing in view.', body_style)]
]
rules_table = Table(rules_data, colWidths=[120, 150, 234])
rules_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('PADDING', (0,0), (-1,-1), 5),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
]))
story.append(rules_table)
story.append(Spacer(1, 10))

# ==================== 4. STEP BY STEP FLOW ====================
story.append(Paragraph('4. Step-by-Step: What Happens When Someone Smokes?', h1_style))
story.append(Paragraph('Here is what happens inside the system in less than 1 second when someone smokes:', body_style))

flow_steps = [
    ('1. Camera Capture', 'The camera records live video continuously and sends frames to the server.'),
    ('2. AI Scanning', 'The 2 AI models scan the frame: one finds the Person, the other finds the Cigarette.'),
    ('3. Smart Rule Verification', 'The system checks: Is the cigarette at the mouth? Is confidence over 50%? Is it NOT glasses?'),
    ('4. Multi-Frame Double Check', 'The AI waits for <b>3 consecutive frames</b> of smoking to ensure it is not a temporary glitch.'),
    ('5. Automatic Photo Saving', 'Once confirmed, it automatically takes <b>4 to 6 evidence photos</b> with boxes drawn around the smoker.'),
    ('6. Database Archiving', 'Saves the event record, time, date, and photo folder path into SQLite database (<code>visionguard.db</code>).'),
    ('7. Live Alert on Dashboard', 'The web dashboard instantly displays red alert cards, updates counters, and shows the evidence photos.')
]

for step_name, step_desc in flow_steps:
    s_box = [
        [Paragraph(f'<b>{step_name}</b>', h2_style)],
        [Paragraph(step_desc, body_style)]
    ]
    t_step = Table(s_box, colWidths=[504])
    t_step.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_step)
    story.append(Spacer(1, 4))

story.append(Spacer(1, 8))

# ==================== 5. DASHBOARD FEATURES ====================
story.append(Paragraph('5. Dashboard Features', h1_style))
story.append(Paragraph('• <b>Live Video Feed:</b> Watch live video with green boxes for people and red boxes for active smoking.<br/>'
                       '• <b>KPI Counters:</b> Real-time counters showing total People, Smokers Now, FPS, and Confirmed Count.<br/>'
                       '• <b>Violation Photos Tab:</b> View multi-frame evidence photo sets taken during violations with zoom view.<br/>'
                       '• <b>Summary Analytics:</b> Color-coded progress bars showing confirmed vs rejected violations.<br/>'
                       '• <b>Settings Controls:</b> Easily adjust AI sensitivity sliders live without restarting.', body_style))

story.append(Spacer(1, 10))
story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=6, spaceAfter=6))
story.append(Paragraph('<b>Status:</b> VisionGuard is fully operational, GPU-accelerated on NVIDIA RTX 4050, and pushed to GitHub (<code>kanth071/Smoking-detection</code>).', ParagraphStyle('EndStatus', parent=body_style, fontName='Helvetica-Oblique', textColor=GREEN)))

# Build Document
doc.build(story, canvasmaker=NumberedCanvas)
print('EASY PDF GENERATED SUCCESSFULLY:', pdf_path)
