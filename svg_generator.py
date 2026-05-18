import svgwrite
from pathlib import Path
import random

def generate_svg_pictograph(prompt: str, output_path: Path, width=1024, height=576) -> Path:
    """
    Generates a clean, corporate SVG pictograph based on prompt keywords.
    Uses zero VRAM and executes in milliseconds.
    """
    prompt = prompt.lower()
    dwg = svgwrite.Drawing(str(output_path), size=(width, height))
    
    # 1. Dark corporate background to match your Next.js UI theme
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0f172a'))
    
    # Base layout offsets
    center_x, center_y = width // 2, height // 2

    # 2. Semantic Routing (Simple line-art generation)
    if "chart" in prompt or "graph" in prompt or "data" in prompt:
        # Draw an ascending bar chart
        colors = ['#1e3a8a', '#2563eb', '#3b82f6', '#60a5fa']
        for i, color in enumerate(colors):
            bar_height = 100 + (i * 80)
            dwg.add(dwg.rect(
                insert=(center_x - 200 + (i * 100), center_y + 150 - bar_height),
                size=(60, bar_height),
                fill=color,
                rx=8, ry=8
            ))
            
    elif "team" in prompt or "user" in prompt or "people" in prompt:
        # Draw abstract team nodes (circles connected by lines)
        nodes = [(center_x - 150, center_y + 50), (center_x, center_y - 100), (center_x + 150, center_y + 50)]
        # Draw connections
        dwg.add(dwg.line(start=nodes[0], end=nodes[1], stroke='#3b82f6', stroke_width=4))
        dwg.add(dwg.line(start=nodes[1], end=nodes[2], stroke='#3b82f6', stroke_width=4))
        dwg.add(dwg.line(start=nodes[0], end=nodes[2], stroke='#1e3a8a', stroke_width=4))
        # Draw nodes
        for nx, ny in nodes:
            dwg.add(dwg.circle(center=(nx, ny), r=40, fill='#2563eb'))
            dwg.add(dwg.circle(center=(nx, ny), r=20, fill='#60a5fa'))
            
    elif "security" in prompt or "phishing" in prompt or "lock" in prompt:
        # Draw a stylized padlock
        dwg.add(dwg.rect(insert=(center_x - 80, center_y - 20), size=(160, 140), fill='#1e3a8a', rx=16, ry=16))
        dwg.add(dwg.rect(insert=(center_x - 50, center_y - 100), size=(100, 100), fill='none', stroke='#3b82f6', stroke_width=20, rx=40, ry=40))
        dwg.add(dwg.circle(center=(center_x, center_y + 50), r=20, fill='#60a5fa'))
        
    else:
        # Fallback abstract geometric pattern
        dwg.add(dwg.circle(center=(center_x, center_y), r=150, fill='none', stroke='#1e3a8a', stroke_width=8))
        dwg.add(dwg.circle(center=(center_x, center_y), r=100, fill='none', stroke='#2563eb', stroke_width=4))
        dwg.add(dwg.circle(center=(center_x, center_y), r=50, fill='#3b82f6'))

    # Save to disk
    dwg.save()
    return output_path
