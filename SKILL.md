---
name: excalidraw-generator
description: Generate architecture diagrams, flowcharts, and technical drawings in Excalidraw JSON format (.excalidraw). Use when the user asks to draw a diagram, create an Excalidraw, or visually represent a system.
---

# Excalidraw Generator

## Quick Start

When the user asks for a diagram to be drawn and saved as an Excalidraw file:
1. Determine the logical components (nodes) and their relationships (edges).
2. Calculate an approximate grid layout for nodes (e.g., Column 1 for inputs, Column 2 for processing, Column 3 for outputs).
3. Generate the standard Excalidraw JSON.
4. Use the `Write` tool to save the JSON to a `.excalidraw` file in the user's workspace (or the path they specify).
5. Tell the user they can drag and drop this file into https://excalidraw.com/ or open it with the VS Code Excalidraw extension.

## Layout & Coordinate Strategy

AI coordinate calculation is tricky. Follow these rules to make it usable:
- **Base Grid**: Start the first element at `x: 100, y: 100`.
- **Containers/Groups**: Draw large boundary rectangles first (e.g., WorkerProc, EngineCore) to enclose related components. Container labels should include the Process scope if applicable (e.g., "Process 1: API Frontend").
- **Avoid Overlap & Calculate Lengths Properly**: 
  - **Professional Naming Conventions**: Avoid putting raw filenames (like `request_listener.py` or `generator_torch`) as labels in diagram nodes, as this looks unprofessional in an architecture diagram. Use capitalized conceptual module names, class names, or service roles instead (e.g., "Request Listener" instead of `request_listener.py`, "Torch Generator" instead of `generator_torch`).
  - **Never hardcode large JSON files**: AI coordinate mental math is prone to errors. When generating complex diagrams with many nodes, ALWAYS use a Python script to calculate coordinates dynamically and generate the `.excalidraw` JSON file.
  - **Script Execution Strategy**: Instead of creating a temporary python file on disk and deleting it, prefer executing the python code directly via the `Shell` tool using `python -c "import json; ..."` with multiline strings. This prevents leaving artifacts or deleting files prematurely before execution completes.
  - **Text Width Formula**: Excalidraw text width is exactly `len(text) * font_size * 0.6`. A string of 20 chars at size 16 takes ~192 pixels. NEVER make a bounding rectangle narrower than `text_width + 40` (for padding).
  - **Text Centering Formula**: To center text in a rectangle: `x_text = rect_x + rect_w/2 - text_w/2`, `y_text = rect_y + rect_h/2 - font_size/2`.
  - **Generous Spacing & Parent Container Sizing**: Give extremely generous spacing! Use at least `80-100` pixels of padding inside containers. MOST IMPORTANTLY, ALWAYS ensure that the parent container's `width` and `height` are strictly calculated to be large enough to completely encapsulate all its child components. `parent_width` must be `> max(child_x + child_width) - parent_x + padding`. `parent_height` must be `> max(child_y + child_height) - parent_y + padding`. It is better to have a diagram that is too spread out than one that overlaps or breaks container boundaries.
  - **Arrow Paths & Avoidance**: Arrows must NEVER cross through rectangles or text. If an arrow text label is used, make sure the arrow's `points` array routes AROUND the text, and give the text a `backgroundColor: "#ffffff"` so it masks the line beneath it. Use `"roundness": { "type": 3 }` for curved routing.
- **Standard Sizes**: 
  - Sub-components: Use `width: 150, height: 40`.
  - Nested components (e.g., tokenizer inside InputPreprocessor): use smaller rectangles like `width: 100, height: 30`.
- **Spacing**: 
  - Horizontal spacing: `200` to `250` pixels.
  - Vertical spacing: `100` to `150` pixels.
  - Leave enough padding (at least 40-50px) inside boundary rectangles so child elements do not touch the container borders.
- **Text Placement**: 
  - ALWAYS use `"textAlign": "center"` and `"verticalAlign": "middle"` for all text elements.
  - For text inside a standard shape (e.g. sub-components), calculate its `x` and `y` so it sits exactly in the center of the shape (e.g., `x_text = x_shape + (width_shape - width_text) / 2`).
  - For large container/group labels, place them **inside the container at the top edge, horizontally centered**. The formula for centering text at the top of a container is: `x_text = container_x + container_w/2 - text_w/2`, `y_text = container_y + 10`.
  - ALWAYS apply these centering formulas in your Python script generators. DO NOT just write `x_text = container_x + 10`, that creates left-aligned text, NOT centered text.

## Supported Element Types
- `rectangle`: Standard box.
  - For group borders/containers (like OpenAIServingChat, WorkerProc), use large transparent rectangles with thin borders, placing other elements inside them.
  - Colors: Use solid background colors (`backgroundColor`) for components. Good choices: `#ffc9c9` (light red), `#b2f2bb` (light green), `#a5d8ff` (light blue), `#ffe066` (light yellow). Use `#transparent` for container borders.
  - Use `fillStyle: "solid"` for filled components.
- `ellipse`: Circle/oval.
- `diamond`: Decision point.
- `text`: Labels and inside shapes.
  - To mimic the hand-drawn font in the example, use `fontFamily: 1` (Virgil).
- `arrow`: Connections. 
  - Requires `points` array relative to its own `x, y`. Example: `[[0, 0], [100, 50]]`.
  - Use `startBinding` and `endBinding` to snap to elements.
  - Colors: Use different `strokeColor` for different data flows (e.g., `#2b8a3e` green, `#c92a2a` red, `#1971c2` blue) to distinguish types of requests or data.
  - Add text labels to arrows to explain the data flow (e.g., "放入请求", "调度").

## Excalidraw JSON Structure

Use this template structure to generate `.excalidraw` files via Python.

**Example Python Script Approach**:
```python
import json

elements = []

def add_container(id, x, y, w, h, text):
    elements.append({
      "id": id, "type": "rectangle",
      "x": x, "y": y, "width": w, "height": h,
      "backgroundColor": "transparent", "strokeWidth": 2, "strokeColor": "#1e1e1e"
    })
    # Horizontal centering at top edge
    text_w = len(text) * 20 * 0.6
    elements.append({
      "id": id + "-text", "type": "text",
      "x": x + w/2 - text_w/2, "y": y + 10, 
      "width": text_w, "height": 20,
      "text": text, "fontSize": 20, "fontFamily": 1,
      "textAlign": "center", "verticalAlign": "middle", "strokeColor": "#1e1e1e"
    })

def add_rect(id, x, y, bg, text, font_size=16):
    # Calculate exact text width: len * size * 0.6
    text_w = len(text) * font_size * 0.6
    # Box must be wider than text + padding
    box_w = max(150, text_w + 40)
    box_h = 40
    
    elements.append({
      "id": id, "type": "rectangle",
      "x": x, "y": y, "width": box_w, "height": box_h,
      "backgroundColor": bg, "strokeWidth": 2, "fillStyle": "solid",
      "strokeColor": "#1e1e1e", "roughness": 1
    })
    
    # Mathematical centering
    elements.append({
      "id": id + "-text", "type": "text",
      "x": x + box_w/2 - text_w/2, "y": y + box_h/2 - font_size/2, 
      "width": text_w, "height": font_size,
      "text": text, "fontSize": font_size, "fontFamily": 1,
      "textAlign": "center", "verticalAlign": "middle", "strokeColor": "#1e1e1e"
    })
    return box_w  # Return width so caller can use it to calculate parent container bounds!

# Add elements...
with open("diagram.excalidraw", "w", encoding="utf-8") as f:
    json.dump({"type": "excalidraw", "version": 2, "elements": elements}, f, indent=2)
```
