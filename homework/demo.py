"""Create a shareable visual-QA result card from a saved SuperTuxKart VLM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# The saved LoRA adapter still relies on the base model. Use the local cache
# during demos so an otherwise complete setup does not fail on a network check.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image, ImageDraw, ImageFont

from .finetune import load
from .generate_qa import generate_qa_pairs


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, text_font, width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=text_font) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def result_card(image_path: Path, question: str, prediction: str, answer: str, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    card_width = 1280
    image = image.resize((card_width, round(image.height * card_width / image.width)), Image.Resampling.LANCZOS)
    margin, panel_height = 48, 330
    canvas = Image.new("RGB", (image.width, image.height + panel_height), "#0b1020")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    label_font, body_font, answer_font = font(23, True), font(29), font(30, True)
    y = image.height + margin
    sections = [("QUESTION", question, "#d9e3ff", body_font), ("MODEL ANSWER", prediction, "#64e7c8", answer_font), ("REFERENCE ANSWER", answer, "#ffcf70", answer_font)]
    for label, text, color, text_font in sections:
        draw.text((margin, y), label, fill=color, font=label_font)
        y += 34
        for line in wrap(draw, text, text_font, canvas.width - 2 * margin):
            draw.text((margin, y), line, fill="white", font=text_font)
            y += 38
        y += 14
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a saved-VLM SuperTuxKart demo card.")
    parser.add_argument("--info-file", type=Path, required=True, help="Path to a *_info.json file.")
    parser.add_argument("--view-index", type=int, default=0, help="Camera view to use.")
    parser.add_argument("--question-index", type=int, default=0, help="Generated question to ask for that frame.")
    parser.add_argument("--model", default="vlm_model", help="Model directory relative to homework/.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vqa_demo.png"))
    args = parser.parse_args()

    qa_pairs = generate_qa_pairs(str(args.info_file), args.view_index)
    if not 0 <= args.question_index < len(qa_pairs):
        raise ValueError(f"question-index must be between 0 and {len(qa_pairs) - 1}")
    qa = qa_pairs[args.question_index]
    image_path = args.info_file.parent / Path(qa["image_file"]).name
    if not image_path.exists():
        raise FileNotFoundError(f"Image for this frame was not found: {image_path}")

    vlm = load(args.model)
    prediction = vlm.generate(str(image_path), qa["question"])
    result_card(image_path, qa["question"], prediction, qa["answer"], args.output)
    print(f"Question: {qa['question']}")
    print(f"Model answer: {prediction}")
    print(f"Reference answer: {qa['answer']}")
    print(f"Saved demo card: {args.output}")


if __name__ == "__main__":
    main()
