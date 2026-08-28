# SuperTuxKart Visual Question Answering

A vision-language-model and CLIP-style retrieval project for answering questions about SuperTuxKart race scenes.

## What it does

The pipeline turns SuperTuxKart scene metadata into two training datasets:

- visual question-answer examples, including ego kart, track, visible-kart counts, and relative positions;
- image-caption pairs for contrastive training.

It then fine-tunes a VLM and a CLIP-style model for multiple-choice visual QA. The held-out evaluation reached **74.2% VLM accuracy** and **63.5% CLIP accuracy**. Training used only the supplied training split.

## Repository layout

- `homework/generate_qa.py` generates visual QA pairs from scene metadata.
- `homework/generate_captions.py` generates image-caption pairs.
- `homework/finetune.py` trains and evaluates the VLM.
- `homework/clip.py` trains and evaluates the CLIP-style model.
- `homework/base_vlm.py` and `homework/data.py` define the model and data path.

## Run it

Create an environment and install dependencies:

```bash
conda create -n tux-vqa python=3.12 -y
conda activate tux-vqa
pip install -r requirements.txt
```

Download the course dataset, then place its `train/` and validation folders under `data/`. The dataset is not committed because of its size and license terms.

Generate the training examples:

```bash
python -m homework.generate_qa generate --data_dir data/train --output_file data/train/generated_qa_pairs.json
python -m homework.generate_captions generate --data_dir data/train --output_file data/train/generated_captions.json
```

Train and evaluate:

```bash
python -m homework.finetune train --output_dir vlm_model
python -m homework.clip train --output_dir clip_model
python -m homework.finetune test vlm_model
python -m homework.clip test clip_model
```

## Demo plan

For a LinkedIn video, capture a 20–30 second sequence of game frames. Overlay each question and the model's answer, then close with a VLM-versus-CLIP result card. Do not present the supplied game footage as an agent playing the game: this project understands race scenes; it does not control the kart.

## Limitations

This is a coursework-scale visual QA system trained on a game-specific dataset. Its metrics do not demonstrate general visual reasoning outside SuperTuxKart scenes.
