import json
from pathlib import Path

import fire

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    with open(info_path) as f:
        info = json.load(f)

    ego_id = view_index
    ego_name = info["karts"][ego_id]
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    ego = next((kart for kart in karts if kart["instance_id"] == ego_id), None)
    ego_center = ego["center"] if ego else (img_width / 2.0, img_height / 2.0)

    captions = [
        f"{ego_name} is the ego car.",
        f"There are {len(karts)} karts in the scene.",
        f"The track is {info['track']}.",
    ]

    for kart in karts:
        if kart["instance_id"] == ego_id:
            continue
        left_right = "left" if kart["center"][0] < ego_center[0] else "right"
        front_back = (
            "in front of"
            if info["distance_down_track"][kart["instance_id"]] > info["distance_down_track"][ego_id]
            else "behind"
        )
        captions.extend(
            [
                f"{kart['kart_name']} is to the {left_right} of the ego car.",
                f"{kart['kart_name']} is {front_back} the ego car.",
            ]
        )

    return captions


def generate_dataset(data_dir: str = "data/train", output_file: str | None = None, max_frames: int | None = None):
    data_dir = Path(data_dir)
    output_path = Path(output_file) if output_file else data_dir / "generated_captions.json"
    caption_pairs = []

    for i, info_path in enumerate(sorted(data_dir.glob("*_info.json"))):
        if max_frames is not None and i >= max_frames:
            break
        with open(info_path) as f:
            info = json.load(f)
        base_name = info_path.stem.replace("_info", "")
        for view_index in range(len(info["detections"])):
            image_path = data_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_path.exists():
                continue
            image_file = f"{data_dir.name}/{base_name}_{view_index:02d}_im.jpg"
            for caption in generate_caption(str(info_path), view_index):
                caption_pairs.append({"image_file": image_file, "caption": caption})

    with open(output_path, "w") as f:
        json.dump(caption_pairs, f)

    print(f"Wrote {len(caption_pairs)} captions to {output_path}")


def check_caption(info_file: str, view_index: int):
    from matplotlib import pyplot as plt

    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_dataset})


if __name__ == "__main__":
    main()
