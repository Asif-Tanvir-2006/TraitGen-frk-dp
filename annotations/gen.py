import json
import random
import argparse


def keypoint_sentence(keypoint, attributes):
    """'<value> <key>, <value> <key> <keypoint>' (same as the COCO loader)"""
    parts = []
    for attribute in attributes:
        for key, value in attribute.items():
            selected = value[0] if isinstance(value, list) else value
            parts.append(f"{selected} {key}")
    return ", ".join(parts) + f" {keypoint}"


def drop_random_keypoints(kp_dict, drop_parts, rng):
    """Remove `drop_parts` random keypoints (parts) with all their attributes."""
    updated = dict(kp_dict)
    num_drop = min(drop_parts, len(updated))
    if num_drop > 0:
        for kp in rng.sample(list(updated.keys()), num_drop):
            del updated[kp]
    return updated


def build_caption(category, drop_parts, rng):
    kp_attrs = category.get("keypoint_attributes_by_category", {})
    kp_attrs = drop_random_keypoints(kp_attrs, drop_parts, rng)

    sentences = [keypoint_sentence(kp, attrs) for kp, attrs in kp_attrs.items()]
    attributes = "; ".join(sentences)
    if attributes:
        attributes += "."

    # No "<|endoftext|>" here: GPT2Decoder appends the EOS token itself.
    return f'It is a species of "{category["name"]}" as it has {attributes}'.rstrip()


def convert(ann_path, out_path, drop_parts=0, copies=1, seed=0):
    with open(ann_path, "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    rng = random.Random(seed)

    records = []
    for _ in range(copies):
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img = images[ann["image_id"]]
            file_name = img.get("filename") or img.get("file_name")
            records.append({
                "imagePath": file_name,  # must be relative to IMAGES_ROOT in a_model.py
                # a fresh random drop for every sample (and every copy)
                "gt": build_caption(categories[ann["category_id"]], drop_parts, rng),
            })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"{ann_path} -> {out_path}: {len(records)} samples "
          f"(drop_parts={drop_parts}, copies={copies})")
    if records:
        print("Example:")
        print(f"  imagePath: {records[0]['imagePath']}")
        print(f"  gt: {records[0]['gt']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ann", required=True)
    parser.add_argument("--test_ann", required=True)
    parser.add_argument("--train_out", default="/kaggle/working/train_new.json")
    parser.add_argument("--test_out", default="/kaggle/working/test_new.json")
    parser.add_argument("--train_drop_parts", type=int, default=0,
                        help="number of random parts (keypoints) to drop per training sample")
    parser.add_argument("--test_drop_parts", type=int, default=0,
                        help="number of random parts to drop per test sample")
    parser.add_argument("--train_copies", type=int, default=1,
                        help="how many times to repeat the train set, each with different random drops")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    convert(args.train_ann, args.train_out, args.train_drop_parts, args.train_copies, args.seed)
    convert(args.test_ann, args.test_out, args.test_drop_parts, 1, args.seed + 1)