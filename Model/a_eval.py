import os
import re
import json
import argparse

import torch
from tqdm import tqdm

import a_model
import a_bioclip_VE
import a_gpt2_TD

MARKER = "This species is likely"
REPORT_EVERY = 100


def normalize(s):
    """lowercase, drop punctuation/underscores, collapse whitespace"""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def extract_species(text):
    """Pull the species name out of 'blah blah. This species is likely X.'"""
    if MARKER not in text:
        return None
    rest = text.split(MARKER, 1)[1]
    rest = re.split(r"[.\n]", rest.strip())[0]  # up to first period/newline
    return normalize(rest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="/kaggle/working/eval_results.json")
    args = parser.parse_args()

    with open(args.test_json, "r") as f:
        data = json.load(f)

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gpt2_TD.GPT2Decoder("openai-community/gpt2"),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.cuda()
    model.eval()  # also turns off LoRA dropout

    results = []
    exact = 0
    contains = 0
    no_marker = 0
    last_reported = 0

    for start in tqdm(range(0, len(data), args.batch_size)):
        batch = data[start:start + args.batch_size]
        paths = [os.path.join(a_model.IMAGES_ROOT, item["imagePath"]) for item in batch]

        with torch.no_grad():
            outputs = model.generate(paths)

        for item, out in zip(batch, outputs):
            gt_species = extract_species(item["gt"])
            pred_species = extract_species(out)

            if pred_species is None:
                no_marker += 1
                is_exact = is_contains = False
            else:
                is_exact = pred_species == gt_species
                is_contains = bool(gt_species) and (
                    gt_species in pred_species or pred_species in gt_species
                )

            exact += is_exact
            contains += is_contains

            results.append({
                "imagePath": item["imagePath"],
                "gt_species": gt_species,
                "pred_species": pred_species,
                "correct": is_exact,
            })

        # running accuracy every REPORT_EVERY images (batches may straddle the boundary)
        processed = len(results)
        if processed // REPORT_EVERY > last_reported // REPORT_EVERY:
            tqdm.write(
                f"[{processed}/{len(data)}] "
                f"exact acc: {exact / processed:.4f} ({exact}/{processed}) | "
                f"lenient acc: {contains / processed:.4f}"
            )
            last_reported = processed

    n = len(data)
    print(f"\nSamples:                     {n}")
    print(f"Exact species accuracy:      {exact / n:.4f} ({exact}/{n})")
    print(f"Lenient (substring) accuracy:{contains / n:.4f} ({contains}/{n})")
    print(f"Outputs missing the marker:  {no_marker}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Per-sample results saved to {args.out}")


if __name__ == "__main__":
    main()