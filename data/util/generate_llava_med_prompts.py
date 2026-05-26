"""
Generate LLaVA-Med prompts for HAM10000 train/val/test splits.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys

import torch

LLAVA_ROOT = os.environ.get("LLAVA_ROOT", "/root/autodl-tmp/data/util/LLaVA-main")
DEFAULT_MODEL_CACHE = "/root/autodl-tmp/model"
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

try:
    from transformers import LlavaForConditionalGeneration
    MODEL_CLASS = LlavaForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "LlavaForConditionalGeneration is required. Update transformers to a version that provides it."
    ) from exc


BENIGN_MALIGNANT = {
    "akiec": "malignant",
    "bcc": "malignant",
    "bkl": "benign",
    "df": "benign",
    "mel": "malignant",
    "nv": "benign",
    "vasc": "benign",
}

DISEASE_NAMES = {
    "akiec": "actinic keratoses or intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesion",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}


def resolve_image_path(img_path: str, data_root: str) -> str:
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path
    if img_path.startswith("data/local/"):
        candidate = os.path.join(data_root, "data", img_path.replace("data/local/", ""))
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(data_root, img_path)
    if os.path.exists(candidate):
        return candidate
    return img_path


def build_instruction(
    disease: str,
    malignancy: str,
    skin_tone: str,
    sex: str,
    age: Optional[float],
    anatomical: str,
) -> str:
    age_str = "unknown age" if pd.isna(age) else f"{int(age)} years old"
    sex_str = sex if sex and sex != "nan" else "unknown sex"
    anatomical_str = anatomical if anatomical and anatomical != "nan" else "unspecified location"
    skin_str = skin_tone if skin_tone else "unspecified skin tone"
    return (
        "Write concise clinical sentences describing this dermoscopic image. "
        f"Diagnosis: {disease} ({malignancy}). "
        f"Patient: {sex_str}, {age_str}. "
        f"Location: {anatomical_str}. "
        f"Skin tone: {skin_str}. "
        "Mention lesion color, structure, border, symmetry, and notable patterns. "
        "Avoid speculation, keep it factual and descriptive."
    )


def clean_response(text: str) -> str:
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]
    text = text.replace("USER:", "").replace("ASSISTANT:", "").strip()
    return " ".join(text.split())


def load_skin_tone_map(skin_tone_csv: Optional[str]) -> Dict[str, str]:
    if not skin_tone_csv:
        return {}
    df = pd.read_csv(skin_tone_csv)
    if "image_id" not in df.columns or "skin_tone" not in df.columns:
        raise ValueError("skin tone CSV must contain columns: image_id, skin_tone")
    return dict(zip(df["image_id"], df["skin_tone"]))


def generate_prompts(
    split_csv: str,
    meta_df: pd.DataFrame,
    skin_tone_map: Dict[str, str],
    data_root: str,
    tokenizer,
    model,
    image_processor,
    device: str,
    output_path: str,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: Optional[int],
) -> None:
    split_df = pd.read_csv(split_csv)
    split_df["image_id"] = split_df["img_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    merged = split_df.merge(meta_df, on="image_id", how="left")

    if limit is not None:
        merged = merged.head(limit).copy()

    prompts = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Generating prompts for {split_csv}"):
        disease_key = row["class"]
        disease_name = DISEASE_NAMES.get(disease_key, disease_key)
        malignancy = BENIGN_MALIGNANT.get(disease_key, "unknown")
        skin_tone = skin_tone_map.get(row["image_id"], "unspecified")
        instruction = build_instruction(
            disease=disease_name,
            malignancy=malignancy,
            skin_tone=skin_tone,
            sex=str(row.get("sex", "unknown")),
            age=row.get("age"),
            anatomical=str(row.get("localization", "")),
        )

        image_path = resolve_image_path(row["img_path"], data_root)
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(device)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        with torch.no_grad():
            generated = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        prompts.append(clean_response(decoded))

    merged = merged.iloc[: len(prompts)].copy()
    merged["llava_prompt"] = prompts
    merged.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp")
    parser.add_argument("--model-id", type=str, default="liuhaotian/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-cache", type=str, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-name", type=str, default="llava-v1.5-7b")
    parser.add_argument("--meta-csv", type=str, default="/root/autodl-tmp/data/HAM10000/input/HAM10000_metadata.csv")
    parser.add_argument("--skin-tone-csv", type=str, default=None)
    parser.add_argument("--split-train", type=str, default="/root/autodl-tmp/data/HAM10000/input/metadata_train.csv")
    parser.add_argument("--split-val", type=str, default="/root/autodl-tmp/data/HAM10000/input/metadata_val.csv")
    parser.add_argument("--split-test", type=str, default="/root/autodl-tmp/data/HAM10000/input/metadata_test.csv")
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.model_cache:
        os.environ.setdefault("HF_HOME", args.model_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.model_cache)

    model_path = args.model_dir or args.model_id
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=args.model_name,
        device_map="auto",
        device=args.device,
    )
    model.eval()

    meta_df = pd.read_csv(args.meta_csv)
    meta_df = meta_df.rename(columns={"image_id": "image_id"})
    skin_tone_map = load_skin_tone_map(args.skin_tone_csv)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split_train and os.path.exists(args.split_train):
        generate_prompts(
            split_csv=args.split_train,
            meta_df=meta_df,
            skin_tone_map=skin_tone_map,
            data_root=args.data_root,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            device=args.device,
            output_path=os.path.join(args.output_dir, "metadata_train_llava.csv"),
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
        )

    if args.split_val and os.path.exists(args.split_val):
        generate_prompts(
            split_csv=args.split_val,
            meta_df=meta_df,
            skin_tone_map=skin_tone_map,
            data_root=args.data_root,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            device=args.device,
            output_path=os.path.join(args.output_dir, "metadata_val_llava.csv"),
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
        )

    if args.split_test and os.path.exists(args.split_test):
        generate_prompts(
            split_csv=args.split_test,
            meta_df=meta_df,
            skin_tone_map=skin_tone_map,
            data_root=args.data_root,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            device=args.device,
            output_path=os.path.join(args.output_dir, "metadata_test_llava.csv"),
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
