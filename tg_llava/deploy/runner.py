import argparse
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re
import torch
import transformers

from tg_llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from tg_llava.conversation import conv_templates, SeparatorStyle
from tg_llava.model.builder import load_pretrained_model
from tg_llava.utils import disable_torch_init
from tg_llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    model_name = model_name + "-qwen2-cross"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "qwen2" in model_name.lower():
        conv_mode = "qwen_2"
    elif "llama3" in model_name.lower():
        conv_mode = "llama_3"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    tokenizer_clip = transformers.AutoTokenizer.from_pretrained(args.text_model_path)
    text_guide_ids = torch.tensor(tokenizer_clip(qs.replace('<image>','').strip()).input_ids, dtype=torch.long).cuda()
    
    img_resized = torch.nn.functional.interpolate(images_tensor, scale_factor=2, mode='bilinear', align_corners=False).squeeze(0)
    
    img_patches = [
        torch.stack([img_resized[:, 0:336, 0:336],
        img_resized[:, 0:336, 336:336*2],
        img_resized[:, 336:336*2, 0:336],
        img_resized[:, 336:336*2, 336:336*2]])
    ]
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            text_guide_ids,
            img_patches,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":

    tg_llava_model_path = "/workspace/TG-LLaVA"
    clip_model_path = "/workspace/clip-vit-large-patch14-336"
    
    prompt = "What is in this image?"
    image_file = '/workspace/test.jpg'

    args = type('Args', (), {
        "model_path": tg_llava_model_path,
        "text_model_path": clip_model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)