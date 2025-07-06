import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a fine-tuned 8-bit LoRA model"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the directory with the fine-tuned model'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='\n',
        help='Initial text prompt to start generation'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-K sampling'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (e.g., cuda:0 or cpu). If not set, uses device_map=auto.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Настройка квантования для 8-bit модели
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=quant_config,
        device_map='auto' if args.device is None else None,
        torch_dtype=torch.float16
    )

    # Перенос на указанный девайс, если задан
    if args.device:
        model.to(args.device)
        device = torch.device(args.device)
    else:
        # transformers сам распределит модель
        device = torch.device(model.device_map.popitem()[1] if hasattr(model, 'device_map') else 'cpu')

    # Токенизация промпта
    inputs = tokenizer(
        args.prompt,
        return_tensors='pt'
    ).to(device)

    # Генерация
    outputs = model.generate(
        **inputs,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(generated)


if __name__ == '__main__':
    main()

