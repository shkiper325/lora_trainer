#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для генерации текста с использованием обученной LoRA-модели.
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def log(msg):
    """Вывод сообщения."""
    print(f"[generate] {msg}", file=sys.stderr, flush=True)


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    """Генерация текста на основе промпта."""

    # Токенизация промпта
    inputs = tokenizer(prompt, return_tensors="pt")

    # Перенос на устройство модели
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Генерация
    log("Генерация текста...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Декодирование
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Генерация текста с использованием обученной LoRA-модели"
    )

    # Основные параметры
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Путь к обученной модели"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Текст для начала генерации (по умолчанию: пусто)"
    )

    # Параметры генерации
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Максимум новых токенов (по умолчанию: 256)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Температура сэмплирования (по умолчанию: 0.8)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K сэмплирование (по умолчанию: 50)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-P (nucleus) сэмплирование (по умолчанию: 0.95)"
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Штраф за повторы (по умолчанию: 1.1)"
    )

    # Параметры устройства
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Использовать только CPU (без GPU)"
    )

    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Отключить 8-bit квантизацию"
    )

    args = parser.parse_args()

    try:
        # Отключение GPU если требуется
        if args.cpu_only:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            log("GPU отключен")

        # Загрузка токенизатора
        log("Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        # Установка pad_token если отсутствует
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Настройка квантизации
        if args.no_quantize or args.cpu_only:
            log("Загрузка модели без квантизации...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16 if not args.cpu_only else torch.float32,
                device_map="cpu" if args.cpu_only else "auto",
                low_cpu_mem_usage=True
            )
        else:
            log("Загрузка модели с 8-bit квантизацией...")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )

        model.eval()

        # Генерация текста
        result = generate_text(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty
        )

        # Вывод результата
        print("\n" + "="*50)
        print(result)
        print("="*50 + "\n")

    except KeyboardInterrupt:
        log("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        log(f"ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
