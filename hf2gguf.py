#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для слияния базовой модели с LoRA-адаптером и конвертации в GGUF.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def log(msg):
    """Вывод сообщения в stderr."""
    print(f"[hf2gguf] {msg}", file=sys.stderr, flush=True)


def run_command(cmd):
    """Запуск команды с выводом и проверкой."""
    log(f"Выполняем: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ОШИБКА: {result.stderr}")
        sys.exit(1)
    return result


def merge_model(base_repo, peft_repo, output_dir, hf_token):
    """Загрузка и слияние базовой модели с LoRA-адаптером."""
    log("Загрузка базовой модели...")
    model = AutoModelForCausalLM.from_pretrained(
        base_repo,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    log("Загрузка LoRA-адаптера...")
    model = PeftModel.from_pretrained(
        model,
        peft_repo,
        token=hf_token,
    )

    log("Слияние весов...")
    model = model.merge_and_unload()

    log("Сохранение объединённой модели...")
    model.save_pretrained(output_dir, safe_serialization=True)

    log("Сохранение токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(base_repo, token=hf_token)
    tokenizer.save_pretrained(output_dir)

    return model


def convert_to_gguf(merged_dir, gguf_file, llama_cpp_dir):
    """Конвертация модели в формат GGUF."""
    llama_cpp_path = Path(llama_cpp_dir).expanduser().resolve()

    # Клонирование llama.cpp если нужно
    if not llama_cpp_path.exists():
        log("Клонирование llama.cpp...")
        run_command([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp",
            str(llama_cpp_path)
        ])

    # Конвертация в GGUF
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        log("ОШИБКА: Скрипт конвертации не найден")
        sys.exit(1)

    log("Конвертация в GGUF (float16)...")
    run_command([
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_file),
        "--outtype", "f16"
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Слияние базовой модели с LoRA и конвертация в GGUF"
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Базовая модель (HF репо или локальный путь)"
    )

    parser.add_argument(
        "--lora",
        required=True,
        help="LoRA-адаптер (HF репо или локальный путь)"
    )

    parser.add_argument(
        "--output",
        default="./merged_model",
        help="Директория для сохранения (по умолчанию: ./merged_model)"
    )

    parser.add_argument(
        "--gguf-name",
        default="model_f16.gguf",
        help="Имя GGUF-файла (по умолчанию: model_f16.gguf)"
    )

    parser.add_argument(
        "--llama-cpp",
        default="./llama.cpp",
        help="Путь к llama.cpp (по умолчанию: ./llama.cpp)"
    )

    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace токен (или переменная HF_TOKEN)"
    )

    args = parser.parse_args()

    # Создание директории вывода
    output_path = Path(args.output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Слияние модели
        merge_model(args.base, args.lora, output_path, args.hf_token)

        # Конвертация в GGUF
        gguf_path = output_path / args.gguf_name
        convert_to_gguf(output_path, gguf_path, args.llama_cpp)

        log(f"Готово: {gguf_path}")

    except KeyboardInterrupt:
        log("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        log(f"ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
