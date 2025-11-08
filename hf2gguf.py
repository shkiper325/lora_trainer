#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для слияния базовой модели с LoRA-адаптером и конвертации в GGUF.

Для систем с ограниченной памятью (32GB RAM):
  python hf2gguf.py --base MODEL --lora ADAPTER --cpu-only --max-memory 20GB --offload

Параметры оптимизации:
  --cpu-only       - работа без GPU
  --max-memory     - лимит RAM (рекомендуется 20-24GB для 32GB систем)
  --offload        - сохранение части модели на диск (медленнее, но безопаснее)
"""

import argparse
import gc
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


def merge_model(base_repo, peft_repo, output_dir, hf_token, cpu_only=False, max_mem="24GB", offload=False):
    """Загрузка и слияние базовой модели с LoRA-адаптером."""

    # Настройки для CPU или GPU
    if cpu_only:
        log(f"Режим CPU: лимит памяти {max_mem}")
        device_map = {"": "cpu"}
        torch_dtype = torch.float16
        max_memory = {"cpu": max_mem}

        # Offload для больших моделей
        offload_folder = None
        if offload:
            offload_folder = str(output_dir / "offload_tmp")
            log(f"Disk offload включён: {offload_folder}")
    else:
        device_map = "auto"
        torch_dtype = torch.float16
        max_memory = None
        offload_folder = None

    log("Загрузка базовой модели...")
    model = AutoModelForCausalLM.from_pretrained(
        base_repo,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    log("Загрузка LoRA-адаптера...")
    peft_kwargs = {"token": hf_token}

    # Передача параметров offload для PEFT
    if offload_folder:
        peft_kwargs["offload_folder"] = offload_folder
        peft_kwargs["device_map"] = device_map
        peft_kwargs["max_memory"] = max_memory

    model = PeftModel.from_pretrained(
        model,
        peft_repo,
        **peft_kwargs
    )

    log("Слияние весов...")
    model = model.merge_and_unload()

    log("Сохранение объединённой модели...")
    model.save_pretrained(output_dir, safe_serialization=True)

    log("Сохранение токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(base_repo, token=hf_token)
    tokenizer.save_pretrained(output_dir)

    # Освобождение памяти
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Очистка временной папки offload
    if offload_folder:
        import shutil
        offload_path = Path(offload_folder)
        if offload_path.exists():
            shutil.rmtree(offload_path)
            log("Временные файлы offload удалены")

    log("Модель сохранена, память освобождена")


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


def quantize_gguf(input_file, output_file, quant_type, llama_cpp_dir):
    """Квантизация GGUF-файла."""
    llama_cpp_path = Path(llama_cpp_dir).expanduser().resolve()
    quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"

    # Сборка llama.cpp если нужно
    if not quantize_bin.exists():
        log("Сборка llama.cpp...")
        build_dir = llama_cpp_path / "build"
        build_dir.mkdir(exist_ok=True)

        run_command(["cmake", "-DLLAMA_CURL=OFF", "-B", str(build_dir), "-S", str(llama_cpp_path)])
        run_command(["cmake",  "--build", str(build_dir), "--config", "Release", "-j"])

        if not quantize_bin.exists():
            log("ОШИБКА: Утилита квантизации не найдена после сборки")
            sys.exit(1)

    log(f"Квантизация в {quant_type}...")
    run_command([
        str(quantize_bin),
        str(input_file),
        str(output_file),
        quant_type
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

    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        help="Тип квантизации (q4_0, q4_1, q5_0, q5_1, q8_0 и т.д.)"
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Использовать только CPU (без GPU)"
    )

    parser.add_argument(
        "--max-memory",
        type=str,
        default="24GB",
        help="Максимальная память для CPU режима (по умолчанию: 24GB)"
    )

    parser.add_argument(
        "--offload",
        action="store_true",
        help="Использовать disk offload для больших моделей"
    )

    args = parser.parse_args()

    # Отключение GPU если требуется
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log("GPU отключен (CUDA_VISIBLE_DEVICES='')")

    # Создание директории вывода
    output_path = Path(args.output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Слияние модели
        merge_model(
            args.base,
            args.lora,
            output_path,
            args.hf_token,
            args.cpu_only,
            args.max_memory,
            args.offload
        )

        # Конвертация в GGUF
        gguf_path = output_path / args.gguf_name
        convert_to_gguf(output_path, gguf_path, args.llama_cpp)

        # Квантизация если требуется
        if args.quantize:
            # Генерация имени квантизированного файла
            if "_f16.gguf" in args.gguf_name:
                quant_name = args.gguf_name.replace("_f16.gguf", f"_{args.quantize}.gguf")
            else:
                quant_name = args.gguf_name.replace(".gguf", f"_{args.quantize}.gguf")

            quant_path = output_path / quant_name
            quantize_gguf(gguf_path, quant_path, args.quantize, args.llama_cpp)
            log(f"Готово: {quant_path}")
        else:
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
