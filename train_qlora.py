#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обучения языковой модели с QLoRA (8-bit квантование + LoRA).
"""

import argparse
import os
import sys
import subprocess

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def log(msg):
    """Вывод сообщения."""
    print(f"[train] {msg}", flush=True)


def send_telegram(message, token, chat_id):
    """Отправка уведомления в Telegram."""
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        log(f"Ошибка отправки в Telegram: {e}")


def send_notify(message):
    """Отправка уведомления через notify-send."""
    try:
        subprocess.run(["notify-send", "Train QLoRA", message], check=False)
    except Exception as e:
        log(f"Ошибка notify-send: {e}")


class NotificationCallback(TrainerCallback):
    """Коллбэк для отправки уведомлений в процессе обучения."""

    def __init__(self, notify_func):
        self.notify = notify_func

    def on_train_begin(self, args, state, control, **kwargs):
        self.notify("Обучение начато")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 1000 == 0:
            self.notify(f"Шаг {state.global_step}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.notify("Обучение завершено")
        return control


def tokenize_and_chunk(dataset, tokenizer, block_size):
    """Токенизация и разбиение текста на чанки."""

    # Токенизация
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False, return_attention_mask=False)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Разбиение на чанки
    def chunk_fn(examples):
        result = {"input_ids": [], "attention_mask": []}
        for ids in examples["input_ids"]:
            for i in range(0, len(ids), block_size):
                chunk = ids[i:i + block_size]
                result["input_ids"].append(chunk)
                result["attention_mask"].append([1] * len(chunk))
        return result

    return tokenized.map(chunk_fn, batched=True, remove_columns=tokenized["train"].column_names)


def create_model(model_name, lora_r, lora_alpha, lora_dropout, target_modules=None):
    """Создание квантованной модели с LoRA."""

    # Конфигурация 8-bit квантования
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Загрузка модели
    log("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)

    # Конфигурация LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Обучение языковой модели с QLoRA"
    )

    # Основные параметры
    parser.add_argument(
        "--model",
        type=str,
        default="sberbank-ai/rugpt3large_based_on_gpt2",
        help="Базовая модель (HF репо или локальный путь)"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Текстовый файл с данными (одна строка = один пример)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./qlora_output",
        help="Директория для сохранения (по умолчанию: ./qlora_output)"
    )

    # Параметры обучения
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Количество эпох (по умолчанию: 3)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Размер батча (по умолчанию: 1)"
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Шаги накопления градиента (по умолчанию: 8)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (по умолчанию: 2e-4)"
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=0.05,
        help="Warmup ratio (по умолчанию: 0.05)"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Размер контекстного окна (по умолчанию: 512)"
    )

    # Параметры LoRA
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (по умолчанию: 8)"
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (по умолчанию: 32)"
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (по умолчанию: 0.05)"
    )

    parser.add_argument(
        "--lora-targets",
        type=str,
        nargs="+",
        default=None,
        help="Целевые модули для LoRA (например: q_proj v_proj)"
    )

    # Дополнительные параметры
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения"
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Сохранять каждые N шагов (по умолчанию: 200)"
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        help="Тип LR scheduler (по умолчанию: cosine)"
    )

    # Уведомления
    parser.add_argument(
        "--notify",
        type=str,
        choices=["telegram", "desktop"],
        default=None,
        help="Способ уведомлений: telegram или desktop"
    )

    parser.add_argument(
        "--tg-token",
        type=str,
        default=os.getenv("TG_TOKEN"),
        help="Telegram bot token (или переменная TG_TOKEN)"
    )

    parser.add_argument(
        "--tg-chat",
        type=str,
        default=os.getenv("TG_CHAT_ID"),
        help="Telegram chat ID (или переменная TG_CHAT_ID)"
    )

    args = parser.parse_args()

    # Настройка уведомлений
    notify_func = None
    if args.notify == "telegram":
        if not args.tg_token or not args.tg_chat:
            log("ОШИБКА: Для Telegram нужны --tg-token и --tg-chat")
            sys.exit(1)
        notify_func = lambda msg: send_telegram(msg, args.tg_token, args.tg_chat)
    elif args.notify == "desktop":
        notify_func = send_notify

    # Загрузка данных
    log("Загрузка данных...")
    raw_data = load_dataset("text", data_files={"train": args.data})

    # Загрузка токенизатора
    log("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Подготовка датасета
    log("Токенизация и чанкинг...")
    lm_dataset = tokenize_and_chunk(raw_data, tokenizer, args.block_size)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Создание модели
    model = create_model(
        args.model,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_targets
    )

    # Конфигурация обучения
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        lr_scheduler_type=args.scheduler,
        fp16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=5,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output, "logs"),
        gradient_checkpointing=True
    )

    # Создание trainer
    callbacks = [NotificationCallback(notify_func)] if notify_func else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Обучение
    try:
        if args.checkpoint:
            log(f"Продолжение обучения с {args.checkpoint}")
            trainer.train(resume_from_checkpoint=args.checkpoint)
        else:
            log("Начало обучения...")
            trainer.train()

        # Сохранение модели
        output_path = os.path.join(args.output, "final")
        trainer.save_model(output_path)
        log(f"Готово: {output_path}")

    except KeyboardInterrupt:
        log("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        log(f"ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
