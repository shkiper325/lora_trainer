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
from peft import LoraConfig, get_peft_model, TaskType
from tg_cred import TG_API_KEY, TG_CHAT_ID
from peft import prepare_model_for_kbit_training
from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model, BnbQuantizationConfig
from transformers import AutoConfig


def parse_args():
    """
    Обрабатывает аргументы командной строки для обучения модели с LoRA.

    Основные параметры:
      --model_name: базовая модель (на HF Hub или локальный путь);
      --train_file: путь к обучающему текстовому файлу (один пример в строке);
      --block_size: размер контекстного окна для токенизации;
      --output_dir: директория для сохранения модели и чекпоинтов;
      --num_train_epochs, --batch_size, --grad_accum_steps, --learning_rate: гиперпараметры обучения;
      --lora_r, --lora_alpha, --lora_dropout: параметры конфигурации LoRA;
      --checkpoint_dir: путь для возобновления обучения с определённого чекпоинта;
      --save_freq: частота сохранения чекпоинтов;
      --lr_scheduler_type: тип планировщика скорости обучения;
      --message: способ уведомления ('notify' для уведомлений Ubuntu или 'tg' для Telegram);
      --warmup_ratio: коэффициент разогрева (warmup ratio) для планировщика скорости обучения.
      --target_modules: указать ['q_proj', 'v_proj'] как модули для LoRA.

    По умолчанию используется 8-битное квантование (load_in_8bit=True).
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune a Causal LM with LoRA in 8-bit mode, '
                    'with TensorBoard logging and checkpoint resume.'
    )
    parser.add_argument(
        '--model_name', type=str, default='sberbank-ai/rugpt3large_based_on_gpt2',
        help='Базовая модель для загрузки (на HF Hub или локальный путь).'
    )
    parser.add_argument(
        '--train_file', type=str, required=True,
        help='Путь к обучающему текстовому файлу (один пример на строке).'
    )
    parser.add_argument(
        '--block_size', type=int, default=512,
        help='Размер контекстного окна для токенизации.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./lora-8bit-clm',
        help='Директория для хранения финальной модели и чекпоинтов.'
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=3,
        help='Количество эпох обучения.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Размер батча для обучения на каждом устройстве.'
    )
    parser.add_argument(
        '--grad_accum_steps', type=int, default=8,
        help='Количество шагов градиентного накопления.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=2e-4,
        help='Начальная скорость обучения.'
    )
    parser.add_argument(
        '--lora_r', type=int, default=8,
        help='Параметр rank для LoRA.'
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=32,
        help='Параметр alpha для LoRA.'
    )
    parser.add_argument(
        '--lora_dropout', type=float, default=0.05,
        help='Dropout для LoRA.'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default=None,
        help='Путь к директории с чекпоинтом для возобновления обучения.'
    )
    parser.add_argument(
        '--save_freq', type=int, default=200,
        help='Сохранять чекпоинт каждые N шагов.'
    )
    parser.add_argument(
        '--lr_scheduler_type', type=str, default=None,
        help='Тип планировщика скорости обучения.'
    )
    parser.add_argument(
        '--message', type=str, choices=['notify', 'tg'], default=None,
        help='Способ уведомления: notify или tg.'
    )
    parser.add_argument(
        '--warmup_ratio', type=float, default=0.05,
        help='Коэффициент разогрева для планировщика скорости обучения.'
    )
    parser.add_argument(
        '--default_target', type=bool, default=False,
        help='Коэффициент разогрева для планировщика скорости обучения.'
    )


    return parser.parse_args()


def send_telegram_notification(message, tg_token, tg_chatid):
    import requests
    url = f'https://api.telegram.org/bot{tg_token}/sendMessage'
    data = {'chat_id': tg_chatid, 'text': message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print('Ошибка отправки Telegram уведомления:', e)


class NotificationCallback(TrainerCallback):
    def __init__(self, send_notification_func):
        self.send_notification = send_notification_func

    def on_train_begin(self, args, state, control, **kwargs):
        self.send_notification('Обучение начато')
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 1000 == 0:
            self.send_notification(f'Пройдено шагов обучения: {state.global_step}')
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.send_notification('Обучение завершено')
        return control


def main():
    args = parse_args()

    # Настройка уведомлений
    send_notification = None
    if args.message == 'notify':
        send_notification = lambda msg: subprocess.Popen(['notify-send', msg])
    elif args.message == 'tg':
        send_notification = lambda msg: send_telegram_notification(msg, TG_API_KEY, TG_CHAT_ID)

    # Загрузка данных и токенизатора
    raw_datasets = load_dataset('text', data_files={'train': args.train_file})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=False, return_attention_mask=False)

    tokenized = raw_datasets.map(tokenize_function, batched=True, remove_columns=['text'])

    # Разбиение строе на чанки
    def chunk_lines(examples):
        result = {'input_ids': [], 'attention_mask': []}
        for ids in examples['input_ids']:
            for i in range(0, len(ids), args.block_size):
                chunk = ids[i:i + args.block_size]
                result['input_ids'].append(chunk)
                result['attention_mask'].append([1] * len(chunk))
        return result

    lm_dataset = tokenized.map(chunk_lines, batched=True, remove_columns=tokenized['train'].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Настройка квантования
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map='cpu',
        torch_dtype=torch.float16
    )

    torch.cuda.empty_cache()
    model.to("cuda:0")  
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA конфигурация
    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    # Если не  указаны модули, то используем q_proj и v_proj
    if args.default_target == True:
        lora_kwargs['target_modules'] = ['q_proj', 'v_proj']
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'finetuned'),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=True,
        optim='paged_adamw_8bit',
        save_strategy='steps',
        save_steps=args.save_freq,
        logging_steps=5,
        report_to='tensorboard',
        logging_dir=os.path.join(args.output_dir, 'logs'),
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset['train'],
        data_collator=data_collator,
        callbacks=[NotificationCallback(send_notification)] if send_notification else None
    )

    if args.checkpoint_dir is not None:
        trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    else:
        trainer.train()

    trainer.save_model()
    print('Обучение завершено. Модель сохранена в:', args.output_dir)


if __name__ == '__main__':
    main()
