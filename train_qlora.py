import argparse
import os
import sys
import subprocess
import torch  # Необходим для указания типа данных при квантовании
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
from tg_cred import TG_API_KEY, TG_CHAT_ID

def parse_args():
    '''
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
      --target_modules: указать ['q_proj', 'v_proj'] как маодули для LoRA

      Параметры квантования:
      --int4: если указан, используется 4-битное квантование (int4);
      --int8: если указан, используется 8-битное квантование (int8).

    Важно:
      - Обязательно укажите ровно один из флагов квантования: либо --int4, либо --int8.
      - Если указаны одновременно --int4 и --int8, программа завершится с ошибкой.
      - Если не указан ни один из флагов, программа также завершится с ошибкой.
    '''
    parser = argparse.ArgumentParser(
        description='Fine-tune a Causal LM with LoRA in 8-bit or 4-bit mode, '
                    'with TensorBoard logging and checkpoint-based resume.'
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
        help='Размер контекстного окна для токенизации (длина чанка).'
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
        help='Путь к директории с чекпоинтом для возобновления обучения (например, \'./lora-8bit-clm/checkpoint-1000\').'
    )
    parser.add_argument(
        '--log_dir', type=str, default=None,
        help='Директория для логов TensorBoard.'
    )
    parser.add_argument(
        '--save_freq', type=int, default=None,
        help='Сохранять чекпоинт каждые N шагов (по умолчанию 200).'
    )
    parser.add_argument(
        '--lr_scheduler_type', type=str, default=None,
        help='Тип планировщика скорости обучения.'
    )
    parser.add_argument(
        '--message', type=str, default=None,
        help='Способ уведомления: \'notify\' для уведомлений Ubuntu или \'tg\' для уведомлений в Telegram.'
    )
    parser.add_argument(
        '--warmup_ratio', type=float, default=0.05,
        help='Коэффициент разогрева (warmup ratio) для планировщика скорости обучения.'
    )
    parser.add_argument(
        '--int4', action='store_true',
        help='Использовать 4-битное квантование (int4).'
    )
    parser.add_argument(
        '--int8', action='store_true',
        help='Использовать 8-битное квантование (int8).'
    )
    parser.add_argument(
        '--target_modules', type=bool, default=False,
        help='Указать [\'q_proj\', \'v_proj\'] как маодули для LoRA'
    )   
    args = parser.parse_args()
    
    # Проверка флагов квантования
    if args.int4 and args.int8:
        print('Ошибка: нельзя указывать одновременно --int4 и --int8')
        sys.exit(1)
    if not (args.int4 or args.int8):
        print('Ошибка: необходимо указать один из флагов квантования: --int4 или --int8')
        sys.exit(1)
    
    return args

def send_telegram_notification(message, tg_token, tg_chatid):
    '''
    Отправляет уведомление через Telegram.
    '''
    import requests
    url = f'https://api.telegram.org/bot{tg_token}/sendMessage'
    data = {'chat_id': tg_chatid, 'text': message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print('Ошибка отправки Telegram уведомления:', e)

class NotificationCallback(TrainerCallback):
    '''
    Колбэк для отправки уведомлений при старте, каждые 1000 шагов и при завершении обучения.
    '''
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

def get_quantization_config(args):
    '''
    Возвращает конфигурацию квантования (BitsAndBytesConfig) в зависимости от выбранного флага.
    
    Если указан --int4, используется 4-битное квантование (int4).
    Если указан --int8, используется 8-битное квантование (int8).
    '''
    if args.int4:
        print(f'Loading base model \'{args.model_name}\' in 4-bit mode...')
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )
    elif args.int8:
        print(f'Loading base model \'{args.model_name}\' in 8-bit mode...')
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        # Данный случай не должен возникнуть, так как проверка выполнена ранее
        print('Ошибка: режим квантования не выбран. Используйте --int4 или --int8.')
        sys.exit(1)

def main():
    args = parse_args()
    
    # Проверка корректности параметра --message
    if args.message is not None and args.message not in ['notify', 'tg']:
        print('Ошибка: параметр --message должен быть \'notify\' или \'tg\'')
        sys.exit(1)

    # Настройка функции отправки уведомлений, если уведомления включены
    send_notification = None
    if args.message == 'notify':
        def send_notification(msg):
            subprocess.Popen(['notify-send', msg])
    elif args.message == 'tg':
        def send_notification(msg):
            send_telegram_notification(msg, TG_API_KEY, TG_CHAT_ID)
    
    if args.log_dir is not None:
        print('Параметр log_dir устарел')
        sys.exit(1)

    # 1. Загрузка датасета (каждая строка файла -> один пример)
    data_files = {'train': args.train_file}
    raw_datasets = load_dataset('text', data_files=data_files)

    # 2. Загрузка токенизатора из базовой модели
    print(f'Loading tokenizer from base model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Функция токенизации текста (без обрезки)
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=False,
            max_length=None,
            return_attention_mask=False
        )

    tokenized_dataset = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    # 4. Функция, разбивающая длинный текст на чанки фиксированной длины
    def chunk_lines(examples):
        all_input_ids = examples['input_ids']
        result_input_ids = []
        result_attention_mask = []
        for input_ids in all_input_ids:
            for i in range(0, len(input_ids), args.block_size):
                chunk = input_ids[i: i + args.block_size]
                result_input_ids.append(chunk)
                result_attention_mask.append([1] * len(chunk))
        return {
            'input_ids': result_input_ids,
            'attention_mask': result_attention_mask
        }

    lm_dataset = tokenized_dataset.map(
        chunk_lines,
        batched=True,
        remove_columns=tokenized_dataset['train'].column_names
    )

    # 6. DataCollator для задач causal LM (без маскирования токенов)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 7. Загрузка модели с квантованием: режим 8-бит или 4-бит выбирается согласно указанному флагу
    bnb_config = get_quantization_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map='auto'
    )

    # 8. Подключение LoRA
    if args.target_modules:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'v_proj'],
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 9. Настройка параметров обучения (TrainingArguments)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'finetuned'),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1,
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
        logging_dir=os.path.join(args.output_dir, 'logs')
    )

    # 10. Используем только тренировочный датасет
    train_dataset = lm_dataset['train']

    # 11. Создание Trainer и добавление колбэка уведомлений (если включены)
    callbacks = []
    if send_notification is not None:
        callbacks.append(NotificationCallback(send_notification))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # 12. Запуск обучения (с учетом возможности возобновления с чекпоинта)
    trainer.train(resume_from_checkpoint=args.checkpoint_dir)

    # 13. Сохранение финальной модели
    trainer.save_model()
    print('Обучение завершено. Модель сохранена в:', args.output_dir)

if __name__ == '__main__':
    main()
