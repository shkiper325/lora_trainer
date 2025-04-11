import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys

def parse_args():
    """
    Парсинг аргументов командной строки.

    Обязательные аргументы:
    --model_name : название модели.
    --model_dir  : путь к директории контрольной точки модели (например, '.../checkpoint-19727').

    Опциональные аргументы:
    --temperature   : температура генерации текста (по умолчанию 0.9).
    --max_new_tokens: максимальное число генерируемых токенов за один ход (по умолчанию 128).

    Режимы квантования (обязательное указание одного из):
    --int8 : использовать 8-битное представление чисел при инференсе.
    --int4 : использовать 4-битное представление чисел при инференсе.

    Валидация:
    - Нельзя указывать одновременно --int8 и --int4.
    - Обязательно нужно указать одну из опций квантования.
    """
    parser = argparse.ArgumentParser(description="Чат с LoRA 8/4-бит моделью.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Название модели")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Путь к директории контрольной точки LoRA модели (напр. '.../checkpoint-19727').")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Параметр температуры генерации текста.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Максимальное число генерируемых токенов за один ход.")
    parser.add_argument("--int8", action='store_true',
                        help="Использовать 8-битное представление чисел при инференсе.")
    parser.add_argument("--int4", action='store_true',
                        help="Использовать 4-битное представление чисел при инференсе.")

    args = parser.parse_args()

    # Валидация: проверка исключающих режимов квантования
    if args.int8 and args.int4:
        parser.error("Ошибка: одновременно указаны опции --int8 и --int4. Укажите только одну из них.")
    elif not args.int8 and not args.int4:
        parser.error("Ошибка: необходимо указать одну из опций --int8 или --int4 для задания размера чисел при инференсе.")

    return args

def main():
    # Парсинг аргументов командной строки
    args = parse_args()

    # Определение устройства: используем CUDA, если доступно, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Настройка конфигурации квантования в зависимости от выбранной опции
    if args.int8:
        # 8-битный режим квантования
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Квантование: 8-битный режим")
    elif args.int4:
        # 4-битный режим квантования
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Квантование: 4-битный режим")

    # Загрузка токенайзера и модели
    print(f"Загрузка модели из: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=bnb_config,
        device_map="auto",  # Автоматическое распределение на доступные устройства
    )
    model.eval()
    print("Модель загружена.")

    # Основной цикл для общения с моделью
    while True:
        try:
            user_input = input("$ ").strip()

            # Выход из программы по ключевым словам
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Завершение работы.")
                break

            # Обработка команды изменения температуры генерации
            if user_input.lower().startswith(("temperature", "temp", "t")):
                try:
                    temperature_value = float(user_input.split()[1])
                    args.temperature = temperature_value
                    print("Новая температура:", args.temperature)
                except (IndexError, ValueError):
                    print("Некорректное значение температуры.")
                continue

        except Exception as e:
            print("Ошибка при вводе команды:", e)
            continue

        # Токенизация ввода пользователя
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Генерация ответа модели
        with torch.no_grad():
            generated_ids = model.generate(
                inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )

        # Декодирование и вывод сгенерированного текста
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('\n')[0]
        print(full_text)

if __name__ == "__main__":
    main()
