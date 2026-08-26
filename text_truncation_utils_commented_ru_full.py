"""
Text trimming / cleaning utilities extracted from utils(8).py.

Contains:
- name cleanup
- attribute filtering
- attribute prioritization
- attribute truncation
- token-based truncation helpers
- product text preparation

Unused functions are intentionally kept for future experiments.
"""

import re
import json
import numpy as np
from tqdm import tqdm

# ============================================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ, ПОСЧИТАННЫЕ ЗАРАНЕЕ НА TRAIN
# ============================================================

# 90-й перцентиль длины name.
# Максимальное количество символов, которое сохраняется из названия товара после безопасного обрезания.
# Значение выбрано на основе анализа длины названий в обучающих данных.
# Максимальная длина названия товара в символах после безопасного обрезания.
# Ограничение применяется для уменьшения влияния слишком длинных названий на токеновый бюджет модели.
NAME_MAX_CHARS = 94

# Атрибуты, которые встретились совсем редко.
MIN_ATTR_FREQ = 5

# Словарь частот ключей атрибутов:
# {
#     "Бренд": 1_500_000,
#     "Цвет": 900_000,
#     ...
# }
# Статистика частоты ключей атрибутов, собранная из обучающих данных.
# Используется только как слабый дополнительный сигнал в функции attr_priority().
# Словарь частот атрибутов.
# Используется как дополнительный слабый сигнал при ранжировании атрибутов: часто встречающиеся поля могут быть полезнее.
ATTR_FREQ = {}

# Общий ориентировочный бюджет для текста атрибутов одного товара
ATTR_TEXT_BUDGET_CHARS = 1000

# Одно значение может занимать максимум 10% этого бюджета
MAX_SINGLE_ATTR_SHARE = 0.10

# Абсолютный верхний предел значения атрибута
MAX_ATTR_VALUE_CHARS = 300

# Атрибуты, которые часто встречаются в названии товара
# Список атрибутов, значения которых разрешено удалять из названия,
# если они уже присутствуют в attributes и не являются важным идентификатором.
NAME_DEDUP_ATTRS = (
    "бренд",
    "brand",
    "производитель",

    "цвет товара",
    "цвет",
    "color",
    "название цвета",

    "бренд в одежде и обуви",

    "размер",
    "size",

    "объем, мл",
    "объем",
    "объём",

    "вес товара, г",

    "материал",

    "комплектация",

    "проба",

    "диоптрии",

    "оптическая сила",

    "расстояние между оптическими центрами",

    "вставка",
)


NAME_DEDUP_ATTR_KEYWORDS = (
    "бренд",
    "brand",
    "производитель",
    "manufacturer",

    "цвет",
    "color",
    "цвет товара",

    "объем памяти",
    "объём памяти",
    "память",
    "memory",
    "ram",
    "rom",

    "объем накопителя",
    "объём накопителя",
    "storage",

    "размер",
    "size",
)


# Регулярные выражения для защиты моделей, артикулов и технических кодов
# от случайного удаления при очистке названия.
MODEL_PROTECTION_PATTERNS = (
    r"\b[A-Z]{2,}\d{2,}[A-Z0-9-]*\b",
    r"\b\d+[A-Z]{2,}\d*\b",
)


# def normalize_for_match(text):
#     text = str(text).lower()
#
#     # унифицируем ё
#     text = text.replace("ё", "е")
#
#     # разные разделители считаем пробелами
#     text = re.sub(r"[-_/,+()\[\]]+", " ", text)
#
#     # схлопываем пробелы
#     text = re.sub(r"\s+", " ", text)
#
#     return text.strip()
#
#
# def should_remove_attr_from_name(key):
#     key_lower = normalize_for_match(key)
#
#     return any(
#         keyword in key_lower
#         for keyword in NAME_DEDUP_ATTR_KEYWORDS
#     )
#
#
# def remove_attribute_value_from_name(name, value):
#     """
#     Удаляет конкретное значение атрибута из name.
#
#     Сначала пытается удалить точное значение.
#     Потом — нормализованный вариант с гибкими пробелами/разделителями.
#     """
#     if name is None or value is None:
#         return name
#
#     name = str(name)
#     value = str(value).strip()
#
#     if not value:
#         return name
#
#     # Совсем короткие значения опасно удалять:
#     # "S", "M", "1", "5" и т.п.
#     if len(value) < 3:
#         return name
#
#     # 1. Точное совпадение без учёта регистра
#     pattern = re.compile(
#         re.escape(value),
#         flags=re.IGNORECASE,
#     )
#
#     new_name = pattern.sub(" ", name)
#
#     if new_name != name:
#         return re.sub(r"\s+", " ", new_name).strip()
#
#     # 2. Более гибкое совпадение.
#     # Например:
#     # "128 GB" ~ "128GB"
#     # "black-blue" ~ "black blue"
#     normalized_value = normalize_for_match(value)
#
#     if not normalized_value:
#         return name
#
#     value_parts = normalized_value.split()
#
#     # Строим regex, допускающий пробелы/дефисы между частями
#     flexible_pattern = r"[\s\-_/,]*".join(
#         re.escape(part)
#         for part in value_parts
#     )
#
#     pattern = re.compile(
#         flexible_pattern,
#         flags=re.IGNORECASE,
#     )
#
#     name = pattern.sub(" ", name)
#
#     # чистим оставшийся мусор
#     name = re.sub(r"\s+", " ", name)
#     name = re.sub(r"\s+([,;:])", r"\1", name)
#
#     return name.strip(" ,;-_")
#
#
# def deduplicate_name_with_attrs(name, attrs):
#     """
#     Удаляет из name значения выбранных атрибутов,
#     если эти значения уже представлены в attrs.
#     """
#     if not isinstance(attrs, dict):
#         return str(name)
#
#     result = str(name)
#
#     for key, value in attrs.items():
#
#         if not should_remove_attr_from_name(key):
#             continue
#
#         # Если value вдруг список
#         if isinstance(value, (list, tuple, set)):
#             values = value
#         else:
#             values = [value]
#
#         for v in values:
#             result = remove_attribute_value_from_name(
#                 result,
#                 v,
#             )
#
#     # финальная очистка
#     result = re.sub(r"\s+", " ", result).strip()
#
#     return result


# Token budget for EACH product before pair tokenization.
# 92 + 92 + ~3 special tokens = <= 188 for BERT/MiniLM pair input.
# Максимальное количество токенов для ОДНОГО товара перед созданием пары.
# Два товара вместе со специальными токенами должны помещаться в контекстное окно модели.
# Максимальное количество токенов для одного товара перед объединением двух товаров в пару.
# Ограничение помогает гарантировать, что пара помещается в контекстное окно модели.
ITEM_MAX_TOKENS = 92


# ============================================================
# ПАТТЕРНЫ ДЛЯ УДАЛЕНИЯ ЧУВСТВИТЕЛЬНЫХ / СЛУЖЕБНЫХ ДАННЫХ
# ============================================================

RE_URL = re.compile(
    r"""
    (?:
        https?://\S+
        |
        www\.\S+
        |
        \b[a-zA-Z0-9.-]+\.(?:ru|com|net|org|рф|io|shop|store|online)\b\S*
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

RE_PHONE = re.compile(
    r"""
    (?<!\w)
    (?:\+?7|8)?
    [\s\-()]*
    (?:\d[\s\-()]*){10}
    (?!\w)
    """,
    re.VERBOSE,
)

RE_TELEGRAM = re.compile(
    r"""
    (?:
        https?://t\.me/\S+
        |
        t\.me/\S+
        |
        telegram\.me/\S+
        |
        (?<!\w)@[A-Za-z0-9_]{5,}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Явные идентификаторы внутри текста:
# SKU: 12345
# ID ABC123
# Артикул: ZX-554-A
# Код товара 123456
RE_LABELED_ID = re.compile(
    r"""
    \b
    (?:
        артикул
        |
        sku
        |
        id
        |
        идентификатор
        |
        код\s+товара
        |
        код\s+продукта
        |
        product\s*id
        |
        item\s*id
        |
        vendor\s*code
    )
    \s*
    [:=№#\-]?
    \s*
    [A-Za-zА-Яа-яЁё0-9._/\-]+
    """,
    re.IGNORECASE | re.VERBOSE,
)

def sanitize_text(text):
    """
    Удаляет из текста:
    - URL;
    - email;
    - телефоны;
    - Telegram-контакты;
    - явно подписанные артикулы / SKU / ID / коды товара.

    После удаления нормализует пробелы.
    """

    if text is None:
        return ""

    text = str(text)

    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_PHONE.sub(" ", text)
    text = RE_TELEGRAM.sub(" ", text)
    text = RE_LABELED_ID.sub(" ", text)

    # Убираем мусорные пробелы
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_duplicate_attrs_from_name(
    name,
    attrs,
):
    """
    Удаляет из name значения некоторых attributes.

    Удаляем только если:
    - ключ атрибута разрешён;
    - значение достаточно длинное;
    - значение реально найдено в name;
    - значение не похоже на модель/id.
    """

    if not isinstance(attrs, dict):
        return name


    result = str(name)


    normalized_name = normalize_for_dedup(
        result
    )


    for key, value in attrs.items():

        key_norm = normalize_for_dedup(
            key
        )


        # Проверяем, можно ли удалять этот тип атрибута
        if not any(
            allowed in key_norm
            for allowed in NAME_DEDUP_ATTRS
        ):
            continue


        value = str(value).strip()


        # Слишком короткие значения опасны
        if len(value) < 3:
            continue


        # Не трогаем модели и коды
        if looks_like_identifier(value):
            continue


        value_norm = normalize_for_dedup(
            value
        )


        # Ищем в имени
        if value_norm not in normalized_name:
            continue


        # Удаляем только отдельное слово/фразу
        pattern = re.compile(
            r"(?<!\w)"
            + re.escape(value)
            + r"(?!\w)",
            flags=re.IGNORECASE,
        )


        result = pattern.sub(
            " ",
            result,
        )


        normalized_name = normalize_for_dedup(
            result
        )


    # чистка после удаления

    result = re.sub(
        r"\s+",
        " ",
        result,
    )

    result = re.sub(
        r"\s+([,.;:])",
        r"\1",
        result,
    )

    return result.strip(
        " ,.;:-"
    )

# Make a simple string for each product
def prepare_name(name, max_chars=NAME_MAX_CHARS):
    if name is None:
        return ""

    # Сначала удаляем ссылки / ID / контакты
    name = sanitize_text(name)

    if len(name) <= max_chars:
        return name

    cut = name[:max_chars]

    # По возможности не режем посреди слова
    last_space = cut.rfind(" ")

    if last_space > max_chars * 0.8:
        cut = cut[:last_space]

    return cut.rstrip()


EMPTY_VALUES = {
    "",
    "none",
    "null",
    "nan",
    "-",
    "--",
    "не указано",
    "нет",
}

CONTACT_ATTR_KEYWORDS = (
    "телефон",
    "номер телефона",
    "email",
    "e-mail",
    "почта",
    "контакт",
    "telegram",
    "whatsapp",
    "viber",
    "ссылка",
    "url",
    "website",
    "web site",
    "сайт",
)

# Ключевые слова атрибутов с самым высоким приоритетом.
# Обычно это уникальные идентификаторы: SKU, OEM, артикулы и подобные поля.
UNIQUE_ATTR_KEYWORDS = (
    "артикул",
    "артикул производителя",
    "код товара",
    "партномер",
    "партномер (артикул производителя)",
    "oem-номер",
    "альтернативные артикулы товара",
    "артикул сопутствующего товара",
    "код производителя",
    "штрих-код",
    "oem",
    "номер артикула",
    "код производителя картриджей для принтеров",
    "артикул lego",
    "артикул поставщика",
    "isbn",
    "допуски oem",
    "код изделия",
    "артикул sku",
    "спецификация oem",
    "артикул №",
    "код производителя ноутбуков",
    "парт номер",
    "спецификации oem",
    "sku",
    "код производителя чернил для принтера",
    "совместимые парт номера",
    "каталожный номер",
    "единая модель/артикул для мультикарточки",
    "код производителя мфу",
    "артикул производителя (part number)",
    "код производителя принтеров",
    "модификация (код производителя)",
    "код производителя flash-накопителей",
    "парт номер аккумулятора",
    "код производителя внешних жестких дисков",
    "сканер штрих-кодов",
    "oem допуски",
    "аон, caller id",
    "двери артикул",
    "код производителя (сиденье)",
    "код производителя адаптеров",
    "код производителя моноблоков",
    "подходит для рулевых реек с oem",
    "распознаваемые штрих-коды",
    "сканер штрих кодов",
    "артикул совместимого экрана для ванны",
    "датчик touch id",
    "код производителя (звукоизоляционный комплект)",
    "код производителя (комплект инсталляции)",
    "код производителя (панель смыва)",
    "код производителя переходников",
    "код производителя фотобарабанов",
    "модель/артикул производителя",
    "спецификация oem гидравлического масла",
    "топ артикул",
    "штрихкод",
)

# Ключевые слова важных характеристик товара.
# Эти атрибуты получают высокий приоритет после уникальных идентификаторов.
IMPORTANT_ATTR_KEYWORDS = (
    "модель",
    "бренд",
    "марка",
    "производитель",
    "версия",
    "размер",
    "объем",
    "объём",
    "вес",
    "цвет",
)

def filter_attrs(attrs):
    """
    Фильтрует исходный словарь атрибутов.

    Удаляет:
    - empty keys/values;
    - placeholder values like None/null;
    - contact/link attributes.

    Сохраняет:
    - product characteristics;
    - identifiers;
    - technical parameters.

    Вход:
        attrs: dictionary {attribute_name: value}

    Выход:
        cleaned dictionary.
    """
    if not isinstance(attrs, dict):
        return {}

    result = {}

    for key, value in attrs.items():
        key = str(key).strip()
        value = str(value).strip()

        if not key or not value:
            continue

        if value.lower() in EMPTY_VALUES:
            continue

        key_lower = key.lower()

        # Контакты и ссылки по-прежнему удаляем целиком
        if any(x in key_lower for x in CONTACT_ATTR_KEYWORDS):
            continue

        result[key] = value

    return result


def attr_priority(key, value, attr_freq=None):
    """
    Рассчитывает полезность одного атрибута.

    Более высокий балл получают:
    - уникальные идентификаторы;
    - важные характеристики товара;
    - короткие компактные значения.

    Более низкий балл получают:
    - очень длинные описания;
    - редкие шумовые атрибуты.

    Полученный балл далее используется в prioritize_attrs().
    """
    key_lower = key.lower()
    value = str(value)

    score = 0.0

    # 1. Уникальные идентификаторы — самый высокий приоритет
    if any(x in key_lower for x in UNIQUE_ATTR_KEYWORDS):
        score += 100.0

    # 2. Важные товарные характеристики
    elif any(x in key_lower for x in IMPORTANT_ATTR_KEYWORDS):
        score += 20.0

    # 3. Короткие значения обычно полезнее длинных описаний
    value_len = len(value)

    if value_len <= 30:
        score += 5.0

    elif value_len <= 100:
        score += 3.0

    elif value_len <= 300:
        score += 1.0

    else:
        score -= 3.0

    # 4. Частота ключа — слабый дополнительный сигнал
    if attr_freq is not None:
        freq = attr_freq.get(key, 0)

        if freq >= 1000:
            score += 1.0

        elif freq < 5:
            score -= 0.5

    return score


def prioritize_attrs(attrs, attr_freq=None):
    """
    Сортирует атрибуты по важности.

    Важно:
    Эта функция НЕ удаляет атрибуты.
    Она только меняет порядок, чтобы полезная
    информация получила доступный текстовый бюджет первой.
    """
    return dict(
        sorted(
            attrs.items(),
            key=lambda kv: attr_priority(
                kv[0],
                kv[1],
                attr_freq,
            ),
            reverse=True,
        )
    )


def truncate_attr_value(value, max_chars):
    """
    Обрезает значение атрибута до max_chars.
    По возможности не режет последнее слово посередине.
    """
    value = str(value).strip()

    if len(value) <= max_chars:
        return value

    cut = value[:max_chars]

    # Если рядом с концом есть пробел — режем по слову
    last_space = cut.rfind(" ")

    if last_space > max_chars * 0.8:
        cut = cut[:last_space]

    return cut.rstrip()


def truncate_attrs(
    attrs,
    total_budget=ATTR_TEXT_BUDGET_CHARS,
    max_single_share=MAX_SINGLE_ATTR_SHARE,
    max_value_chars=MAX_ATTR_VALUE_CHARS,
):
    """
    Ограничивает длину значений атрибутов.

    Одно значение не может быть длиннее:
        min(
            MAX_ATTR_VALUE_CHARS,
            total_budget * MAX_SINGLE_ATTR_SHARE
        )

    Также весь блок атрибутов не превышает total_budget символов.

    Порядок attrs сохраняется, поэтому более приоритетные атрибуты,
    которые уже подняты prioritize_attrs(), получают место первыми.
    """

    if not attrs:
        return {}

    # Например:
    # min(300, 1000 * 0.10) = 100 символов
    single_value_limit = min(
        max_value_chars,
        max(1, int(total_budget * max_single_share)),
    )

    result = {}
    used_chars = 0

    for key, value in attrs.items():
        key = str(key).strip()
        value = str(value).strip()

        if not key or not value:
            continue

        # Сначала режем слишком длинное значение
        value = truncate_attr_value(
            value,
            single_value_limit,
        )

        # Именно столько символов займёт атрибут:
        # "Ключ: значение "
        prefix = f"{key}: "
        attr_len = len(prefix) + len(value)

        remaining = total_budget - used_chars

        if remaining <= len(prefix):
            break

        # Если весь атрибут уже помещается
        if attr_len <= remaining:
            result[key] = value
            used_chars += attr_len + 1
            continue

        # Если значение частично помещается —
        # используем оставшееся место
        remaining_for_value = remaining - len(prefix)

        if remaining_for_value > 0:
            value = truncate_attr_value(
                value,
                remaining_for_value,
            )

            if value:
                result[key] = value

        # Бюджет закончился
        break

    return result


# def limit_attrs_total_length(
#     attrs,
#     max_chars=ATTR_TEXT_MAX_CHARS,
# ):
#     """
#     Ограничивает суммарный размер блока Attributes.
#
#     Атрибуты должны быть заранее отсортированы через prioritize_attrs().
#
#     Важные атрибуты идут первыми.
#     Если очередной атрибут целиком не помещается в оставшийся бюджет,
#     мы его пропускаем и пробуем следующий.
#
#     Значение атрибута здесь дополнительно НЕ режется.
#     """
#
#     if not attrs:
#         return {}
#
#     result = {}
#     used_chars = 0
#
#     for key, value in attrs.items():
#         key = str(key).strip()
#         value = str(value).strip()
#
#         if not key or not value:
#             continue
#
#         attr_text = f"{key}: {value}"
#
#         # +1 — пробел между атрибутами
#         required_chars = len(attr_text)
#
#         if result:
#             required_chars += 1
#
#         # Атрибут целиком помещается
#         if used_chars + required_chars <= max_chars:
#             result[key] = value
#             used_chars += required_chars
#
#         # Не помещается — НЕ режем его посередине,
#         # а пробуем следующий атрибут.
#         else:
#             continue
#
#     return result


def truncate_text_to_tokens(
    text,
    tokenizer,
    max_tokens,
):
    """
    Обрезает текст до заданного количества токенов.

    Назначение:
    - гарантировать, что текст товара помещается в ограничение модели;
    - сохранить начало текста, так как первые токены обычно содержат
      наиболее важную информацию (название, бренд, основные характеристики).

    Важно:
    - функция не изменяет смысл текста;
    - удаляет только часть текста, которая выходит за предел токенового бюджета.

    Параметры:
        text:
            Исходная строка текста товара.

        tokenizer:
            Токенизатор используемой модели.

        max_tokens:
            Максимальное количество разрешённых токенов.

    Возвращает:
        Текст, помещающийся в заданное количество токенов.
    """

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if len(ids) <= max_tokens:
        return text

    ids = ids[:max_tokens]

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

def prepare_pair_for_model(
    text1,
    text2,
    tokenizer,
    item_max_tokens=ITEM_MAX_TOKENS,
):
    """
    Подготавливает пару товаров перед передачей в модель.

    Назначение:
    - ограничить размер каждого товара отдельно;
    - не допустить переполнения общего контекстного окна модели;
    - сохранить баланс информации между двумя товарами пары.

    Логика:
    1. Каждый товар обрезается до ITEM_MAX_TOKENS.
    2. После этого пара передаётся в токенизатор модели.

    Это позволяет избежать ситуации, когда один очень длинный товар
    полностью вытесняет информацию второго товара.
    """

    text1 = truncate_text_to_tokens(
        text1,
        tokenizer,
        item_max_tokens,
    )

    text2 = truncate_text_to_tokens(
        text2,
        tokenizer,
        item_max_tokens,
    )

    return text1, text2


def _create_product_text(row):
    """
    Создаёт итоговое текстовое представление одного товара.

    Этапы обработки:
    1. Получение исходного названия товара.
    2. Очистка имени от лишнего шума.
    3. Фильтрация атрибутов.
    4. Ранжирование атрибутов по важности.
    5. Ограничение длины атрибутов.
    6. Формирование единой строки для модели.

    Главная цель:
    получить компактное представление товара, сохранив признаки,
    наиболее полезные для задачи matching.
    """
    category = str(row["category"])

    try:
        attrs = json.loads(row["attributes"])
    except Exception:
        attrs = {}

    attrs = filter_attrs(attrs)

    # Удаляем из name информацию,
    # которая уже явно представлена в выбранных attrs
    # name = remove_duplicate_attrs_from_name(
    #     row["name"],
    #     attrs,
    # )

    # И только потом оставляем старый safety-limit,
    # если он всё ещё нужен
    # name = prepare_name(row['name'])

    # В процессе тестирования было выявлено, что название товара лучше оставить прежним
    name = row['name']

    attrs = prioritize_attrs(
        attrs,
        ATTR_FREQ,
    )

    attrs = truncate_attrs(attrs)

    attr_text = " ".join(
        f"{key}: {value}"
        for key, value in attrs.items()
    )

    return (
        f"Name: {name} "
        f"Category: {category} "
        f"Attributes: {attr_text}"
    )


def analyze_pair_token_lengths(
    pairs,
    tokenizer,
    max_seq_length=188,
    sample_size=None,
):
    """
    Анализирует распределение длины текстовых пар в токенах.

    Используется для EDA и настройки обрезания.

    Показывает:
    - среднюю длину пары;
    - перцентили длины;
    - долю пар, превышающих лимит модели;
    - количество токенов, которое будет потеряно после truncation.

    Помогает определить:
    - достаточно ли текущего token budget;
    - нужно ли менять стратегию очистки текста.
    """

    """
    Анализирует длины пар ДО tokenizer truncation.

    pairs:
        List[Tuple[str, str]]

    Возвращает словарь со статистиками.
    """

    if sample_size is not None and len(pairs) > sample_size:
        pairs = pairs[:sample_size]

    len1_list = []
    len2_list = []
    pair_len_list = []

    for text1, text2 in tqdm(pairs, desc="Analyzing token lengths"):
        # отдельно каждый товар
        ids1 = tokenizer(
            text1,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        ids2 = tokenizer(
            text2,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        len1 = len(ids1)
        len2 = len(ids2)

        # Важно:
        # это именно фактическая токенизация пары,
        # включая специальные токены модели.
        pair_ids = tokenizer(
            text1,
            text_pair=text2,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        pair_len = len(pair_ids)

        len1_list.append(len1)
        len2_list.append(len2)
        pair_len_list.append(pair_len)

    len1_arr = np.asarray(len1_list)
    len2_arr = np.asarray(len2_list)
    pair_arr = np.asarray(pair_len_list)

    over_mask = pair_arr > max_seq_length
    under_mask = pair_arr < max_seq_length

    stats = {
        "n_pairs": len(pair_arr),

        "item1_mean": float(len1_arr.mean()),
        "item1_p50": float(np.percentile(len1_arr, 50)),
        "item1_p90": float(np.percentile(len1_arr, 90)),
        "item1_p95": float(np.percentile(len1_arr, 95)),
        "item1_p99": float(np.percentile(len1_arr, 99)),

        "item2_mean": float(len2_arr.mean()),
        "item2_p50": float(np.percentile(len2_arr, 50)),
        "item2_p90": float(np.percentile(len2_arr, 90)),
        "item2_p95": float(np.percentile(len2_arr, 95)),
        "item2_p99": float(np.percentile(len2_arr, 99)),

        "pair_mean": float(pair_arr.mean()),
        "pair_p50": float(np.percentile(pair_arr, 50)),
        "pair_p90": float(np.percentile(pair_arr, 90)),
        "pair_p95": float(np.percentile(pair_arr, 95)),
        "pair_p99": float(np.percentile(pair_arr, 99)),

        "share_over_188": float(over_mask.mean()),
        "share_under_188": float(under_mask.mean()),

        "mean_overflow_tokens": (
            float((pair_arr[over_mask] - max_seq_length).mean())
            if over_mask.any()
            else 0.0
        ),

        "median_overflow_tokens": (
            float(np.median(pair_arr[over_mask] - max_seq_length))
            if over_mask.any()
            else 0.0
        ),
    }

    return stats


def _load_and_process_data(data_path):
    """
    Загружает исходные данные товаров и выполняет предварительную обработку.

    Этапы:
    - чтение файла;
    - приведение колонок к ожидаемому формату;
    - создание подготовленных структур данных,
      используемых дальнейшим пайплайном.

    Возвращает:
        Обработанный DataFrame с товарами.
    """

    df = pd.read_parquet(data_path)
    df['text'] = df.apply(_create_product_text, axis=1)
    return df

def _build_pairs(match_df, data_df):
    """
    Создаёт пары товаров для задачи matching.

    Назначение:
    - сопоставить идентификаторы товаров;
    - собрать текст первого и второго товара;
    - подготовить входные данные для модели.

    Возвращает:
    - пары текстов товаров;
    - идентификаторы товаров;
    - дополнительную информацию для дальнейшего сохранения результатов.
    """

    id_to_text = dict(zip(data_df['id'], data_df['text']))
    pairs, labels, pair_ids = [], [], []
    has_target = 'target' in match_df.columns

    for _, row in match_df.iterrows():
        id1, id2 = row['id1'], row['id2']
        text1, text2 = id_to_text.get(id1, ""), id_to_text.get(id2, "")
        if not text1 or not text2:
            continue
        pairs.append((text1, text2))
        pair_ids.append((id1, id2))
        if has_target:
            labels.append(row['target'])

    return pairs, pair_ids, (np.array(labels) if has_target else None)

