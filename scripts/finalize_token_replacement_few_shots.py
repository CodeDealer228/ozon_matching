"""Apply the reviewed decisions to the fixed 60-row few-shot template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_token_replacement_counterfactuals import (
    replace_nested,
    replace_text,
    validate_decision,
)


EXPECTED_PREVIEW_SHA256 = "9da421697486b7cc60dea90596f2f5cf34659cdb54216451365e8cff70784be5"


def skip(reason: str) -> dict[str, object]:
    return {"status": "skip", "replacements": [], "skip_reason": reason}


def replace(source: str, target: str) -> dict[str, object]:
    return {
        "status": "replace",
        "replacements": [{"source_token": source, "replacement_token": target}],
        "skip_reason": "",
    }


# Ordered exactly like the reviewed template: three examples per category.
REVIEWED_DECISIONS: dict[str, list[dict[str, object]]] = {
    "Автотовары": [
        skip("Карточки относятся к разным моделям автомобилей; замена диаметра не исправит исходное расхождение."),
        skip("HB2151 — конкретный артикул помпы; безопасный реальный артикул-замена не подтвержден."),
        skip("Исходные карточки указывают разные модели Kia и разное количество накидок."),
    ],
    "Аптека": [
        skip("В исходных карточках уже различаются ароматы, поэтому пара ненадёжна для контрфактуальной замены."),
        skip("Радиус кривизны привязан к линейке линз; другой поддерживаемый радиус не подтвержден."),
        skip("Ширина 22 см связана с конкретной моделью корсета ПР-311; произвольная замена создаст противоречие."),
    ],
    "Бытовая техника": [
        skip("Совместимость и партномера суппортов различаются; замена одного бренда сделает список несовместимым."),
        skip("Карточки указывают разные модели вентиляторов CFS 160S и CFS 250S."),
        skip("Исходная пара объединяет чайник и кофемолку, поэтому не подходит для positive few-shot."),
    ],
    "Бытовая химия": [
        replace("250", "300"),
        replace("aos", "fairy"),
        skip("Единственный общий токен — служебное количество упаковок 1; содержательной замены нет."),
    ],
    "Галантерея и аксессуары": [
        skip("Модель LQ-139L-4B1 принадлежит Casio; замена только бренда создаст невозможное сочетание."),
        skip("Исходные карточки описывают разные брелоки, а общий токен 1 не задаёт товар."),
        replace("27х14х38", "30х15х40"),
    ],
    "Детские товары": [
        replace("105", "120"),
        skip("Количество 15 связано с конкретным набором «Мягкие кирпичики mini»; подтвержденной вариации нет."),
        skip("Noname — отсутствие бренда, а не содержательное значение для замены."),
    ],
    "Дом и сад": [
        skip("XXL входит в название конкретной копилки и не является независимо заменяемым размером."),
        skip("Размер 30х40 зашит в артикул К-316-3040; замена только размера создаст противоречие."),
        replace("21", "31"),
    ],
    "Канцелярские товары": [
        replace("3", "5"),
        replace("а5", "а4"),
        skip("Team является частью устойчивого названия Putin Team; отдельная замена разрушит название."),
    ],
    "Красота и гигиена": [
        replace("500", "1000"),
        replace("13", "21"),
        replace("15", "20"),
    ],
    "Мебель": [
        replace("1000", "2000"),
        replace("4", "2"),
        skip("Размер 23 см указан только у одной конкретной вешалки; второй item не даёт надёжного варианта."),
    ],
    "Музыкальные инструменты": [
        skip("Карточки уже различаются по натяжению и модели 2001/2001L; бренд La Bella нельзя менять отдельно."),
        skip("Модель 16027 Nanoweb принадлежит Elixir; замена только бренда создаст невозможный товар."),
        skip("MG16XU и MG16 — разные модели, а число 16 является частью их обозначения."),
    ],
    "Обувь": [
        skip("Ralf Ringer — составной бренд; отдельная замена одного слова разрушит название."),
        skip("Артикул 5-515004-2342 связан с Pixel; замена только бренда создаст противоречие."),
        replace("puma", "adidas"),
    ],
    "Одежда": [
        skip("Исходные карточки расходятся по размеру и назначению; общие числа относятся к составу и росту."),
        skip("Карточки расходятся по размеру 46/48 и росту модели; предложенные числа небезопасны."),
        replace("52", "54"),
    ],
    "Продукты питания": [
        replace("300", "250"),
        replace("1000", "800"),
        skip("Extra Cream — линейка Kimbo; замена только бренда на Italco создаст невозможное сочетание."),
    ],
    "Спорт и отдых": [
        skip("HS-3-003 — конкретный артикул приманки; другой реальный артикул не подтвержден."),
        replace("600d", "420d"),
        replace("11", "20"),
    ],
    "Строительство и ремонт": [
        skip("LSP-0504 и LGO связаны с Lussole; замена только бренда создаст неверную модель."),
        skip("Provence «снежная мята» — конкретная линейка Siana; менять только бренд нельзя."),
        replace("ip20", "ip44"),
    ],
    "Товары для животных": [
        replace("60", "100"),
        replace("170", "200"),
        skip("Габариты связаны с конкретным артикулом клетки 7861042; произвольная замена размера противоречива."),
    ],
    "Хобби и творчество": [
        replace("16", "24"),
        skip("Размер 19,5×16,9 связан с конкретным артикулом термотрансфера 7517374."),
        replace("40", "100"),
    ],
    "Электроника": [
        skip("Fondina Bicolore — цельное название серии MyPads; замена одного слова разрушит модель."),
        replace("dht11", "dht22"),
        skip("Замена английского Blue оставит русские поля цвета «синий» и создаст противоречие."),
    ],
    "Ювелирные изделия": [
        skip("Исходные кольца различаются пробой, размером и моделью; общий бренд Sokolov недостаточен."),
        skip("Карточки различаются размером, вставкой и покрытием; общие бренд, проба и вес небезопасны."),
        skip("Проба 925 связана с серебром, 777 — с названием модели; безопасной независимой замены нет."),
    ],
}


def validate_materialization(row: dict[str, object], decision: dict[str, object]) -> None:
    for operation in decision["replacements"]:
        source = str(operation["source_token"])
        target = str(operation["replacement_token"])
        for side in ("item_1", "item_2"):
            item = row["input"][side]
            _, name_changes = replace_text(str(item["name"]), source, target)
            _, attribute_changes = replace_nested(item["attributes"], source, target)
            if name_changes + attribute_changes == 0:
                raise ValueError(f"replacement_not_applied:{row['input']['category']}:{side}:{source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("artifacts/token_replacement_counterfactual/few_shot_annotation_template_v1.json"),
    )
    args = parser.parse_args()
    value = json.loads(args.path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unexpected_schema_version")
    if value.get("source_preview_sha256") != EXPECTED_PREVIEW_SHA256:
        raise ValueError("unexpected_source_preview_identity")
    categories = value.get("categories")
    if not isinstance(categories, dict) or set(categories) != set(REVIEWED_DECISIONS):
        raise ValueError("unexpected_categories")

    replace_count = 0
    skip_count = 0
    for category, decisions in REVIEWED_DECISIONS.items():
        rows = categories[category]
        if not isinstance(rows, list) or len(rows) != len(decisions):
            raise ValueError(f"unexpected_category_size:{category}")
        for row, decision in zip(rows, decisions, strict=True):
            tokens = row["input"]["offered_tokens"]
            validated = validate_decision(decision, tokens, max_replacements=2)
            validate_materialization(row, validated)
            row["decision"] = validated
            if validated["status"] == "replace":
                replace_count += 1
            else:
                skip_count += 1

    value["status"] = "human_annotated_reviewed"
    args.path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": value["status"],
        "categories": len(categories),
        "examples": replace_count + skip_count,
        "replace": replace_count,
        "skip": skip_count,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
