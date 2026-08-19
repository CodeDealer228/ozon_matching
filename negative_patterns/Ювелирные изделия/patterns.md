## Pattern 1

**Pattern:** A ring and a pendant are different jewellery product types even when the metal and seller are the same.

**Concrete difference:** The first item is a size-17 silver ring, whereas the second is a silver pendant with natural prasiolite.

**Why this is a strict negative signal:** The items have mutually exclusive wearing forms (finger ring versus hanging pendant), so the shared seller and 925 silver do not identify one product.

**Evidence pair id:** pair_0989

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0989", "source_row_index": 63193, "fold": 2, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 146028976015, "name": "milana silver кольцо серебро 925 пробы", "attributes": {"целевая аудитория": "взрослая", "артикул": "17.0", "бренд": "milana silver", "коллекция": "базовая коллекция", "о драгоценном металле": "родированное серебро - покрытие тонким слоем родия (металл платиновой группы) предохраняет серебряные изделия от почернения и позволяет сохранить блеск. внешний вид изделий с родированным покрытием отличается более блестящей и белой поверхностью.", "проба": "925", "комплектация": "ювелирное изделие, фирменная упаковка, пломба, бирка, памятка по уходу, подарочный минерал", "размер изделия": "17", "пол": "женский", "тип": "кольцо ювелирное", "гарантийный срок": "7 дней", "вес товара, г": "4.1", "средний вес изделия, гр": "4.1", "состав ювелирного изделия": "серебро"}, "formatted_text_length": 740}, "item2": {"id": 283467987210, "name": "milana silver / кулон из серебра с празиолитом", "attributes": {"состав ювелирного изделия": "серебро", "проба": "925", "вставка": "празиолит натуральный", "минимальный вес (г)": "3.6 г", "комплектация": "ювелирное изделие, фирменная упаковка, пломба, бирка, памятка по уходу, подарочный минерал", "страна производства": "таиланд", "валюта": "rub", "цвет": "зеленый"}, "formatted_text_length": 354}}
```

## Pattern 2

**Pattern:** A bracelet explicitly intended for the wrist is not the same product as a neck chain, even if both are 925 silver chain jewellery from the same maker.

**Concrete difference:** Item 1 is a 21-cm Singapore-weave bracelet for the wrist, while item 2 is a Nona-weave chain described for the neck.

**Why this is a strict negative signal:** The listed product type and intended body location differ, and the weave also differs; these are product-defining rather than packaging differences.

**Evidence pair id:** pair_0983

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0983", "source_row_index": 191083, "fold": 2, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 446676665852, "name": "ювелирный завод титан браслет серебро позолоченное 925 пробы, плетение сингапур", "attributes": {"целевая аудитория": "взрослая", "артикул": "бя330ва8пгз-с881_21", "бренд": "ювелирный завод титан", "вид плетения цепочки": "сингапур", "коллекция": "базовая коллекция", "цвет товара": "золотой", "страна-изготовитель": "россия", "покрытие": "золото", "о драгоценном металле": "позолоченное серебро. декоративное покрытие тонким слоем золота позволяет серебряным украшениям быть более стойкими к факторам окружающей среды, не окисляться и не чернеть.  золото, предназначенное для нанесения позолоты, не имеет примесей.", "вставка": "без вставки", "вставки дополнительные": "без вставки", "длина украшения, см": "21", "вид замка": "шпрингельный", "проба": "925", "модель браслета": "цепочка", "комплектация": "браслет; пломба; бирка; подарочная упаковка", "упаковка": "подарочная упаковка", "размер изделия": "21", "пол": "женский", "тематика украшения": "классическая", "тип": "браслет ювелирный", "вид браслета": "на запястье", "вес товара, г": "1.39", "средний вес изделия, гр": "1.39", "состав ювелирного изделия": "серебро позолоченное"}, "formatted_text_length": 1091}, "item2": {"id": 472446462060, "name": "ювелирный завод титан / серебряная цепочка 925 пробы на шею женская", "attributes": {"состав ювелирного изделия": "серебро", "проба": "925", "вид плетения цепочки": "нонна", "вид замка": "шпрингельный", "вставка": "без вставки", "огранка": "алмазная", "покрытие": "золото", "комплектация": "цепочка; пломба; бирка; подарочная упаковка", "страна производства": "россия", "валюта": "rub", "цвет": "золотистый"}, "formatted_text_length": 389}}
```

## Pattern 3

**Pattern:** A silver-plated brass religious icon is a negative match for a 925 sterling-silver pendant icon.

**Concrete difference:** Item 1 declares brass with silvering and blackening, while item 2 declares 925 silver and is sold as a jewellery pendant.

**Why this is a strict negative signal:** The base material is explicitly different (brass versus silver), not merely a different surface colour, and the product classes also differ.

**Evidence pair id:** pair_0985

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0985", "source_row_index": 291415, "fold": 3, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 678604983162, "name": "atlanta иконка atlanta, серебристый", "attributes": {"цвет товара": "серебристый/черный", "тип": "иконка", "коллекция": "православие и мир", "пол": "для девочек, для мальчиков, женский, мужской", "материал": "латунь", "отделка": "серебрение, чернение", "вставка": "без вставок", "комплектация": "нательная иконка - 1 штука"}, "formatted_text_length": 311}, "item2": {"id": 592705554265, "name": "подвеска - иконка \"владимирская икона божией матери\"", "attributes": {"целевая аудитория": "взрослая", "артикул": "5306656в", "бренд": "aurum.ural", "коллекция": "весна-лето 2023", "цвет товара": "серебристый", "страна-изготовитель": "россия", "о драгоценном металле": "родированное серебро - покрытие тонким слоем родия (металл платиновой группы) предохраняет серебряные изделия от почернения и позволяет сохранить блеск. внешний вид изделий с родированным покрытием отличается более блестящей и белой поверхностью.", "дизайн": "подарок", "детские украшения": "да", "проба": "925", "комплектация": "подвеска в фирменной упаковке", "упаковка": "картонная коробка", "пол": "женский", "тематика украшения": "декоративная", "тип": "подвеска ювелирная", "гарантийный срок": "6 месяцев", "вес товара, г": "3.40", "средний вес изделия, гр": "3.40", "состав ювелирного изделия": "серебро"}, "formatted_text_length": 847}}
```

## Pattern 4

**Pattern:** Otherwise similar plain wedding rings are negative when one is 585 white gold and the other is 925 silver.

**Concrete difference:** Item 1 is a 585 white-gold Yuveros wedding ring, whereas item 2 is a 925 silver Azade ring.

**Why this is a strict negative signal:** The declared precious-metal material and fineness identify distinct product specifications; a silvery appearance does not make gold and silver interchangeable.

**Evidence pair id:** pair_0967

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0967", "source_row_index": 211959, "fold": 3, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 498216217177, "name": "кольцо из белого золота р. 15,5 юверос 123000-1-б", "attributes": {"модель": "123000-1-б", "артикул производителя": "123000-1-б", "бренд": "юверос", "тип": "кольцо", "вид": "обручальное", "пол": "унисекс", "материал изделия": "белое золото", "проба": "585", "вставка": "без вставки", "количество карат": "0", "размер": "15,5", "средний вес изделия, г": "1,44", "цвет материала": "серебристый", "цвет вставки": "без вставки", "форма вставки": "без вставки", "покрытие": "родирование", "особенности дизайна": "без особенностей", "изображение/надпись": "нет", "страна-производитель": "россия", "код товара": "100045195365"}, "formatted_text_length": 583}, "item2": {"id": 446676676479, "name": "azade кольцо серебряное обручальное мужское", "attributes": {"целевая аудитория": "взрослая", "артикул": "кл072с_19.0", "бренд": "azade", "коллекция": "весна-лето 2022", "цвет товара": "белый", "страна-изготовитель": "россия", "о драгоценном металле": "оксидированное серебро - серебро, покрытое тонкой оксидной пленкой. эта пленка наносится на поверхность серебра с целью защитить металл от потускнения, улучшить его декоративные свойства и сделать его стойким к коррозии.", "количество камней": "без камней", "модель кольца": "классическое", "детские украшения": "нет", "проба": "925", "размер изделия": "19", "пол": "мужской", "тематика": "свадебная", "тип": "кольцо", "средний вес изделия, гр": "2.92", "основной материал украшения": "серебро"}, "formatted_text_length": 717}}
```

## Pattern 5

**Pattern:** Earrings that differ jointly in gold colour and disclosed diamond setting are a strict negative configuration.

**Concrete difference:** Item 1 is white-gold earrings with ten round diamonds totaling 0.47 carat, while item 2 is red-gold earrings with no stone specification.

**Why this is a strict negative signal:** These are independent product-defining specifications within the same form factor, so the shared brand and 585 fineness are insufficient for a match.

**Evidence pair id:** pair_0976

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0976", "source_row_index": 10693, "fold": 3, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 17180003234, "name": "master brilliant серьги master brilliant, белое золото, 585 проба, бриллиант", "attributes": {"тип": "серьги", "пол": "унисекс", "вид металла": "золото", "тип золота": "белое", "проба": "585", "вставка": "бриллиант", "вставка подробнее": "бриллиант 10 шт. 0.47 карат,", "форма вставки": "круг", "средний вес изделия": "4.07 г", "тип замка": "английский", "комплектация": "ювелирное украшение, фирменный пакет магазина, фирменный подарочный футляр."}, "formatted_text_length": 430}, "item2": {"id": 309237659037, "name": "master brilliant серьги ювелирные золото красное 585 пробы", "attributes": {"артикул": "3941034", "бренд": "master brilliant", "коллекция": "базовая коллекция", "проба": "585", "тип": "серьги ювелирные", "средний вес изделия, гр": "4", "состав ювелирного изделия": "золото красное"}, "formatted_text_length": 271}}
```

## Pattern 6

**Pattern:** Two same-brand chains with different named weave, length, SKU, and target wearer are negative matches.

**Concrete difference:** The Sokolov chain is SKU 968040402, 65 cm, female, and double rhomb weave; the Diamant chain is SKU 96-180-02040-1, 40 cm, male, and curb weave.

**Why this is a strict negative signal:** The independent catalogue identifier together with incompatible construction and length distinguishes the chain products, despite both being 925 silver chains.

**Evidence pair id:** pair_0978

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0978", "source_row_index": 278742, "fold": 1, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 652835088451, "name": "цепочка из серебра 65 см sokolov 968040402", "attributes": {"бренд": "sokolov", "модель": "968040402", "артикул производителя": "968040402_65", "тип": "цепочка", "пол": "женский", "материал изделия": "серебро", "проба": "925", "вид плетения": "ромб", "размер, см": "65", "регулируемая длина": "нет", "средний вес изделия, г": "4,55", "цвет материала": "серебристый", "вид замка": "карабин", "покрытие": "родирование", "особенности дизайна": "двойное плетение", "страна-производитель": "россия", "код товара": "100029612945"}, "formatted_text_length": 493}, "item2": {"id": 601295549755, "name": "diamant, цепь из серебра, 96-180-02040-1, панцирное, 925 проба", "attributes": {"целевая аудитория": "взрослая", "артикул": "96-180-02040-1_40", "бренд": "diamant", "вид плетения цепочки": "панцирное", "коллекция": "базовая коллекция", "страна-изготовитель": "россия", "детские украшения": "нет", "проба": "925", "комплектация": "цепь, бирка, пломба, подарочная упаковка", "упаковка": "подарочная упаковка", "размер изделия": "40", "пол": "мужской", "тип": "цепочка", "гарантийный срок": "гарантийный срок изделий составляет 6 (шесть) календарных месяцев для ювелирных изделий, 3 (три) года для часов из драгоценных металлов, 12 (двенадцать) месяцев для часов из недрагоценных металлов, и исчисляется со дня передачи изделия потребителю", "средний вес изделия, гр": "2.28", "основной материал украшения": "серебро"}, "formatted_text_length": 786}}
```

## Pattern 7

**Pattern:** For same-brand 585 necklaces, a yellow-gold prasiolite design is a negative match for a white-gold diamond design.

**Concrete difference:** Item 1 is yellow gold with a 0.8-carat prasiolite, whereas item 2 is white gold with twenty-two diamonds.

**Why this is a strict negative signal:** The metal colour and the complete primary-stone configuration disagree simultaneously within an otherwise similar jewellery type.

**Evidence pair id:** pair_0980

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0980", "source_row_index": 305640, "fold": 3, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 712964666705, "name": "diamant, колье из желтого золота с празиолитом, 53-370-01774-2, декоративные, 585 проба", "attributes": {"целевая аудитория": "взрослая", "артикул": "53-370-01774-2_38\\40\\42", "бренд": "diamant", "коллекция": "базовая коллекция", "страна-изготовитель": "россия", "о драгоценном металле": "желтое золото - классика золотых украшений. сплав золота с медью. в зависимости от пробы, чем больше содержание золота в сплаве, тем ярче его цвет. серебро немного отбеливает природный истинный цвет золота, но все равно – мы любим желтое золото за его солнечный теплый цвет. украшения из желтого золота всегда в тренде, но особенно эффектно они выглядят на смуглой коже.", "вставка": "празиолит", "вес основных вставок, карат": "0.8", "о главном камне украшения": "празиолит – уже очень редкий в природе минерал. это весьма твердые, имеющие мягкую зеленую цветовую гамму камни. натуральный празиолит выцветает на солнце и в итоге может превратиться в горный хрусталь. ", "детские украшения": "нет", "проба": "585", "комплектация": "колье, бирка, пломба, подарочная упаковка", "упаковка": "подарочная упаковка", "размер изделия": "38-42", "пол": "женский", "тип": "колье ювелирное", "гарантийный срок": "гарантийный срок изделий составляет 6 (шесть) календарных месяцев для ювелирных изделий, 3 (три) года для часов из драгоценных металлов, 12 (двенадцать) месяцев для часов из недрагоценных металлов, и исчисляется со дня передачи изделия потребителю", "средний вес изделия, гр": "1.51", "состав ювелирного изделия": "золото желтое"}, "formatted_text_length": 1489}, "item2": {"id": 730144502205, "name": "diamant колье diamant, белое золото, 585 проба, бриллиант, длина 42 см.", "attributes": {"размер": "42 см", "пол": "женский", "вид металла": "золото", "тип золота": "белое", "проба": "585", "средний вес изделия": "1.17 г", "вставка": "бриллиант", "вставка подробнее": "21 бр круг 17 0,90*1,00 0,078 2/3 а, 1 бр круг 57 0,021 3/6 а", "форма вставки": "круг"}, "formatted_text_length": 342}}
```

## Pattern 8

**Pattern:** Distinct explicit article codes are a strict negative signal for otherwise similar same-brand earrings when their stone-set configuration also conflicts.

**Concrete difference:** Diamant article 51-120-02089-1 has a 0.408-carat fianit setting, while article 51-120-01685-1 specifies six 1.25-mm round fianits.

**Why this is a strict negative signal:** Both records give incompatible catalogue articles and different recorded stone configurations; the shared brand, metal, fineness, and lock do not override those identifiers.

**Evidence pair id:** pair_0964

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0964", "source_row_index": 211845, "fold": 1, "category": "Ювелирные изделия", "target": 0.0, "item1": {"id": 498216212198, "name": "diamant, серьги из золота с фианитами, 51-120-02089-1, декоративные, украшение из красного золота 585 пробы", "attributes": {"целевая аудитория": "взрослая", "артикул": "51-120-02089-1", "бренд": "diamant", "коллекция": "базовая коллекция", "страна-изготовитель": "россия", "размеры, мм": "3х18х1", "вставка": "фианит", "вес основных вставок, карат": "0.408", "детские украшения": "нет", "длина украшения, см": "1.8", "вид замка": "английский", "проба": "585", "комплектация": "серьги, бирка, пломба, подарочная упаковка", "упаковка": "подарочная упаковка", "пол": "женский", "тип": "серьги ювелирные", "гарантийный срок": "гарантийный срок изделий составляет 6 (шесть) календарных месяцев для ювелирных изделий, 3 (три) года для часов из драгоценных металлов, 12 (двенадцать) месяцев для часов из недрагоценных металлов, и исчисляется со дня передачи изделия потребителю", "средний вес изделия, гр": "1.74", "состав ювелирного изделия": "золото красное"}, "formatted_text_length": 920}, "item2": {"id": 180388694744, "name": "diamant серьги diamant, красное золото, 585 проба, фианит", "attributes": {"артикул": "51-120-01685-1", "тип": "серьги", "пол": "женский", "вид металла": "золото", "тип золота": "красное", "проба": "585", "вставка": "фианит", "вставка подробнее": "6 фианит недраг круг 1,25", "форма вставки": "круг", "средний вес изделия": "1.85 г", "тип замка": "английский"}, "formatted_text_length": 341}}
```

## Annotation caveats

- `pair_0957` is positive although it contrasts a 585 red-gold pendant with fianit against a 375 red-gold heart pendant with no listed stone. Therefore fineness or stone presence alone is not used as a negative rule.
- `pair_0960` and `pair_0993` are positive despite explicit sapphire-versus-iolite ring descriptions; gemstone name alone is not a safe category-wide negative signal.
- `pair_0965`, `pair_0972`, and `pair_0990` are positive despite different stated sizes, and `pair_0990` also has different brands. Ring size and brand alone are not treated as strict negatives.
- Several negative labels are suspiciously near duplicates or variants: `pair_0966` (two Talant crosses differing mainly in stated weight), `pair_0975` (near-identical Sokolov topaz/fiанит descriptions), `pair_0982` (same Sokolov model 94013648 at different sizes), and `pair_0998` (same brand/material/stone/size but a large weight discrepancy). These were not generalized into rules.
