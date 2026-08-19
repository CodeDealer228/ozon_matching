## Pattern 1

**Pattern:** Beads of the same brand and nominal size are negative when their bead construction is explicitly different (round versus Delica cylindrical).

**Concrete difference:** The first listing is Miyuki Round 11/0, while the second is Miyuki Delica cylindrical 11/0.

**Why this is a strict negative signal:** The shared brand, colour family, material, and size make the named bead form the decisive variant identifier.

**Evidence pair id:** pair_0850

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0850", "source_row_index": 89093, "fold": 1, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 206158501329, "name": "miyuki / бисер японский miyuki round 11/0 (#4214) зеленый", "attributes": {"цвет": "зеленый, зеленая мята", "материал изделия": "стекло", "размер бисера": "11/0", "возрастные ограничения": "7+", "покрытие бисера": "гальванизированное", "форма бисера": "круглый", "форма отверстия": "круглое", "вид бисера": "круглый; цветной; японский", "страна производства": "япония", "комплектация": "бисер 10 г; зип пакетик", "валюта": "rub"}, "formatted_text_length": 410}, "item2": {"id": 266288065352, "name": "бисер японский miyuki delica цилиндр 11/0 db-2188 мята полуматовый, серебряная линия внутри, duracoat, 5 грамм", "attributes": {"бренд": "miyuki", "цвет товара": "зеленый", "материал": "стекло", "тип": "бисер"}, "formatted_text_length": 205}}
```

## Pattern 2

**Pattern:** Beads with different explicit colour/article identifiers and finishes are negative even when brand, origin, material, size, and pack weight agree.

**Concrete difference:** The Preciosa 10/0 beads are glossy colour 53250 in one item and transparent with silver coating, colour 57060, in the other.

**Why this is a strict negative signal:** These identifiers select the actual bead variant, rather than describing a general compatible bead category.

**Evidence pair id:** pair_0851

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0851", "source_row_index": 186872, "fold": 1, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 438086709007, "name": "preciosa / бисер чешский для вышивки и плетения 10/0, 50г 53250", "attributes": {"цвет": "галопогосский зеленый", "материал изделия": "стекло", "размер бисера": "10/0", "покрытие бисера": "глянцевый", "форма бисера": "круглый", "форма отверстия": "круглое", "вид бисера": "круглый; вышивальный; стеклянный", "страна производства": "чехия", "комплектация": "бисер 50 г - 1 шт", "валюта": "rub"}, "formatted_text_length": 376}, "item2": {"id": 721554547149, "name": "бисер для плетения чешский preciosa 10/0 прозрачный с серебр. покрытием упаковка 50 гр", "attributes": {"размер бисера": "10/0 (2.2 - 2.4 мм)", "бренд": "preciosa", "цвет товара": "зеленый", "название цвета": "зеленый 57060", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-2/6735811610.jpg", "страна-изготовитель": "чехия", "материал": "стекло", "тип": "бисер", "вес товара, г": "50"}, "formatted_text_length": 376}}
```

## Pattern 3

**Pattern:** A multi-set plasticine offer is negative when the number of complete sets differs explicitly.

**Concrete difference:** The first offer contains five 16-colour, 320 g plasticine sets, whereas the second offers two such sets.

**Why this is a strict negative signal:** The quantities refer to complete included sets, so they materially change what is purchased despite the same base plasticine kit.

**Evidence pair id:** pair_0854

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0854", "source_row_index": 81375, "fold": 2, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 188978617630, "name": "комплект 5 наб, пластилин гамма мультики 16цв 320гр со стеком к/к", "attributes": {"тип": "пластилин", "в комплекте": "[пластилин гамма мультики 16цв 320гр со стеком к/к] x 5 шт."}, "formatted_text_length": 179}, "item2": {"id": 188978648013, "name": "пластилин гамма мультики 16 цветов 320 грамм, со стеком к/к, 2 набора", "attributes": {"возраст ребенка": "от 3 лет", "возраст учащегося": "без ограничения", "артикул": "k-70293-2-part71kms43", "бренд": "гамма", "цвет товара": "разноцветный", "страна-изготовитель": "россия", "материал": "полимерный материал", "минимальный возраст ребенка": "от 3 лет", "количество в упаковке, шт": "16", "число цветов, шт": "16", "цветовая гамма": "классическая", "вид пластилина": "классический", "пол ребенка": "унисекс", "тип": "пластилин", "вес товара, г": "320"}, "formatted_text_length": 525}}
```

## Pattern 4

**Pattern:** Named aircraft variants are negative when the aircraft designation differs, especially together with incompatible build specifications.

**Concrete difference:** One kit is an Airbus A320neo at 1/144 with 149 elements and 25 cm finished length; the other is named Airbus A321neo and lists a 1/35, 270-element model.

**Why this is a strict negative signal:** The aircraft model designation identifies a different subject, and the scale and element count reinforce that it is not the same kit.

**Evidence pair id:** pair_0856

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0856", "source_row_index": 187766, "fold": 1, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 438086747340, "name": "zvezda / сборная модель пассажирский авиалайнер аэробус а320neo, 1/", "attributes": {"материал изделия": "пластик", "количество элементов": "149", "масштаб": "1/144", "длина готовой модели": "25 см", "вес товара с упаковкой (г)": "500 г", "длина упаковки": "30 см", "высота упаковки": "10 см", "ширина упаковки": "15 см", "валюта": "rub"}, "formatted_text_length": 323}, "item2": {"id": 412316863719, "name": "сборная модель пассажирский авиалайнер аэробус а321neo", "attributes": {"вид техники": "военная", "комплектация": "без клея в комплекте", "бренд": "sembo block", "цвет товара": "разноцветный", "страна-изготовитель": "россия", "материал": "abs пластик", "минимальный возраст ребенка": "от 9 лет", "длина модели, см": "12.1", "нация": "сша", "масштаб модели для сборки": "1/35", "тип": "сборная модель военной техники", "количество элементов, шт": "270"}, "formatted_text_length": 431}}
```

## Pattern 5

**Pattern:** Artist easels are negative when their explicitly named construction type differs.

**Concrete difference:** The first is a lyre easel with a fixed mast, while the second is a tripod easel.

**Why this is a strict negative signal:** “Лира” and “тренога” are different easel constructions, not merely alternate wording for one product.

**Evidence pair id:** pair_0857

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0857", "source_row_index": 128795, "fold": 1, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 300647808116, "name": "не определен мольберт лира классика (неподвижная мачта) arg-6, артикул arg-6", "attributes": {"цвет товара": "светлое дерево", "тип": "мольберт", "вид мольберта": "лира", "особенности": "регулировка высоты, регулировка наклона"}, "formatted_text_length": 223}, "item2": {"id": 9395, "name": "мольберт деревянный, 145 см.", "attributes": {"бренд": "galleon", "цвет товара": "светло-коричневый", "страна-изготовитель": "китай", "материал": "сосна", "наличие полочки у мольберта": "нет", "тип": "мольберт художественный", "тип мольберта": "тренога"}, "formatted_text_length": 243}}
```

## Pattern 6

**Pattern:** Mosaic sets are negative when their explicitly stated number and size range of elements, as well as board configuration, differ.

**Concrete difference:** One mosaic is named as 60 elements of 15 mm, while the other is 270 elements of 10, 15, and 20 mm with two boards.

**Why this is a strict negative signal:** The differing component inventory defines two distinct play sets even though both are round mosaics with six colours.

**Evidence pair id:** pair_0860

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0860", "source_row_index": 204641, "fold": 1, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 481036369424, "name": "десятое королевство мозаика круглая, 60 элементов по 15 мм, 6 цветов", "attributes": {"тип": "мозаика", "количество элементов": "15 шт.", "минимальный возраст": "3 лет", "материал": "пластик"}, "formatted_text_length": 187}, "item2": {"id": 523986013548, "name": "мозаика круглая, 270 элементов по 10, 15, 20 мм, 6 цветов, 2 платы", "attributes": {"бренд": "десятое королевство", "страна-изготовитель": "россия", "тип": "мозаика"}, "formatted_text_length": 163}}
```

## Pattern 7

**Pattern:** Buttons with different diameter and different number of sewing holes are negative variants.

**Concrete difference:** The first button is 18 mm with four holes, whereas the second is 31 mm with two holes.

**Why this is a strict negative signal:** Diameter and hole count are functional, physical specifications that select a different button model.

**Evidence pair id:** pair_0862

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0862", "source_row_index": 50582, "fold": 4, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 111669286476, "name": "пуговица nn 03955 d 18мм 4 прокола (10шт)", "attributes": {"бренд": "нет бренда", "количество отверстий в пуговицах": "4", "название цвета": "501-молочный", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-5/6354081929.jpg", "длина, см": "1.8", "материал": "пластик", "количество в упаковке, шт": "10", "тип": "пуговицы", "вид застежки": "пуговицы/пуговица"}, "formatted_text_length": 348}, "item2": {"id": 266288039929, "name": "000-пуговка / пуговица nn 03816 d 31мм 2 прокола (10шт)", "attributes": {"цвет": "молочный", "количество предметов в упаковке": "10 шт.", "материал изделия": "пластик", "диаметр пуговиц": "31", "назначение пуговиц": "для одежды; для творчества и рукоделия; для изготовления одежды", "тип пуговиц": "пришивные", "особенности пуговиц": "красивые", "форма пуговиц": "круглые", "страна производства": "россия", "комплектация": "10 шт.в упаковке", "валюта": "rub"}, "formatted_text_length": 440}}
```

## Pattern 8

**Pattern:** Different named wooden puzzle forms are negative even when both are single-player wooden puzzles for the same age group.

**Concrete difference:** One puzzle is “Волшебный круг” and the other is “Колумбово яйцо”.

**Why this is a strict negative signal:** The titles name the puzzle’s form and task, which are the core product identity rather than a generic puzzle attribute.

**Evidence pair id:** pair_0866

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0866", "source_row_index": 163791, "fold": 2, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 377957284895, "name": "головоломка 2.0 «волшебный круг», 083216 woodland (сибирский сувенир)", "attributes": {"артикул производителя": "4624186", "бренд": "woodland", "возрастное ограничение": "3+", "тип": "3d головоломка", "формат": "дорожная версия", "количество игроков, минимальное": "1", "количество игроков, максимальное": "1", "материал головоломки": "дерево", "производитель": "woodland toys", "код товара": "100025614898"}, "formatted_text_length": 391}, "item2": {"id": 627065281171, "name": "головоломка \"колумбово яйцо \"", "attributes": {"возраст ребенка": "от 3 лет", "артикул": "3980637", "бренд": "нет бренда", "страна-изготовитель": "россия", "вид выпуска товара": "фабричное производство", "материал": "дерево", "минимальный возраст ребенка": "от 3 лет", "упаковка": "пленка", "тип": "головоломка"}, "formatted_text_length": 297}}
```

## Pattern 9

**Pattern:** Diamond-painting kits are negative when they jointly differ in canvas size, coverage type, support, and the artwork’s stated theme.

**Concrete difference:** The first is a 25×21 cm partial-layout cardboard kit with a children’s-space plot; the second is a 40×50 cm full-layout framed canvas with an astrology theme.

**Why this is a strict negative signal:** This conjunction identifies materially different kit configurations, avoiding any inference from the shared generic product type alone.

**Evidence pair id:** pair_0869

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0869", "source_row_index": 288044, "fold": 3, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 678604837232, "name": "не определен стиль для дома алмазная мозаика 25х21 см / частичная выкладка", "attributes": {"бренд": "стиль для дома", "тип": "набор алмазной мозаики", "выкладка": "частичная", "форма страз": "круглая", "материал страз": "акрил", "количество цветов": "12", "количество полотен": "1", "сюжет": "детский, космос", "особенности": "лоток для страз, пинцет, стилус, холст", "ширина": "20 см", "высота": "2 см", "основа": "картонная"}, "formatted_text_length": 407}, "item2": {"id": 566935848125, "name": "алмазная мозаика. картина стразами на подрамнике (размер 40х50 см, полная выкладка, круглые стразы)", "attributes": {"возрастные ограничения": "от 12 лет", "бренд": "новый мир", "основа холста": "подрамник", "цвет товара": "разноцветный", "название цвета": "ga74245", "страна-изготовитель": "китай", "форма страз": "круглая", "размеры, мм": "400 х 500", "выкладка/вышивка": "полная", "длина по горизонтали, см": "50", "материалы набора": "стразы", "число цветов, шт": "25", "комплектация": "в комплекте: 1) клеевая основа на холсте с подрамником (размер 40х50 см) – 1 шт; 2) ручка (стилус) – 1 шт; 3) клей для ручки – 1 шт; 4) лоток для страз – 1 шт; 5) стразы круглые матовые – комплект по цветам.", "основа картины": "холст цветной", "тематика": "астрология и знаки зодиака", "число полотен": "1", "тип": "алмазная мозаика", "длина по вертикали, см": "40"}, "formatted_text_length": 825}}
```

## Pattern 10

**Pattern:** Double-pointed knitting needles are negative variants when their diameter differs while the construction and length agree.

**Concrete difference:** One set has 3 mm needles and the other has 2 mm needles; both are 21 cm, three-piece double-pointed sets.

**Why this is a strict negative signal:** Needle diameter controls the usable stitch gauge, so it distinguishes the sold tool even where the rest of the set specification matches.

**Evidence pair id:** pair_0872

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0872", "source_row_index": 116128, "fold": 3, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 274877915256, "name": "спицы для вязания, чулочные, d 3 мм, 21 см, 3 шт", "attributes": {"бренд": "нет бренда", "цвет товара": "разноцветный", "страна-изготовитель": "китай", "длина, см": "21", "материал": "металл", "вид спиц": "чулочные", "тип": "спицы для вязания"}, "formatted_text_length": 233}, "item2": {"id": 137439061906, "name": "арт узор спицы для вязания, чулочные, гибкие, d = 2 мм, 21 см, 3 шт", "attributes": {"тип": "спицы", "тип спиц": "чулочные", "покрытие": "без покрытия", "материал": "металл"}, "formatted_text_length": 169}}
```

## Pattern 11

**Pattern:** A phoneme-specific speech-therapy puzzle is negative when it targets a different explicitly named sound.

**Concrete difference:** One “Круговые пазлы” game teaches sound “З”, whereas the other teaches sound “Л”.

**Why this is a strict negative signal:** The target phoneme determines the educational content of the game, despite the shared format and minimum age.

**Evidence pair id:** pair_0885

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0885", "source_row_index": 121116, "fold": 3, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 283467913633, "name": "логопедическая игра круговые пазлы. звук з , 3", "attributes": {"целевая аудитория": "детская", "бренд": "нет бренда", "страна-изготовитель": "китай", "вид выпуска товара": "фабричное производство", "минимальный возраст ребенка": "от 3 лет", "тип": "настольная игра"}, "formatted_text_length": 258}, "item2": {"id": 420906917004, "name": "лас играс / логопедическая игра «круговые пазлы. звук «л», 3+", "attributes": {"страна производства": "китай", "валюта": "rub"}, "formatted_text_length": 126}}
```

## Pattern 12

**Pattern:** Stretched artist canvases are negative when their stated dimensions and aspect ratio differ.

**Concrete difference:** One cotton canvas on a stretcher is 30×60 cm, while the other is 35×45 cm.

**Why this is a strict negative signal:** Canvas dimensions determine the physical product and painting surface, even though both are white cotton canvases on stretchers from the same brand.

**Evidence pair id:** pair_0890

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0890", "source_row_index": 326892, "fold": 4, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 764504240329, "name": "холст этюдный, 30х60 см", "attributes": {"бренд": "туюкан", "зернистость": "мелкозернистый", "тип холста": "на подрамнике", "основа холста": "подрамник", "цвет товара": "белый", "страна-изготовитель": "россия", "плотность, г/м2": "256", "размеры, мм": "300х600", "форма холста": "прямоугольник", "длина меньшей стороны, см": "30", "материал холста": "хлопок", "грунт": "эмульсионный", "предназначение холста": "масло", "единиц в одном товаре": "1", "тип": "холст", "тип подрамника": "модульный", "длина большей стороны, см": "60"}, "formatted_text_length": 499}, "item2": {"id": 42949826169, "name": "туюкан / холст этюдный, 35х45 см", "attributes": {"материал холста": "хлопок", "тип холста": "на подрамнике", "доп. опции холста": "классическая натяжка; прогрунтован", "страна производства": "россия", "цвет": "белый", "комплектация": "холст", "вес товара с упаковкой (г)": "700 г", "длина упаковки": "45 см", "высота упаковки": "35 см", "ширина упаковки": "2 см", "валюта": "rub"}, "formatted_text_length": 362}}
```

## Pattern 13

**Pattern:** Natural-leather sheets are negative when the stated sheet format differs.

**Concrete difference:** The first Crazy Horse leather sheet is A3 and the second is A4, with corresponding 42 cm versus 21×30 cm dimensions.

**Why this is a strict negative signal:** A3 and A4 are distinct physical sheet sizes, so the offers cannot denote the same leather cut.

**Evidence pair id:** pair_0895

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0895", "source_row_index": 128392, "fold": 1, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 300647792399, "name": "кожа натуральная а3 crazy horse 1,4 мм.", "attributes": {"бренд": "нет бренда", "цвет товара": "коричневый", "длина, см": "42", "материал": "кожа", "количество в упаковке, шт": "1", "единиц в одном товаре": "1", "тип": "природные материалы"}, "formatted_text_length": 230}, "item2": {"id": 652835041494, "name": "без бренда кожа натуральная а4 crazy horse 1,4 мм.", "attributes": {"цвет товара": "фуксия", "тип": "природные материалы", "назначение": "рукоделие", "ширина": "0.21 м", "длина": "0.3 м"}, "formatted_text_length": 180}}
```

## Pattern 14

**Pattern:** Workbooks with the same publisher and line-template count are negative when their named learning topic differs.

**Concrete difference:** One “Лесная мастерская” copybook is “Игралочка”, while the other is the “Транспорт” version.

**Why this is a strict negative signal:** The title specifies the educational content; the common publisher and eight-line format are not enough to make different themed editions identical.

**Evidence pair id:** pair_0898

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0898", "source_row_index": 70147, "fold": 2, "category": "Хобби и творчество", "target": 0.0, "item1": {"id": 163208817785, "name": "лесная мастерская пропись с линейками «игралочка», 8 линеек", "attributes": {"издательство": "лесная мастерская", "темы обучения": "письмо"}, "formatted_text_length": 139}, "item2": {"id": 549755883552, "name": "лесная мастерская, пропись с линейками \"транспорт\", 8 линеек в наборе", "attributes": {"возраст ребенка": "от 3 лет", "бренд": "лесная мастерская", "цвет товара": "разноцветный", "страна-изготовитель": "китай", "материал": "бумага", "минимальный возраст ребенка": "от 3 лет", "количество в упаковке, шт": "1", "пол ребенка": "унисекс", "тематика": "автомобили и мотоциклы", "тип": "трафарет для рисования"}, "formatted_text_length": 389}}
```

## Annotation caveats

- `pair_0864` is labelled negative although the names describe what appears to be the same Art Uzor 10 mm, 10 m lace elastic; the second colour name is truncated to “м”, so no reliable difference is available.
- `pair_0871` is a negative pair of 30×40 cm, full-layout, round-rhinestone diamond mosaics; the titles and incomplete fields do not establish a safely reusable distinction.
- `pair_0873` is negative despite the same brand, 17×23 cm, partial round-rhinestone framed diamond-mosaic configuration; one title names “Святая Анастасия” and the other is generic, so it may be an underspecified near-duplicate.
- `pair_0876` is negative although the titles are effectively identical and the core 40×50 cm framed, full-layout diamond-mosaic specifications agree; the descriptive attributes conflict, making it unsuitable as evidence.
- `pair_0879` has conflicting thread colour information (545/457 versus 356, with inconsistent colour fields) despite otherwise matching Maxag 50/2, 5000-yard specifications.
- `pair_0882` has identical product names but different stated zipper colours (beige versus black); it may be a valid colour variant, but the name hides the distinction.
- `pair_0883` has the same code, quantity, and nominal size with an incomplete first colour name; treat the colour distinction as uncertain.
- `pair_0888` is labelled negative for what may be the same Step Puzzle “Ми-ми-мишки” product; only the left record explicitly gives 260 pieces.
- `pair_0889` is positive despite a generic “пазлы для малышей” listing being paired with a specifically named 14-piece New Year vehicle puzzle, so a thematic difference cannot be used when one product is underspecified.
- `pair_0891` is a negative near-match for the same “Шашки, нарды, шахматы” game; the player counts conflict (2 versus 2–4), but the edition distinction is not otherwise clear.
- `pair_0892` and `pair_0893` are close same-family craft-kit pairs with different dimensions or design names; their sparse and inconsistent attributes make them weaker than the patterns above.
- Positive `pair_0859` pairs a generic 24×30 partial diamond mosaic with a named religious design, and positive `pair_0894` pairs a generic 33×43 full diamond mosaic with a specific listing. These show that a product-type match or a single size/configuration match alone is insufficient.
