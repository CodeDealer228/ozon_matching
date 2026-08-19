## Pattern 1

**Pattern:** Kitchen or dining tables with different tabletop footprints are negative even when their material, style, height, and construction are otherwise close.

**Concrete difference:** The first Solarius table is 140×60×75 cm with a white finish, while the second is 90×60×75 cm and unpainted/beige.

**Why this is a strict negative signal:** A 50 cm difference in the table's long side denotes a separate size variant rather than the same purchasable table.

**Evidence pair id:** pair_0455

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0455", "source_row_index": 245243, "fold": 4, "category": "Мебель", "target": 0.0, "item1": {"id": 575525645838, "name": "solarius стол кухонный классика из массива сосны, белая столешница 40 мм и ножки, 140х60х75 см", "attributes": {"цвет товара": "белый", "тип": "стол кухонный", "конструкция": "стандартный", "назначение": "для кухни и столовой", "форма столешницы": "прямоугольная", "материал столешницы": "массив дерева, сосна", "толщина столешницы": "4 см", "материал основания": "массив дерева", "материал ножек": "массив дерева", "количество ножек": "4 шт.", "стиль": "классический", "размеры (дхшхв)": "60x140x75 см", "вес": "23.9 кг", "габариты транспортной упаковки": "144х64х14 см", "вес в транспортной упаковке": "23.9 кг"}, "formatted_text_length": 587}, "item2": {"id": 154618908455, "name": "стол кухонный классика из массива сосны, столешница 40 мм, без шлифовки и покраски, 90х60х75 см", "attributes": {"артикул": "600-10350", "бренд": "solarius", "цвет товара": "бежевый", "название цвета": "бежевый", "страна-изготовитель": "россия", "глубина, см": "60", "форма поставки": "в разобранном виде", "форма столешницы": "прямоугольник", "высота, см": "75", "вид выпуска товара": "фабричное производство", "материал столешницы": "массив дерева", "количество ножек": "4", "комплектация": "стол - 1 шт", "назначение (помещение)": "для дачи", "стиль дизайна": "классический", "длина столешницы, см": "60", "толщина столешницы, мм": "40", "вид стола": "обеденный", "ширина столешницы, см": "90", "способ раскладывания столешницы": "нераскладной", "тип": "стол обеденный", "вес товара, г": "16200", "ширина, см": "90", "регулировка по высоте": "нет"}, "formatted_text_length": 806}}
```

## Pattern 2

**Pattern:** Beds from the same named model line with different sleeping-place widths are negative variants.

**Concrete difference:** The two Proxson Milton beds specify 140×200 cm versus 80×200 cm sleeping places.

**Why this is a strict negative signal:** Sleeping-place width is a core buyable bed variant; the shared model name and mechanism do not make the 140 cm and 80 cm versions identical.

**Evidence pair id:** pair_0461

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0461", "source_row_index": 184721, "fold": 4, "category": "Мебель", "target": 0.0, "item1": {"id": 429496836105, "name": "кровать proxson milton 140х200 с подъемным механизмом, двуспальная, односпальная", "attributes": {"артикул": "нмо-3740786", "ширина спального места, см": "140", "вид основания": "ортопедические ламели", "бренд": "pro × son", "покрытие корпуса": "матовое", "цвет товара": "бежевый", "название цвета": "дуб шамони светлый (124)/дуб кантербери", "тип матраса": "без матраса", "страна-изготовитель": "россия", "форма поставки": "в разобранном виде", "конструктивные особенности": "ортопедические свойства", "высота изножья, см": "34.5", "расположение изголовья": "снизу", "высота, см": "85", "высота изголовья, см": "85", "исполнение": "деревянное", "длина, см": "200", "длина спального места, см": "200", "ящики для белья": "нет", "вид выпуска товара": "фабричное производство", "материал обивки": "без обивки", "материал корпуса": "лдсп", "матрас в комплекте": "нет", "макс. высота матраса, см": "25", "макс. нагрузка, кг": "120", "назначение (помещение)": "для дома", "жесткость матраса": "переменный", "стиль дизайна": "английский", "тип": "двуспальная кровать", "гарантийный срок": "18 мес.", "вес товара, г": "105622", "ширина, см": "140", "с ножками": "да", "подъемный механизм": "да"}, "formatted_text_length": 1123}, "item2": {"id": 137438979739, "name": "не определен кровать proxson milton 80х200 с подъемным механизмом, двуспальная, односпальная", "attributes": {"бренд": "pro × son", "тип": "двуспальная кровать", "стиль": "английский", "число мест": "односпальная", "особенности": "ортопедические свойства", "тип основания": "ортопедические ламели", "материал каркаса": "лдсп", "материал обивки": "отсутствует", "максимальная нагрузка": "120 кг", "высота кровати": "85 см", "высота изголовья": "85 см", "длина кровати": "200 см", "ширина кровати": "80 см", "длина спального места": "200 см", "ширина спального места": "80 см", "габариты транспортной упаковки": "200х80х30 см", "вес в транспортной упаковке": "68 кг"}, "formatted_text_length": 634}}
```

## Pattern 3

**Pattern:** An adult multi-person dining table is a strict negative match for a children's height-adjustable table-and-chair offering.

**Concrete difference:** Item 1 is a 120×75×73 cm non-folding dining table for six, whereas item 2 is a children's table with a chair and height adjustment.

**Why this is a strict negative signal:** The intended user, configuration, and included furniture differ, so these are not alternative listings of one item.

**Evidence pair id:** pair_0465

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0465", "source_row_index": 199918, "fold": 1, "category": "Мебель", "target": 0.0, "item1": {"id": 463856601029, "name": "стол обеденный нераскладной стол 120, 75х75х73 см", "attributes": {"артикул": "стол обеденный 75*75 (сосна)", "бренд": "нет бренда", "цвет товара": "бежевый", "название цвета": "сосна", "страна-изготовитель": "россия", "глубина, см": "75", "форма поставки": "в разобранном виде", "форма столешницы": "прямоугольник", "высота, см": "73", "исполнение": "деревянное", "вид выпуска товара": "фабричное производство", "материал столешницы": "массив дерева", "количество ножек": "4", "комплектация": "1 шт", "назначение (помещение)": "для дома", "стиль дизайна": "лофт", "длина столешницы, см": "120", "толщина столешницы, мм": "70", "вид стола": "обеденный", "ширина столешницы, см": "75", "способ раскладывания столешницы": "нераскладной", "тип": "стол обеденный", "гарантийный срок": "24 ", "ширина, см": "75", "количество персон": "6"}, "formatted_text_length": 786}, "item2": {"id": 257698104400, "name": "не определен стол со стулом детский, свала , сосна", "attributes": {"бренд": "не известен", "тип": "стол", "особенности": "регулировка высоты", "материал каркаса": "массив дерева", "материал столешницы": "массив дерева", "габариты транспортной упаковки": "59х50х50 см", "вес в транспортной упаковке": "10 кг"}, "formatted_text_length": 298}}
```

## Pattern 4

**Pattern:** A bedside cabinet with an open shelf and no doors is negative against an office cabinet with two drawers, even when the outer dimensions agree.

**Concrete difference:** Both cabinets are 41×34×55 cm, but the first is the Sokol КТ-09,1 bedside open-shelf cabinet and the second is an office cabinet with two drawers.

**Why this is a strict negative signal:** Storage construction and intended placement are explicit, decisive product specifications, preventing dimensions alone from causing a false match.

**Evidence pair id:** pair_0468

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0468", "source_row_index": 323222, "fold": 1, "category": "Мебель", "target": 0.0, "item1": {"id": 755914286860, "name": "тумба прикроватная сокол кт-09,1", "attributes": {"бренд": "сокол", "модель": "кт-09,1", "артикул производителя": "sk_52254", "цвет производителя": "венге; дуб беленный", "форма поставки": "в разобранном виде", "способ установки": "напольный", "корпус": "полуоткрытый", "дверцы": "нет", "открытые полки": "1", "ширина, см": "41", "глубина, см": "34", "высота, см": "55", "ножки": "нет", "колесики": "нет", "ручки": "да", "задняя стенка у открытых полок": "да", "необычный дизайн": "нет", "замок": "нет", "подходит для спальни": "да", "открытие дверцы нажатием": "нет", "игровая (геймерская)": "нет", "покрытие": "матовое", "код товара": "100024343328"}, "formatted_text_length": 609}, "item2": {"id": 360777310326, "name": "мебельная фабрика сокол тумба офисная тумба офисная_1073965_мебельная фабрика сокол, 2 ящ., 41х34х55 см", "attributes": {"артикул": "1073965", "бренд": "мебельная фабрика сокол", "цвет товара": "темно-коричневый", "страна-изготовитель": "россия", "глубина, см": "34", "форма поставки": "в разобранном виде", "материал фасада": "лдсп", "высота, см": "55", "вид выпуска товара": "фабричное производство", "материал корпуса": "лдсп", "стиль дизайна": "современный", "тип": "тумба офисная", "гарантийный срок": "24 мес", "ширина, см": "41", "количество ящиков": "2"}, "formatted_text_length": 535}}
```

## Pattern 5

**Pattern:** Garden-swing cushions with incompatible intended shape and footprint are negative, even if their color matches.

**Concrete difference:** The first is a 100×70×5 cm waterproof garden-furniture cushion; the second is a 140×45 cm swing seat cushion.

**Why this is a strict negative signal:** The dimensions and stated product form (general cushion versus swing seat) identify different fitted cushion variants.

**Evidence pair id:** pair_0470

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0470", "source_row_index": 124953, "fold": 4, "category": "Мебель", "target": 0.0, "item1": {"id": 292057864914, "name": "подушка для садовой мебели красная пуговица 70x100 см, 5 см серая", "attributes": {"артикул производителя": "sp0570100gre", "бренд": "красная пуговица", "тип товара": "подушка для садовой мебели", "материал обивки": "оксфорд", "материал наполнителя": "ппу", "длина подушки, в сантиметрах": "100", "ширина подушки, в сантиметрах": "70", "толщина подушки, в сантиметрах": "5", "подходит для": "садовых качелей", "особенности": "водонепроницаемая поверхность", "цвет": "серый", "цвет производителя": "серый", "количество в упаковке": "1", "страна производства товара": "россия", "габариты товара (дхшхв), в сантиметрах": "100x70x5", "код товара": "600015076256"}, "formatted_text_length": 630}, "item2": {"id": 249108127195, "name": "сиденье для качелей,45х140см", "attributes": {"целевая аудитория": "взрослая", "артикул": "sp0545140gre", "бренд": "нет бренда", "цвет товара": "серый", "название цвета": "серый 45x140", "страна-изготовитель": "россия", "форма поставки": "в собранном виде", "исполнение": "бескаркасное", "длина, см": "140", "вид выпуска товара": "фабричное производство", "комплектация": "подушка 1 шт.", "назначение (помещение)": "для дачи", "тип": "сиденье для качелей", "вес товара, г": "1000", "ширина, см": "45"}, "formatted_text_length": 474}}
```

## Pattern 6

**Pattern:** Bathroom vanity sets are negative when their included basin models have different shapes and dimensions.

**Concrete difference:** The first Bauedge set contains a round 41×41 cm built-in basin, while the second contains a rectangular Bau Hotel basin sized 48×14×37 cm.

**Why this is a strict negative signal:** The basin is a named, included component of each three-piece set; differing geometry and size make the sets distinct purchasable configurations.

**Evidence pair id:** pair_0476

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0476", "source_row_index": 99272, "fold": 1, "category": "Мебель", "target": 0.0, "item1": {"id": 231928274489, "name": "bauedge комплект: тумба bau dream blackwood графит 60,раковина bau element,смеситель hotel stil", "attributes": {"серия": "dream", "бренд": "bauedge", "артикул производителя": "bd2850", "модель": "bd1060-bl_ns0001-dg", "монтаж": "подвесной", "высота, см": "45", "ширина, см": "60", "глубина, см": "45", "тип раковины": "встраиваемая", "форма раковины": "круглая", "длина раковины, см": "41", "ширина раковины, см": "41", "материал корпуса": "лдсп", "материал фасадов": "лдсп", "цвет": "черный", "количество ящиков": "1", "установка над стиральной машиной": "нет", "комплектация": "тумба; раковина; смеситель", "страна-производитель": "россия", "код товара": "600016013106"}, "formatted_text_length": 636}, "item2": {"id": 309237737782, "name": "комплект: тумба bau dream blackwood 60, 1 ящик, раковина прямоугольная bau hotel 48х37, мрамор, смеситель hotel stil, черный матовый", "attributes": {"артикул": "bd3013", "расположение": "настенная", "бренд": "bauedge", "покрытие корпуса": "влагостойкое", "покрытие фасада": "матовое", "цвет товара": "бежевый", "название цвета": "дуб/белый/мрамор/черный", "страна-изготовитель": "россия", "глубина, см": "45", "форма поставки": "в собранном виде", "материал фасада": "лдсп", "ориентация": "универсальный", "высота, см": "45", "исполнение": "деревянное", "вид выпуска товара": "фабричное производство", "материал корпуса": "лдсп", "количество предметов в комплекте": "3", "тип открывания дверей": "без дверей", "комплектация": "тумба под раковину;раковина;смеситель;документация", "назначение (помещение)": "для ванной", "форма раковины": "прямоугольник", "размер раковины,см (ш*в*г)": "48х14х37", "стиль дизайна": "классический", "тип": "тумба с раковиной", "особенности": "крепление к стене", "гарантийный срок": "2 года", "ширина, см": "60", "раковина в комплекте": "да"}, "formatted_text_length": 1021}}
```

## Pattern 7

**Pattern:** Kitchen countertops with different length and thickness specifications are negative variants despite sharing the same brand and decor.

**Concrete difference:** Both Skif countertops are “дуб ниагара” and 600 mm wide, but one is 3050×600×26 mm and the other is 1900×600×38 mm.

**Why this is a strict negative signal:** Cut length and thickness determine the physical countertop SKU and cannot be reconciled as the same item.

**Evidence pair id:** pair_0482

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0482", "source_row_index": 32859, "fold": 1, "category": "Мебель", "target": 0.0, "item1": {"id": 68719627647, "name": "столешница для кухни скиф 3050х600x26мм с кромкой. цвет - дуб ниагара", "attributes": {"артикул": "stol30-26x3050", "бренд": "скиф", "цвет товара": "темно-коричневый", "название цвета": "дуб ниагара", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-h/6083606093.jpg", "страна-изготовитель": "россия", "форма поставки": "в собранном виде", "форма столешницы": "прямоугольник", "высота, мм": "26", "длина, мм": "3050", "вид выпуска товара": "фабричное производство", "материал столешницы": "влдсп", "назначение мебельной фурнитуры": "универсальное", "комплектация": "столешница - 1 шт, кромка с клеевым слоем - 2 шт по 60 см.", "способ раскладывания столешницы": "нераскладной", "тип": "столешница", "гарантийный срок": "1 год", "вес товара, г": "27000", "ширина, мм": "600"}, "formatted_text_length": 744}, "item2": {"id": 60129598561, "name": "столешница для кухни скиф 1900х600x38мм с кромкой. цвет - дуб ниагара", "attributes": {"цвет товара": "дуб ниагара", "тип": "столешница", "назначение": "для кухонных гарнитуров", "материал": "лдсп", "длина": "190 см", "ширина": "60 см", "глубина": "3.8", "толщина": "3.8 см", "вес": "27.72 кг", "форма столешницы": "прямоугольная", "форма внешнего края": "скругленный край", "особенности": "влагостойкость", "комплектация": "кромка в цвет столешницы"}, "formatted_text_length": 428}}
```

## Pattern 8

**Pattern:** Named half-bar chair models are negative when their model names and core construction differ, even if color and use are similar.

**Concrete difference:** The first is Woodville “баодин к” with textile upholstery and 50 cm width; the second is “алст к” with natural velour upholstery, metal body, 47 cm width, and a 101 cm height.

**Why this is a strict negative signal:** The model identifiers and explicit structural specifications distinguish two separate rotating half-bar chair products.

**Evidence pair id:** pair_0487

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0487", "source_row_index": 250749, "fold": 2, "category": "Мебель", "target": 0.0, "item1": {"id": 584115670221, "name": "woodville полубарный стул баодин к б/к крутящийся латте / черный", "attributes": {"цвет товара": "бежевый", "тип": "стул", "ширина": "50 см", "материал обивки": "текстиль", "назначение": "полубарный", "габариты транспортной упаковки": "53х61х68 см", "вес в транспортной упаковке": "9 кг"}, "formatted_text_length": 277}, "item2": {"id": 223338319830, "name": "полубарный стул алст к крутящийся латте / черный", "attributes": {"артикул": "36176", "бренд": "мебели.net", "глубина, см": "40", "высота, см": "101", "вид выпуска товара": "фабричное производство", "материал обивки": "велюр натуральный", "материал корпуса": "металл", "тип": "полубарный стул", "ширина, см": "47", "регулируемая спинка": "нет", "с подлокотниками": "нет", "регулировка по высоте": "нет"}, "formatted_text_length": 382}}
```

## Annotation caveats

- `pair_0452` is labelled negative although both listings describe an E1 Express two-door mirrored wardrobe sized 180×45×220; the visible distinction is chiefly “слоновая кость” versus “белый снег”. Treat its label as support only for an explicit color/SKU variant, not for a broad wardrobe rule.
- `pair_0472` and `pair_0473` have very close or matching sofa dimensions and specifications while being labelled negative; incomplete model names and conflicting metadata make them unsuitable as general rules.
- Positive `pair_0494` changes the Triks cabinet height (220 versus 200), interior filling, and mirror/drawer features; positive `pair_0499` changes the Linda bed's legs and lifting/storage configuration. These labels are counterevidence to any unconditional rule that every size or feature difference is negative.
