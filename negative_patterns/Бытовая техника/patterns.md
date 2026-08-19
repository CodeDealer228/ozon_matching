## Pattern 1

**Pattern:** A pair is a strict negative when the listings describe different appliance types with incompatible primary functions, even if colour or country of manufacture overlap.

**Concrete difference:** The first item is a battery-powered ceramic night light, while the second is a steam humidifier with a water reservoir and humidification function.

**Why this is a strict negative signal:** Lighting and air humidification are different product functions and do not identify the same sellable item.

**Evidence pair id:** pair_0147

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0147", "source_row_index": 149260, "fold": 1, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 352187356078, "name": "led electric / керамический светильник ночник", "attributes": {"глубина предмета": "5.5 см", "ширина предмета": "10 см", "высота предмета": "12 см", "питание": "от батареек", "степень пылевлагозащиты": "ip20", "страна производства": "китай", "гарантийный срок": "1 год", "цвет": "белый", "декоративные элементы": "без элементов", "комплектация": "светильник", "вес с упаковкой (кг)": "0.75 кг", "длина упаковки": "22 см", "высота упаковки": "12 см", "ширина упаковки": "21 см", "валюта": "rub"}, "formatted_text_length": 467}, "item2": {"id": 747324357573, "name": "modengo увлажнитель воздуха a301_белое, белый", "attributes": {"фильтры": "водяной фильтр", "макс. площадь помещения, кв.м": "18", "автоотключение": "да", "основной материал корпуса": "пластик", "бренд": "modengo", "цвет товара": "белый", "название цвета": "белое", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-z/6860746151.jpg", "страна-изготовитель": "китай", "размеры, мм": "81*81*109", "вид увлажнителя": "паровой", "установка увлажнителя": "на стол", "комплектация": "usb-кабель для зарядки, увлажнитель воздуха", "мощность, вт": "1.5", "системы защиты": "автоотключение при отсутствии воды", "температура пара": "холодный пар", "тип выключателя": "механический", "верхний долив воды": "да", "тип": "увлажнитель воздуха", "объем, мл": "250", "гарантийный срок": "3 месяца", "вес товара, г": "120", "производительность, л/ч": "0.045"}, "formatted_text_length": 804}}
```

## Pattern 2

**Pattern:** Gas and induction cooking panels are strict negatives when the panel's heating technology is explicitly different.

**Concrete difference:** One Weissgauff panel is gas-fired with gas burners, whereas the other is an induction panel with touch control and residual-heat indication.

**Why this is a strict negative signal:** The energy source and heating mechanism are core product-defining specifications, not interchangeable listing details.

**Evidence pair id:** pair_0125

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0125", "source_row_index": 8479, "fold": 4, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 17179908772, "name": "weissgauff / газовая варочная панель hgg 641 bgh, черный", "attributes": {"модель": "hgg 641 bgh", "цвет": "черный", "управление": "механическое", "страна производства": "китай", "ширина предмета": "60 см", "глубина предмета": "51 см", "тип рабочей поверхности": "закаленное стекло", "тип варочной панели": "газовая", "число конфорок": "4", "доп. опции варочной поверхности": "автоматический электроподжиг, вок-конфорка, газ-контроль конфорок", "тип нагревательного элемента": "газовый", "тип конфорок": "газовые", "мощность устройства": "6.2 квт", "наличие рамки": "нет", "решетки": "чугунные", "защита от детей": "нет", "гарантийный срок": "1 год", "комплектация": "варочная панель - 1 шт; инструкция", "вес с упаковкой (кг)": "13.7 кг", "вес без упаковки (кг)": "13.3 кг", "длина упаковки": "15 см", "высота упаковки": "55 см", "ширина упаковки": "64 см", "валюта": "rub"}, "formatted_text_length": 830}, "item2": {"id": 481036408253, "name": "weissgauff индукционная варочная панель hi 641 bsc, черный", "attributes": {"бренд": "weissgauff", "цвет товара": "черный", "тип управления": "сенсорное", "страна-изготовитель": "китай", "размеры ниши для встраивания панели, мм": "560 x 490", "особенности варочной панели": "быстрый нагрев", "наличие рамки": "нет", "тип панели (плиты)": "индукционная", "индикация": "остаточное тепло", "партномер": "423434", "количество конфорок": "4", "количество конфорок с овальным расширением": "1", "конструкция панели": "независимая", "края варочной панели": "обработанные без рамки", "установка": "стандартная", "системы защиты": "блокировка от детей", "материал рабочей поверхности панели": "закаленное стекло", "таймер": "с отключением конфорок", "тип": "варочная панель", "гарантийный срок": "1 год"}, "formatted_text_length": 758}}
```

## Pattern 3

**Pattern:** A fully gas stove and a combined gas-electric stove are strict negatives when their oven and burner configurations differ.

**Concrete difference:** The first product is the Gefest 6100-02 001 gas stove, while the second has a gas-electric oven and three gas plus one electric burner.

**Why this is a strict negative signal:** The fuel configuration changes the appliance itself and its installation and use requirements.

**Evidence pair id:** pair_0139

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0139", "source_row_index": 357943, "fold": 1, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 833223806840, "name": "плита газовая gefest 6100-02 0001 (6100-02 к), коричневый", "attributes": {"бренд": "gefest", "страна-изготовитель": "беларусь", "партномер": "6100-02 001", "тип": "плита"}, "formatted_text_length": 167}, "item2": {"id": 352187401856, "name": "комбинированная плита gefest 6110-02 0301", "attributes": {"габариты (вxшxг) (см)": "85x60x60", "тип духовки": "газоэлектрическая", "количество конфорок": "3 газовые + 1 электрическая", "электроподжиг конфорок": "есть", "гриль": "электрический", "вертел": "есть", "таймер": "меxанический", "цвет": "коричневый", "вес (кг)": "44.5", "гарантия": "2 года", "страна-производитель": "беларусь"}, "formatted_text_length": 370}}
```

## Pattern 4

**Pattern:** Built-in dishwashers with different form factor, width, and place-setting capacity are strict negatives.

**Concrete difference:** Electrolux KEGB9420W is a 59.6 cm full-size 15-place-setting machine, while EEA12100L is a 44.6 cm narrow 9-place-setting machine.

**Why this is a strict negative signal:** Width and capacity define different appliance configurations and prevent the listings from being the same product.

**Evidence pair id:** pair_0106

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0106", "source_row_index": 39076, "fold": 4, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 85899455689, "name": "electrolux / встраиваемая посудомоечная машина kegb9420w", "attributes": {"модель": "kegb9420w", "цвет": "черный", "тип управления": "электронное; сенсорное", "вместимость комплектов посуды": "15 комплектов", "страна производства": "италия", "высота предмета": "81.8 см", "ширина посудомоечной машины": "59,6 см", "глубина предмета": "55 см", "количество программ мойки": "8", "программы мойки посуды": "быстрая 30 мин, предварительное ополаскивание; 60 мин, 90 мин, 160 мин, auto sense, machine care; extrahygiene, extrapower, glasscare", "доп. опции посудомоечной машины": "функция airdry; сенсор чистоты воды; индикатор \"луч на полу\"", "тип сушки посудомоечной машины": "конденсационная", "тип посудомоечной машины": "полноразмерная", "класс энергопотребления": "класс c", "комплектация": "посудомоечная машина - 1 шт.", "вес без упаковки (кг)": "38.9 кг", "вес с упаковкой (кг)": "40.1 кг", "встраиваемая техника": "да", "гарантийный срок": "12 месяцев", "установка техники": "встраиваемая", "длина упаковки": "63 см", "высота упаковки": "67 см", "ширина упаковки": "89 см", "валюта": "rub"}, "formatted_text_length": 1052}, "item2": {"id": 429496819949, "name": "electrolux встраиваемая посудомоечная машина eea12100l", "attributes": {"бренд": "electrolux", "особенности посудомоечной машины": "полка для чашек", "глубина, см": "55", "отложенный старт, ч": "3", "класс сушки": "d", "класс энергопотребления": "f", "функционал посудомоечной машины": "использование средств 3-в-1", "высота, см": "81.8", "индикация": "наличие соли", "инверторный двигатель": "да", "управление": "электронное", "партномер": "eea12100l", "макс. уровень шума, дб": "49", "количество температурных режимов": "3", "комплектация": "посудомоечная машина, документация", "количество программ": "5", "системы защиты": "блокировка от детей", "вместимость, комплектов": "9", "размер (дхшхв), см": "55х44,6х81,8", "тип": "встраиваемая посудомоечная машина", "вид посудомоечной машины": "узкая (45 см)", "гарантийный срок": "12 месяцев", "класс мойки": "a", "расход воды за цикл, л": "9.9", "ширина, см": "44.6"}, "formatted_text_length": 870}}
```

## Pattern 5

**Pattern:** Two range hoods with different explicit manufacturer model codes are strict negatives when the codes also denote different width variants.

**Concrete difference:** The listings name Smeg KTR90XE and KTR110XE; the first explicitly has a width of 90 cm.

**Why this is a strict negative signal:** The differing model identifiers distinguish separate Smeg hood variants despite matching brand, finish, and headline performance figures.

**Evidence pair id:** pair_0108

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0108", "source_row_index": 288877, "fold": 2, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 678604871239, "name": "вытяжка настенная smeg ktr90xe серебристый", "attributes": {"бренд": "smeg", "модель": "ktr90xe", "гарантия": "зависит от условий продавца", "тип": "настенная", "вид конструкции": "купольная", "режим работы": "отвод/рециркуляция", "производительность, в м?/ч": "788", "мощность, в вт": "278", "уровень шума, в дб": "70", "дополнительные функции": "интенсивный режим", "материал корпуса": "нержавеющая сталь", "цвет": "серебристый", "цвет окантовки": "серебристый", "управление": "поворотное", "количество скоростей": "3", "дисплей": "нет", "фильтр": "жировой", "количество моторов": "1", "количество ламп освещения": "2", "тип ламп освещения": "светодиодная", "мощность лампы освещения, в вт": "2", "диаметр воздуховода, в см": "15", "вес, в кг": "16.5", "высота, в см": "90", "ширина, в см": "90", "глубина, в см": "48", "страна-производитель": "италия", "код товара": "100000026227"}, "formatted_text_length": 832}, "item2": {"id": 652835103662, "name": "настенная вытяжка smeg ktr110xe", "attributes": {"бренд": "smeg", "цвет товара": "серебристый", "страна-изготовитель": "италия", "вид фильтра вытяжки": "жировой", "партномер": "ktr110xe", "макс. уровень шума, дб": "50", "производительность отвода, куб.м/ч": "788", "мощность, вт": "278", "тип": "вытяжка", "установка вытяжки": "пристенная", "тип освещения вытяжки": "светодиодная лампа", "гарантийный срок": "24 мес."}, "formatted_text_length": 397}}
```

## Pattern 6

**Pattern:** Vacuum-bag consumables tied to different compatible vacuum brands/models are strict negatives, especially when pack contents differ.

**Concrete difference:** One listing is three bags for Zanussi ZAN2242, while the other is twelve bags plus two microfilters for Bestron K2000.

**Why this is a strict negative signal:** Fitment is decisive for replacement bags, and these listings state different compatible machines and unequal bundled quantities.

**Evidence pair id:** pair_0120

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0120", "source_row_index": 123661, "fold": 3, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 292057807914, "name": "не определен мешки синтетические 3 шт для пылесоса zanussi zan2242", "attributes": {"производитель совместимого пылесоса": "zanussi"}, "formatted_text_length": 134}, "item2": {"id": 618475392277, "name": "мешки синтетические 12 шт + 2 микрофильтра для пылесоса bestron k2000", "attributes": {"бренд": "нет бренда", "страна-изготовитель": "китай", "материал пылесборника": "синтетика", "комплектация": "мешки синтетические 12 шт + 2 микрофильтра", "количество в упаковке, шт": "12", "тип": "пылесборник", "гарантийный срок": "2 недели"}, "formatted_text_length": 319}}
```

## Pattern 7

**Pattern:** Consumable cartridge packs with unequal explicitly stated unit counts are strict negatives, even when they serve the same Braun cleaning-station ecosystem.

**Concrete difference:** The first Clean&Renew pack contains eight cartridges, while the second CCR4 listing contains four.

**Why this is a strict negative signal:** Pack quantity is part of the sold SKU and the human label marks the 8-pack and 4-pack as non-matching.

**Evidence pair id:** pair_0127

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0127", "source_row_index": 336866, "fold": 4, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 790273995249, "name": "braun / картриджи clean&renew ccr 8 для станций", "attributes": {"количество предметов в упаковке": "8", "совместимость с бритвами": "braun со станцией clean&charge", "страна производства": "ирландия", "вес товара с упаковкой (г)": "1.69 г", "длина упаковки": "20 см", "высота упаковки": "18 см", "ширина упаковки": "12 см", "валюта": "rub"}, "formatted_text_length": 328}, "item2": {"id": 489626294949, "name": "braun ccr4 картридж с чистящей жи", "attributes": {"бренд": "braun", "цвет товара": "голубой", "предназначено для": "очистки", "страна-изготовитель": "германия", "материал": "пластик", "партномер": "ccr4", "комплектация": "4 шт", "единиц в одном товаре": "4", "тип": "сменный картридж для очистки", "гарантийный срок": "1 год", "вес товара, г": "700"}, "formatted_text_length": 332}}
```

## Pattern 8

**Pattern:** Dryers with distinct model identifiers and materially different drying specification sets are strict negatives.

**Concrete difference:** Bosch WQG233DKPL has class A+++ and 14 drying programmes, whereas WQG235A0IT has class A++ and 10 programmes.

**Why this is a strict negative signal:** Model, energy class, programme count, and depth identify different dryer configurations rather than alternate listings for one unit.

**Evidence pair id:** pair_0145

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0145", "source_row_index": 210726, "fold": 2, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 489626401868, "name": "сушильная машина bosch wqg233dkpl", "attributes": {"цвет товара": "белый", "тип": "сушильная машина", "тип сушки": "конденсационная с тепловым насосом", "максимальная загрузка": "8 кг", "класс сушки": "a", "класс энергоэффективности": "a+++", "режимы / функции": "бережная сушка, быстрая сушка, полотенца, постельное бельё, проветривание, синтетика, сорочки (рубашки), удаление аллергенов, хлопок", "количество режимов сушки": "14", "ширина": "598 мм", "высота": "842 мм", "глубина": "648 мм", "вес": "53.7 кг", "установка": "отдельно стоящая", "тип загрузки": "фронтальная", "напряжение, в": "220-240", "уровень шума": "64 дб", "дополнительные функции": "контроль влажности, отложенный старт", "особенности конструкции": "дисплей", "сторона открывания двери": "слева", "материал бака": "нержавеющая сталь", "безопасность": "защита от детей", "гарантийный срок": "1 г."}, "formatted_text_length": 829}, "item2": {"id": 343597479186, "name": "сушильная машина bosch wqg235a0it", "attributes": {"бренд": "bosch", "цвет товара": "белый", "особенности конструкции": "тепловой насос", "страна-изготовитель": "турция", "глубина, см": "61.3", "отложенный старт, ч": "24", "наличие дисплея": "да", "класс энергопотребления": "a++", "функциональные особенности": "отсрочка запуска", "высота, см": "84.2", "вес, кг": "54.2", "инверторный двигатель": "нет", "загрузка белья для сушки, кг": "8", "управление": "электронное", "партномер": "wqg235a0it", "количество программ сушки": "10", "размеры, мм (шхгхв)": "598х613х842", "обработка паром": "да", "тип": "сушильная машина", "тип сушки": "конденсационная", "тип загрузки": "фронтальная", "гарантийный срок": "1 год", "ширина, см": "59.8"}, "formatted_text_length": 693}}
```

## Pattern 9

**Pattern:** Complete appliances from different brands with nonmatching explicit model names are strict negatives even when their broad type, colour, and main control scheme coincide.

**Concrete difference:** The pair compares the Hansa BHKW611301 gas panel with the Leran GH 64552 WG gas panel.

**Why this is a strict negative signal:** These are separate branded appliance models; shared generic features such as four gas burners and white glass do not establish product identity.

**Evidence pair id:** pair_0135

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0135", "source_row_index": 251626, "fold": 4, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 584115708711, "name": "hansa / варочная панель hansa bhkw611301 белый", "attributes": {"модель": "bhkw611301", "цвет": "белый", "управление": "механическое", "страна производства": "польша", "высота предмета": "5.26 см", "ширина предмета": "59 см", "глубина предмета": "52 см", "тип рабочей поверхности": "закаленное стекло", "тип варочной панели": "газовая", "число конфорок": "4", "доп. опции варочной поверхности": "газ-контроль; автоподжиг; дополнительные форсунки для баллонного газа", "тип конфорок": "газовые", "диаметр конфорок": "53 мм; 73 мм; 138 мм", "наличие рамки": "нет", "зона расширения электрических конфорок": "конфорка повышенной мощности (wok)", "решетки": "чугунные", "гарантийный срок": "1 год", "комплектация": "варочная панель", "вес с упаковкой (кг)": "14.45 кг", "вес без упаковки (кг)": "13.1 кг", "длина упаковки": "60 см", "высота упаковки": "13 см", "ширина упаковки": "70 см", "валюта": "rub"}, "formatted_text_length": 856}, "item2": {"id": 773094216506, "name": "leran газовая варочная панель gh 64552 wg, белый", "attributes": {"бренд": "leran", "цвет товара": "белый", "тип управления": "механическое", "глубина, см": "52", "вид электроподжига": "автоматический", "электроподжиг": "да", "цвет рамки варочной панели": "белый", "материал решетки варочной панели": "чугун", "тип панели (плиты)": "газовая", "минимальная толщина столешницы, мм": "49", "количество конфорок": "4", "количество конфорок с двойным рядом пламени": "1", "конструкция панели": "независимая", "мощность варочной панели, вт": "8800", "системы защиты": "газконтроль", "особые конфорки": "wok", "материал рабочей поверхности панели": "закаленное стекло", "тип": "варочная панель", "гарантийный срок": "12 месяцев", "ширина, см": "60"}, "formatted_text_length": 705}}
```

## Pattern 10

**Pattern:** A sectional aluminium heating radiator and a portable oil-filled electric heater are strict negatives despite both being described as radiators.

**Concrete difference:** The first listing is a six-section aluminium Optima radiator, while the second is a Resanta oil heater with 1500 W power and floor installation.

**Why this is a strict negative signal:** They are different heating-system product types with different construction and operating context.

**Evidence pair id:** pair_0149

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0149", "source_row_index": 263540, "fold": 1, "category": "Бытовая техника", "target": 0.0, "item1": {"id": 618475319413, "name": "/ радиатор алюминиевый optima 500/96/6", "attributes": {"материал радиатора": "алюминий", "количество секций": "6 шт.", "конструкция радиатора": "секционный", "тип установки": "универсальное", "гарантийный срок": "120", "высота предмета": "58 см", "ширина предмета": "80 см", "валюта": "rub"}, "formatted_text_length": 279}, "item2": {"id": 515396103687, "name": "масляный радиатор", "attributes": {"макс. площадь помещения, кв.м": "20", "бренд": "ресанта", "тип управления": "механическое", "страна-изготовитель": "китай", "размеры, мм": "360x130x650", "управление обогревателем": "поворотное", "установка": "напольная", "мощность, вт": "1500", "управление со смартфона": "нет", "тип": "обогреватель", "вид обогревателя": "масляный", "гарантийный срок": "1", "ширина, см": "135"}, "formatted_text_length": 393}}
```

## Annotation caveats

- `pair_0100` is labelled positive although one filter is stated for Melitta Caffeo Avanza and the other for Siemens TK68001/04; the first is called universal, so its compatibility claim is ambiguous.
- `pair_0131` is labelled negative although both names state the Edisson ES 30 V water heater; the shared model string makes this suspicious.
- `pair_0141` is labelled positive despite different Samsung vacuum-filter model strings, so compatible-parts model mismatch is not alone a universal negative rule in this sample.
- `pair_0148` is labelled positive although it compares Teka HLB 8510P (70 L, 59.5 cm high) with compact HLC 8400 (40 L, 45.5 cm high); treat appliance model and size differences in this category as potentially noisy unless independently corroborated.
- `pair_0116`, `pair_0124`, and `pair_0126` have sparse or near-identical descriptions with insufficient decisive detail and should not be generalized from.
