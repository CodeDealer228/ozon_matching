## Pattern 1

**Pattern:** A named toy series is a negative match when one item is explicitly a crane and the other is explicitly a toy car.

**Concrete difference:** The first item is a "кран 'нордик'", while the second is a "машинка нордпласт \"нордик\"".

**Why this is a strict negative signal:** The shared manufacturer and "Нордик" wording do not override the explicitly different toy types.

**Evidence pair id:** pair_0253

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0253", "source_row_index": 20214, "fold": 3, "category": "Детские товары", "target": 0.0, "item1": {"id": 42949742347, "name": "нордпласт / кран 'нордик', микс", "attributes": {"страна производства": "россия", "комплектация": "уточни у продавца", "валюта": "rub"}, "formatted_text_length": 132}, "item2": {"id": 240518311115, "name": "машинка нордпласт \"нордик\"", "attributes": {"бренд": "нордпласт", "тип": "машинка"}, "formatted_text_length": 82}}
```

## Pattern 2

**Pattern:** Feeding utensils alone are a negative match to a multi-item feeding-dish set that includes tableware and other feeding accessories.

**Concrete difference:** One item contains two feeding utensils, whereas the other is a nine-item set with utensils, a bib, a sippy cup, plates, lids, and a straw.

**Why this is a strict negative signal:** The stated contents establish different product configurations, not merely different listing text for the same set.

**Evidence pair id:** pair_0256

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0256", "source_row_index": 35643, "fold": 2, "category": "Детские товары", "target": 0.0, "item1": {"id": 77309527556, "name": "силиконовые приборы для кормления", "attributes": {"бренд": "tiny twinkle", "цвет товара": "серый", "максимальный возраст ребенка": "до 3 лет", "минимальный возраст ребенка": "6 месяцев", "пол ребенка": "унисекс", "количество предметов": "2", "тип": "приборы для кормления"}, "formatted_text_length": 263}, "item2": {"id": 463856493616, "name": "не определен силиконовый набор посуды для кормления", "attributes": {"цвет товара": "синий", "бренд": "khan premium", "тип": "комплект посуды", "количество предметов": "9", "комплектация": "ложка и вилка - 2 шт; нагрудник силиконовый - 1шт.; непроливающий поильник 1 шт крышка -1шт.; глубокая тарелка на присоске -1шт. тарелка с 3 отделениями на присосках -1шт.; крышка с трубочкой - 1шт.", "минимальный возраст": "4 мес.", "материал": "силикон", "особенности": "использование в свч, можно мыть в посудомоечной машине, можно стерилизовать, на присоске, нескользящее покрытие, тарелка с секциями", "диаметр тарелки": "20 см", "объем": "300 мл", "дополнительная информация": "детский силиконовый набор посуды для кормления малыша"}, "formatted_text_length": 710}}
```

## Pattern 3

**Pattern:** Philips Avent pacifiers with different exact model suffixes and non-overlapping stated age bands are negative matches.

**Concrete difference:** The pair contrasts Ultra Air SCF080/06 for 0–6 months with SCF080/08 for 6–18 months.

**Why this is a strict negative signal:** Both the model code and the intended age range identify separate pacifier variants despite the same brand, line, material, and two-piece case configuration.

**Evidence pair id:** pair_0257

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0257", "source_row_index": 247527, "fold": 3, "category": "Детские товары", "target": 0.0, "item1": {"id": 575525751844, "name": "пустышка силиконовая philips avent ultra air scf080/06 с футляром для стерилизации 0-6 мес, 2 шт", "attributes": {"артикул": "ot2802651", "бренд": "philips avent", "цвет товара": "фиолетовый", "страна-изготовитель": "нидерланды", "особенности бутылочек и сосок": "для новорожденных", "максимальный возраст ребенка": "6 месяцев", "минимальный возраст ребенка": "0 месяцев", "форма бутылочки/соски": "ортодонтическая", "комплектация": "2 пустышки, футляр для хранения и стерилизации", "пол ребенка": "унисекс", "тип": "пустышка"}, "formatted_text_length": 508}, "item2": {"id": 386547158096, "name": "philips / соска-пустышка ultra air scf080/08 с футляром, 2шт, 6-18 мес", "attributes": {"цвет": "фиолетовый, фуксия", "материал наконечника/пустышки/соски": "силикон", "тип пустышки": "ортодонтическая", "модель пустышки": "scf080/08", "особенности пустышки": "ортодонтическая пустышка; без бисфенола а; с кольцом-держателем", "возрастная категория": "6-18 мес", "количество предметов в упаковке": "2 шт", "вес товара с упаковкой (г)": "48 г", "страна производства": "нидерланды", "комплектация": "пустышка - 2 шт; футляр для стерилизации и хранения - 1 шт; инструкция - 1 шт", "валюта": "rub"}, "formatted_text_length": 574}}
```

## Pattern 4

**Pattern:** Bicycles in the same wheel-size line are negative matches when the text specifies different named variants and model years.

**Concrete difference:** One bicycle is "maxiscoo air 18 pro 2024" in black, while the other is the "air 18" standard 2023 in gray.

**Why this is a strict negative signal:** "Pro 2024" and "standard 2023" are explicit variant/version identifiers; the shared Air 18 wording is insufficient for identity.

**Evidence pair id:** pair_0259

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0259", "source_row_index": 241102, "fold": 4, "category": "Детские товары", "target": 0.0, "item1": {"id": 566935699851, "name": "maxiscoo air 18 pro 2024 черный матов.", "attributes": {"цвет товара": "черный матов."}, "formatted_text_length": 88}, "item2": {"id": 558345854751, "name": "велосипед детский maxiscoo air 18\" стандарт 2023 серый матовый", "attributes": {"форма поставки": "в разобранном виде", "привод велосипеда": "цепной", "передний тормоз": "клещевой", "задний тормоз": "роллерный", "бренд": "maxiscoo", "цвет товара": "серый", "название цвета": "серый матовый", "страна-изготовитель": "китай", "конструкция вилки": "жесткая", "материал рамы": "магниевый сплав", "размер рамы, дюймы": "универсальный", "количество скоростей велосипеда": "1", "макс. нагрузка, кг": "50", "минимальный возраст ребенка": "от 4 лет", "модельный год": "2023", "количество колес": "2", "комплектация": "велосипед - 1 шт.", "тип педалей": "классический", "вид обода": "одинарный", "конструкция руля": "изогнутый", "материал седла": "искусственная кожа", "регулировка сидения": "да", "пол ребенка": "унисекс", "боковые колеса в комплекте": "да", "конструкция рулевой колонки": "резьбовая", "регулировка руля": "да", "амортизация велосипеда": "rigid (жесткий)", "тип": "детский велосипед", "вид колес": "надувные", "вид детского велосипеда": "двухколесный", "вес в собранном состоянии, кг": "10", "диаметр колес, дюймы": "18"}, "formatted_text_length": 1068}}
```

## Pattern 5

**Pattern:** Scale toy cars are negative matches when their explicitly named real-vehicle models differ.

**Concrete difference:** The first model is a Chevy Nomad and the second is a Lada, even though both are orange inertial cars with light and sound.

**Why this is a strict negative signal:** The real-vehicle model is a decisive identity field for a toy-car model; common mechanism and effects do not make a Chevy Nomad a Lada.

**Evidence pair id:** pair_0261

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0261", "source_row_index": 193129, "fold": 2, "category": "Детские товары", "target": 0.0, "item1": {"id": 455266536330, "name": "модель машины 1:36 chevy nomad свет, звук, инерция оранжевый", "attributes": {"бренд": "mx", "тип": "машинка"}, "formatted_text_length": 109}, "item2": {"id": 386547161952, "name": "r&ro / машинка lada инерционная 11,5см, звук и свет", "attributes": {"цвет": "оранжевый", "материал игрушки": "металл; пластик", "особенности игрушки": "открывающиеся части", "марка автомобиля": "ваз; lada", "механизм движения": "инерционный", "эффекты": "звуковые; световые", "количество предметов в упаковке": "1 шт.", "высота предмета": "4 см", "ширина предмета": "4.8 см", "страна производства": "китай", "валюта": "rub"}, "formatted_text_length": 406}}
```

## Pattern 6

**Pattern:** A model vehicle with the same named UAZ 3962 fire-service subject is a negative match when the stated scale differs.

**Concrete difference:** One UAZ 3962 fire-service model is scale 1:32 and the other is scale 1:43.

**Why this is a strict negative signal:** Scale determines the physical model variant even when brand, subject vehicle, service theme, color, and inertial mechanism align.

**Evidence pair id:** pair_0271

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0271", "source_row_index": 94682, "fold": 2, "category": "Детские товары", "target": 0.0, "item1": {"id": 214748509372, "name": "автоград, машина металлическая \"уаз 3962. пожарная служба\", инерция, 1:32, свет и звук", "attributes": {"типоразмер батареек": "lr54 (lr1130, v10ga, ag10, g10, rw49)", "количество батареек": "3", "бренд": "автоград", "марка автомобиля": "уаз", "вид транспорта": "масштабные модели", "цвет товара": "красный", "особенности машинок": "звуковые эффекты", "страна-изготовитель": "китай", "длина, см": "12", "материал": "металл", "масштаб модели": "1/32", "вид службы": "пожарная", "тип": "машинка", "количество элементов, шт": "1"}, "formatted_text_length": 502}, "item2": {"id": 790274010841, "name": "автоград машина металлическая «уаз 3962. пожарная служба», инерция, 1:43", "attributes": {"артикул производителя": "9351059", "бренд": "автоград", "возрастное ограничение": "3+", "набор": "нет", "тип": "легковая машина", "вид": "транспорт спецслужб", "тематика": "пожарная служба", "марка": "уаз", "материал": "металл; пластик", "цвет": "красный", "работает от электропитания": "нет", "количество машинок": "1", "трансформер": "нет", "инерционность": "да", "особенности": "открываются двери", "масштаб": "1/43", "код товара": "100056874481"}, "formatted_text_length": 510}}
```

## Pattern 7

**Pattern:** A child car seat and a base for that car-seat line are negative matches because one is the restraint and the other is its installation accessory.

**Concrete difference:** The first item is a Britax Römer Baby-Safe Core infant car seat, while the second is the Baby-Safe Core Base with Isofix and a support leg.

**Why this is a strict negative signal:** Compatibility or shared line naming does not make an accessory base the car seat itself.

**Evidence pair id:** pair_0274

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0274", "source_row_index": 288711, "fold": 4, "category": "Детские товары", "target": 0.0, "item1": {"id": 678604864539, "name": "автокресло britax roemer baby-safe core midnight grey", "attributes": {"тип автокресла": "автолюлька", "группа": "0 (0 -13 кг)", "крепление": "система isofix", "установка": "против хода движения", "для авиаперелётов": "нет", "длина, см": "64", "ширина, см": "44", "высота, см": "61", "вес, кг": "3.9", "длина упаковки, см": "77.0", "ширина упаковки, см": "45.0", "высота упаковки, см": "32.0", "вес упаковки, кг": "5.0", "гарантийный срок": "3 года", "страна производства": "китай"}, "formatted_text_length": 455}, "item2": {"id": 755914350445, "name": "база для автокресла britax roemer baby-safe core base", "attributes": {"рекомендуемый возраст": "от 0 до 1 года", "группа": "0+", "артикул": "2000038436", "бренд": "britax roemer", "вид аксессуара для автокресла": "база для автокресла", "цвет товара": "черный", "название цвета": "чёрный", "страна-изготовитель": "китай", "стандарты безопасности": "стандарт i-size (ece r129)", "материал": "пластик", "внешние размеры, см": "62х32х16", "комплектация": "база с изофикс и опорной ногой", "упаковка": "картонная коробка", "пол ребенка": "унисекс", "размер упаковки (длина х ширина х высота), см": "66х35х19", "тип": "аксессуар для автокресла", "гарантийный срок": "2 года", "вес товара, г": "5200"}, "formatted_text_length": 662}}
```

## Pattern 8

**Pattern:** Diaper pants with the same brand, line, size, and weight range are negative matches when their declared package counts differ.

**Concrete difference:** The two Pampers Premium Care size-4, 9–15 kg listings declare 76 and 38 diaper pants respectively.

**Why this is a strict negative signal:** The package quantity is an explicit sellable-unit distinction, while the other named product identifiers are held constant.

**Evidence pair id:** pair_0276

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0276", "source_row_index": 41342, "fold": 4, "category": "Детские товары", "target": 0.0, "item1": {"id": 94489327788, "name": "pampers / подгузники-трусики premium care, размер 4, 9-15 кг, 76 штук", "attributes": {"тип подгузников": "подгузники-трусики", "весовая группа": "9-15 кг", "количество предметов в упаковке": "76 шт.", "вес товара с упаковкой (г)": "2698 г", "страна производства": "россия", "комплектация": "подгузники - 76 шт", "длина упаковки": "44 см", "высота упаковки": "29 см", "ширина упаковки": "25 см", "валюта": "rub"}, "formatted_text_length": 395}, "item2": {"id": 137439069614, "name": "подгузники трусики pampers premium care 4 размер 9-15 кг 38 шт", "attributes": {"бренд": "pampers", "страна-изготовитель": "россия", "максимальный возраст ребенка": "до 3 лет", "минимальный возраст ребенка": "1 месяц", "количество подгузников в одной упаковке": "38", "единиц в одном товаре": "38", "пол ребенка": "унисекс", "тип": "подгузники-трусики"}, "formatted_text_length": 340}}
```

## Pattern 9

**Pattern:** Electronic keyboard toys with different stated key counts are negative matches, even when the brand and broad product type match.

**Concrete difference:** The first Bondibon synthesizer has 61 keys with a USB cable, whereas the second has 49 keys and a power supply.

**Why this is a strict negative signal:** The number of keys is a concrete product-specification difference that identifies a different instrument configuration.

**Evidence pair id:** pair_0281

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0281", "source_row_index": 315803, "fold": 3, "category": "Детские товары", "target": 0.0, "item1": {"id": 738734416506, "name": "bondibon инструм. муз. на батар, синтезатор клавишник bondibon, 61 клавиша, с микрофоном и usb-шнуром, стере вв4949", "attributes": {"инструмент": "пианино", "минимальный возраст": "5 лет", "материал": "пластик", "питание": "от usb, от батареек, от сети", "тип батареек": "aa", "в комплекте": "usb-кабель", "количество мелодий": "10", "особенности": "звуковые эффекты, регулировка громкости, функция записи"}, "formatted_text_length": 395}, "item2": {"id": 137439033134, "name": "инструм. муз. на батар., синтезатор клавишник bondibon, 49 клавиш, с микрофоном и блоком питания,сте", "attributes": {"бренд": "bondibon", "режимы": "запись", "максимальный возраст ребенка": "до 10 лет", "минимальный возраст ребенка": "от 1 года", "вид детского музыкального инструмента": "синтезатор", "вид питания": "от сети 220в", "пол ребенка": "унисекс", "тип": "детский музыкальный инструмент"}, "formatted_text_length": 387}}
```

## Pattern 10

**Pattern:** A character soft toy and the same named character explicitly sold in a costume edition are negative matches when the costume is present only in one item.

**Concrete difference:** Both items name the 21 cm Pomposhki "единорожка", but the second is explicitly "в пижамке".

**Why this is a strict negative signal:** The costume wording denotes a separately specified edition rather than an omitted generic descriptive field.

**Evidence pair id:** pair_0287

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0287", "source_row_index": 117427, "fold": 4, "category": "Детские товары", "target": 0.0, "item1": {"id": 274877970792, "name": "мягкая игрушка pomposhki \"единорожка\", 21 см", "attributes": {"бренд": "pomposhki", "страна-изготовитель": "китай", "материал": "полипропилен", "минимальный возраст ребенка": "от 3 лет", "пол ребенка": "унисекс", "тип": "мягкая игрушка"}, "formatted_text_length": 228}, "item2": {"id": 335007488902, "name": "мягкая игрушка pomposhki \"единорожка\", в пижамке, 21 см", "attributes": {"высота/длина": "21 см", "цвет товара": "разноцветный", "тип": "мягкая игрушка", "минимальный возраст": "3 лет", "габариты транспортной упаковки": "15.50х31.50х11.50 см", "вес в транспортной упаковке": "0.219 кг"}, "formatted_text_length": 277}}
```

## Pattern 11

**Pattern:** Educational sets that share a title phrase are negative matches when one is explicitly a stamp-based "логические задания" set and the other is a fuzzy-wire set.

**Concrete difference:** The first item is described as a reusable logical-tasks set with stamps, while the second is named as a fuzzy-wire logical-tasks set.

**Why this is a strict negative signal:** The stated activity materials and mechanism identify different set contents; the overlap on the brand-like word and title fragment is not sufficient.

**Evidence pair id:** pair_0290

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0290", "source_row_index": 224747, "fold": 3, "category": "Детские товары", "target": 0.0, "item1": {"id": 523986120342, "name": "iq-zabiaka / развивающий набор «логические задания» со штампами, многораз", "attributes": {"страна производства": "китай", "комплектация": "уточни у продавца", "валюта": "rub"}, "formatted_text_length": 173}, "item2": {"id": 429496782870, "name": "развивающий набор пушистая проволока zabiaka \"логические задания\"", "attributes": {"бренд": "zabiaka", "материал": "картон", "максимальный возраст ребенка": "до 6 лет", "минимальный возраст ребенка": "от 3 лет", "пол ребенка": "унисекс", "тип": "обучающая игра", "развитие навыков": "внимание"}, "formatted_text_length": 283}}
```

## Pattern 12

**Pattern:** Soft toys of the same named animal are negative matches when their stated heights are materially different.

**Concrete difference:** One "енотик" soft toy is 15 cm and the other is 37 cm.

**Why this is a strict negative signal:** The explicitly stated height differentiates the physical toy variant while the product type and animal wording are otherwise aligned.

**Evidence pair id:** pair_0294

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0294", "source_row_index": 365395, "fold": 4, "category": "Детские товары", "target": 0.0, "item1": {"id": 850403672486, "name": "мягкая игрушка енотик , 15 см", "attributes": {"бренд": "нет бренда", "страна-изготовитель": "китай", "высота игрушки, см": "15", "материал": "искусственный мех", "минимальный возраст ребенка": "от 3 лет", "пол ребенка": "унисекс", "вид мягкой игрушки": "традиционные", "тип": "мягкая игрушка"}, "formatted_text_length": 281}, "item2": {"id": 541165920447, "name": "не определен мягкая игрушка енотик 37 см", "attributes": {"высота/длина": "37 см", "цвет товара": "серый", "тип": "мягкая игрушка", "минимальный возраст": "3 лет", "материал": "текстиль", "наполнитель": "синтепон", "дополнительная информация": "плюшевый и мягкий", "габариты транспортной упаковки": "22х13х26 см", "вес в транспортной упаковке": "0.2 кг"}, "formatted_text_length": 339}}
```

## Annotation caveats

- Several negative pairs appear near-duplicate or under-specified and should not be generalized into negative rules: pair_0250, pair_0252, pair_0255, pair_0260, pair_0262, pair_0269, pair_0270, pair_0272, pair_0273, pair_0288, and pair_0292.
- pair_0279 is internally suspicious: item1 is named as a soft toy but its `тип` is "одежда для мягкой игрушки".
- The positive counterexamples show noisy or inconsistent attributes: pair_0251 names diaper sizes 4 and 5 and reports 80 in both titles but item2 attributes say 54; pair_0284 has different stated colors; pair_0293 has different named brands. Therefore brand alone, a loosely shared title, or a single inconsistent auxiliary attribute is not used as a negative rule.
