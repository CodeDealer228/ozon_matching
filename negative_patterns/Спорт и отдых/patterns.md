## Pattern 1

**Pattern:** For an otherwise comparable fitness resistance band, an explicit different maximum load is a negative signal.

**Concrete difference:** The Onlitop bands are specified for 30 kg versus 55 kg maximum load, and their listed thicknesses also differ.

**Why this is a strict negative signal:** Resistance level is the defining purchasable variant of this band; shared brand and exercise purpose do not make distinct loads interchangeable.

**Evidence pair id:** pair_0701

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0701", "source_row_index": 188334, "fold": 4, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 438086772198, "name": "onlitop, фитнес-резинка, 30х5х0,15 см, нагрузка 30 кг, цвет фиолетовый", "attributes": {"целевая аудитория": "взрослая", "бренд": "onlitop", "цвет товара": "фиолетовый", "страна-изготовитель": "китай", "особенности фитнес-инвентаря": "эспандер- лента", "длина, см": "30", "материал": "латекс", "макс. нагрузка, кг": "30", "тренируемые группы мышц": "ноги", "количество в упаковке, шт": "1", "комплектация": "фитнес-резинка, 30х5х0,15 см, нагрузка 30 кг, цвет фиолетовый, 1 штука", "тип": "фитнес-резинка", "вес, г": "40", "ширина, см": "5"}, "formatted_text_length": 515}, "item2": {"id": 532576068891, "name": "фитнес-резинка onlitop 30х3.2х0.5 см, нагрузка 55 кг, цвет черный 3551210", "attributes": {"тип": "ленточный", "назначение": "для ног", "максимальная нагрузка": "55 кг", "материал": "резина", "цвет": "черный", "палитра": "черный", "количество в упаковке": "1 шт", "вес нетто": "0.09 кг", "габариты без упаковки": "300х32х5 мм", "единица товара": "штука", "вес, кг": "0,09", "длина, мм": "290", "ширина, мм": "145", "высота, мм": "20"}, "formatted_text_length": 409}}
```

## Pattern 2

**Pattern:** Outdoor tarps with explicitly different finished side dimensions are negative candidates even if their use and colour description overlap.

**Concrete difference:** One tarp is 500 by 500, while the other is 800 by 290 cm, with polyester versus Oxford stated as the tent material.

**Why this is a strict negative signal:** The cut dimensions determine the physical product and cannot be reconciled by the shared universal-tarp use or the “морская волна” colour description.

**Evidence pair id:** pair_0702

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0702", "source_row_index": 122341, "fold": 2, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 283467967762, "name": "не определен тент туристический кемпинговый, садовый, пляжный/навес для автомобиля", "attributes": {"цвет товара": "морская волна", "площадь": "25 м³", "длина": "500", "ширина": "500", "тип": "тент", "назначение": "универсальный", "форма": "прямоугольная", "особенности": "уф-защита, быстросборный, водонепроницаемость", "тип каркаса": "отсутствует", "комплектация": "люверсы", "материал тента": "полиэстер", "габариты транспортной упаковки": "55х55х12 см", "вес в транспортной упаковке": "5.5 кг"}, "formatted_text_length": 475}, "item2": {"id": 781684082475, "name": "тент универсальный оксфорд для туризма, отдыха, дачи 2,9х8", "attributes": {"материал тента": "оксфорд", "бренд": "нет бренда", "тип каркаса": "отсутствует", "цвет товара": "синий", "название цвета": "морская волна", "страна-изготовитель": "россия", "плотность, г/м2": "220", "длинная сторона, см": "800", "комплектация": "люверсы", "сезон": "демисезонный", "короткая сторона, см": "290", "водостойкость тента, мм в.ст.": "от 1000 до 3000"}, "formatted_text_length": 419}}
```

## Pattern 3

**Pattern:** Monofilament line with different stated diameter and breaking load is a negative product variant, especially when the pack count also differs.

**Concrete difference:** The Momoi line is 0.30 mm and 10 kg in a three-piece set versus 0.22 mm and 5.5 kg as one piece.

**Why this is a strict negative signal:** Diameter, tensile rating, and quantity are explicit sellable specifications for fishing line, not merely incomplete catalog metadata.

**Evidence pair id:** pair_0708

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0708", "source_row_index": 173956, "fold": 4, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 403727031590, "name": "монофильная леска для рыбалки momoi spinning line silver 0,30 мм, 10,0 кг, 100 м, серебряная, 3 штуки", "attributes": {"сезон": "лето", "бренд": "momoi", "разрывная нагрузка, кг": "10", "цвет товара": "серебристый", "страна-изготовитель": "япония", "размотка, м": "300", "количество нитей": "1", "упаковка": "картонная коробка", "кол-во в комплекте, шт": "3", "толщина, мм": "0.3", "тип": "монофильная леска для рыбалки"}, "formatted_text_length": 402}, "item2": {"id": 360777356900, "name": "монофильная леска для рыбалки momoi spinning line silver 0,22 мм, 5,5 кг, 100 м, серебряная, 1 штука", "attributes": {"количество": "1 шт.", "длина": "100", "диаметр": "0.22", "разрывная нагрузка": "5.5", "цвет товара": "silver, (1 штука)", "тип": "монофильная леска", "сезон": "внесезонный", "количество нитей": "1 шт.", "дополнительная информация": "вид ловли: спиннинговая, ночная", "количество в комплекте": "1"}, "formatted_text_length": 399}}
```

## Pattern 4

**Pattern:** Different explicit manufacturer model codes and physical length distinguish manual pumps within the same brand.

**Concrete difference:** The Intex pump is model 68614 at 36 cm, whereas the other is 68615 at 48 cm.

**Why this is a strict negative signal:** A model-code change accompanied by a 12 cm dimensional change identifies separate purchasable pump models.

**Evidence pair id:** pair_0711

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0711", "source_row_index": 145223, "fold": 1, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 343597398536, "name": "насос ручной 36см intex 68614 (1123607574)", "attributes": {"бренд": "intex", "цвет товара": "черный", "страна-изготовитель": "китай", "количество насадок, шт": "3", "тип": "насос для матраса", "вид насоса": "ручной"}, "formatted_text_length": 208}, "item2": {"id": 137438969197, "name": "intex / насос ручной intex 68615, 48 см", "attributes": {"тип насоса": "ручной", "материал корпуса": "пластик", "высота предмета": "48 см", "страна производства": "китай", "комплектация": "насос ручной - 1шт.", "вес с упаковкой (кг)": "1.15 кг", "валюта": "rub", "цвет": "черный"}, "formatted_text_length": 267}}
```

## Pattern 5

**Pattern:** Protective knee pads with explicitly different size and wearer segment are negative variants.

**Concrete difference:** One listing is size XS for adults, while the other is size M and explicitly describes the age group as children.

**Why this is a strict negative signal:** Size and intended wearer determine fit; common gymnastics/dance use and two-piece packaging are insufficient to establish identity.

**Evidence pair id:** pair_0718

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0718", "source_row_index": 228868, "fold": 4, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 532576076945, "name": "lb наколенники для танцев, размер: xs,", "attributes": {"целевая аудитория": "взрослая", "бренд": "lb", "цвет товара": "черный", "название цвета": "black", "страна-изготовитель": "китай", "высота товара в упаковке, мм": "270", "длина товара в упаковке, мм": "200", "материал": "трикотаж", "количество заводских упаковок": "1", "комплектация": "2", "размер": "xs", "вид спорта": "гимнастика", "тип": "наколенники для танцев", "вес товара, г": "150"}, "formatted_text_length": 423}, "item2": {"id": 77309505251, "name": "не определен спортивные наколенники для танцев и гимнастики, фитнеса", "attributes": {"размер": "m", "цвет товара": "серый", "тип": "наколенники для гимнастики", "вид защиты": "наколенники", "вид спорта": "гимнастика, танцы, легкая атлетика, плавание, фитнес", "возрастная группа": "детские (yth)", "подробная комплектация": "наколенники 2 шт", "дополнительная информация": "подробная комплектация: наколенники 2 шт", "габариты транспортной упаковки": "18х24х6 см", "вес в транспортной упаковке": "0.15 кг"}, "formatted_text_length": 490}}
```

## Pattern 6

**Pattern:** Ice augers with different stated drilling diameter are distinct variants despite a shared brand, rotation side, and product type.

**Concrete difference:** The Tonar augers have 130 mm versus 100 mm drilling diameter.

**Why this is a strict negative signal:** Bore diameter is the core operating size of an ice auger and is separately encoded in the models LS-130L and LS-100L1.KS.

**Evidence pair id:** pair_0722

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0722", "source_row_index": 232017, "fold": 4, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 541165988165, "name": "ледобур лр-130 спорт левое вращение (ls-130l) тонар", "attributes": {"длина в сложенном положении": "1173 мм", "вес": "3.1 кг", "максимальная глубина бурения": "1141 мм", "длина шнека": "778 мм", "диаметр шнека/бурения": "130 мм", "сторона вращения": "левая", "тип": "ледобур", "материал": "пластик", "тип ножей": "сферический", "дополнительная информация": "материал ручки: пластик"}, "formatted_text_length": 367}, "item2": {"id": 489626389767, "name": "ледобур лр-100скс сотка казанцева спорт левое вращение (ls-100l1.ks) тонар", "attributes": {"запасные ножи в комплекте": "нет", "бренд": "тонар", "длина шнека, см": "77", "глубина бурения, см": "113", "диаметр бурения, мм": "100", "направление вращения": "против часовой стрелки", "тип": "ледобур", "вид ледобура": "ручной"}, "formatted_text_length": 311}}
```

## Pattern 7

**Pattern:** Whey protein with different explicit flavour variants is negative when the flavour is part of the selling name.

**Concrete difference:** The 500 g RPS whey listings are “орехи в шоколаде” and “карамель.”

**Why this is a strict negative signal:** Flavour is a deliberate consumer-facing variant for ingestible sports nutrition, even when protein type and net weight match.

**Evidence pair id:** pair_0728

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0728", "source_row_index": 240573, "fold": 3, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 558345916519, "name": "сывороточный протеин rps whey protein, 500 гр. орехи в шоколаде", "attributes": {"бренд": "rps", "страна-изготовитель": "россия", "срок годности в днях": "720", "состав": "содержание белка в сывороточном концентрате достигает 80%! данное свойство позволяет качественно наращивать мышечный вес, особенно эффективность наблюдается при приеме перед и после физических упражнений.\nдействие whey protein усиливается благодаря обогащению l-глютамином. ведь согласно ученым, 60% в мышцах среди аминокислот – это глютамин. эта аминокислота способствует удержанию катаболических процессов, укрепляет иммунитет, и активизирует гормон, ответственный за рост мышечных тканей. к тому же, разработанная формула исключает любое содержание сахаров, что делает добавку еще более полезной.", "вкусовой акцент (вкус)": "орех", "тип": "протеин"}, "formatted_text_length": 817}, "item2": {"id": 670015021140, "name": "rps nutrition сывороточный протеин rps nutrition, пакет 500гр, карамель", "attributes": {"вкус": "карамель", "количество порций": "16", "количество упаковок": "1", "упаковка": "мягкая упаковка", "вес": "500 г", "тип": "сывороточный концентрат", "форма выпуска": "порошок", "энергетическая ценность в 100 г": "энергетическая ценность - 382 ккал, белки - 72.5 г, жиры - 3.5 г, углеводы - 14.6 г.", "содержание белков в 100 г": "72.5", "дополнительная информация": "состав на порцию 30 г: энергетическая ценность - 115 ккал, белки - 21.8 г, жиры - 1 г, углеводы - 4.4 г. ингредиентный состав: концентрат сывороточного белка, эмульгатор (микрокристаллическая целлюлоза), ванилин кристаллический, загустители (ксантановая и гуаровая камеди), ароматизатор пищевой \"тоффи\", соль поваренная пищевая, краситель пищевой натуральный кармин, сукралоза (подсластитель)."}, "formatted_text_length": 843}}
```

## Pattern 8

**Pattern:** Fishing reels that share only a spool size but name different model lines are negative candidates when their component specifications also diverge.

**Concrete difference:** Kaida Gallant 5000 is paired with Kaida Differ 5000, with 5+1 versus 3+1 bearings and 5.1:1 versus 5.2:1 gearing.

**Why this is a strict negative signal:** “5000” identifies a size class, whereas Gallant and Differ are distinct model names corroborated by different mechanical specifications.

**Evidence pair id:** pair_0737

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0737", "source_row_index": 47994, "fold": 3, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 111669172617, "name": "катушка kaida gallant 5000", "attributes": {"рука": "универсальная", "количество шарикоподшипников": "5+1 шт.", "размер": "5000", "передаточное число": "5.1:1", "лесоёмкость": "0, 235/295 - 0, 285/200 - 0, 330/150 мм/м мм/м", "вес": "395 г", "тип": "катушка", "тип катушки": "безынерционная", "вид ловли": "карповая, спиннинговая, фидерная", "фрикционный тормоз": "передний", "нагрузка на фрикцион": "8 кг", "особенности": "металлическая шпуля, с двух сторон"}, "formatted_text_length": 439}, "item2": {"id": 515396086974, "name": "катушка kaida differ 5000", "attributes": {"сезон": "весна", "бренд": "kaida", "тормозная система катушки": "фрикционный", "реверс катушки": "да", "цвет товара": "красный", "название цвета": "темно-серый", "страна-изготовитель": "китай", "особенности снасти": "алюминиевая шпуля", "лескоёмкость": "0,35/310, 0,4/240, 0,45/190", "передаточное число катушки": "5.2:1", "ручка катушки": "с двух сторон", "вид ловли": "фидерная", "материал": "пластик", "единиц в одном товаре": "1", "количество подшипников": "3+1", "фрикцион": "передний", "вид катушки": "безынерционная", "диаметр катушки, мм": "55", "размер шпули": "5000", "намотка за оборот ручки, см": "86", "тип": "катушка", "вес товара, г": "379"}, "formatted_text_length": 659}}
```

## Pattern 9

**Pattern:** Air rifles with the same brand, caliber, and energy limit are negative if their operating mechanism and overall length differ.

**Concrete difference:** The Hatsan listings state electropneumatic AEG and 105 cm versus spring-piston and 1240 mm.

**Why this is a strict negative signal:** The power mechanism and total length are fundamental configuration properties, so matching brand and 4.5 mm caliber are insufficient.

**Evidence pair id:** pair_0740

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0740", "source_row_index": 314286, "fold": 1, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 730144590494, "name": "винтовка пневматическая hatsan alpha 3 дж. кал. 4,5мм (переломка, пластик)", "attributes": {"количество в упаковке": "1 шт.", "цвет товара": "черный", "тип": "пневматическая винтовка", "дульная энергия": "3 дж", "назначение": "для охоты, для спортивной стрельбы, для тренировки", "принцип действия": "электропневматический (aeg)", "тип снаряда": "пули", "калибр": "4,5 мм", "вместимость": "2", "вид ствола": "нарезной", "длина ствола": "450 мм", "предохранитель": "автоматический, флажковый", "особенности": "морозоустойчивость, под левую руку, под правую руку", "вид крепления": "для прицела", "назначение адаптера": "для приклада", "вид опоры": "мягкая", "вид мушки": "закрытая, открытая, светящаяся", "материал": "пластик", "вес": "2.9 кг", "длина": "105 см", "ширина": "4 см", "высота": "10 см", "подробная комплектация": "винтовка, инструкция, коробка", "дополнительная информация": "пневматическая винтовка"}, "formatted_text_length": 869}, "item2": {"id": 472446412661, "name": "hatsan пневматическая винтовка, до 3 дж, 1240 мм", "attributes": {"боеприпасы": "пули", "бренд": "hatsan", "калибр оружия (мм)": "4,5", "цвет товара": "черный матовый", "страна-изготовитель": "турция", "источник энергии": "пружина", "принцип действия оружия": "пружинно-поршневой", "особенности оружия": "крепление для прицела", "общая длина, мм": "1240", "дульная энергия": "до 3 дж", "материал винтовки": "металл", "приклад": "постоянный", "режим стрельбы": "одиночный", "скорость выстрела, м/с": "120", "ёмкость магазина, шт": "1", "тип": "пневматическая винтовка", "тип ствола": "нарезной", "вес, г": "3650"}, "formatted_text_length": 579}}
```

## Pattern 10

**Pattern:** Parts for the same scooter model are negative when the component type is explicitly different.

**Concrete difference:** For Kugoo M4, one listing is a two-piece tyre set and the other is a rear fork.

**Why this is a strict negative signal:** Compatibility with the same vehicle model does not make a tyre and a structural fork the same replacement part.

**Evidence pair id:** pair_0741

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0741", "source_row_index": 285699, "fold": 1, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 670014965222, "name": "electromiro / комплект 1+1. покрышка для kugoo m4, 2 штуки", "attributes": {"комплектация": "комплект 1+1. покрышка для kugoo m4 - 2 штуки", "страна производства": "китай", "валюта": "rub"}, "formatted_text_length": 186}, "item2": {"id": 506806235172, "name": "задняя вилка для электросамоката kugoo m4/m4pro", "attributes": {"бренд": "kugoo-original", "цвет товара": "черный", "страна-изготовитель": "китай", "количество в одной упаковке, шт": "1", "единиц в одном товаре": "1", "вид роликов, самокатов": "электрический самокат", "модель самоката": "kugoo m4", "тип": "запчасти для тюнинга электросамокатов"}, "formatted_text_length": 335}}
```

## Pattern 11

**Pattern:** Fixed-weight dumbbells are negative variants when their explicitly stated single-dumbbell weight differs.

**Concrete difference:** The Torres dumbbells are listed as 5 kg versus 1.5 kg each.

**Why this is a strict negative signal:** Weight is the primary performance specification for a non-adjustable dumbbell; shared brand, one-unit count, and neoprene grip do not override it.

**Evidence pair id:** pair_0747

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0747", "source_row_index": 178710, "fold": 2, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 420906801568, "name": "torres гантели pl55015, 1 шт., черный цвет", "attributes": {"бренд": "torres", "цвет товара": "черный", "конструкция гантель": "неразборные", "покрытие диска": "без покрытия", "форма дисков": "шестиугольная", "форма грифа": "прямой", "вес грифа, кг": "5", "кол-во гантелей, шт": "1", "покрытие грифа": "неопреновая", "тип": "гантели", "вес, кг": "5"}, "formatted_text_length": 331}, "item2": {"id": 592705493042, "name": "неразборная гантель неопреновая torres pl5501, 1 x 1,5 кг, серый", "attributes": {"артикул производителя": "pl550115-1-5kg", "модель": "pl5501-1", "бренд": "torres", "страна-производитель": "китай", "тип": "неразборная гантель", "количество гантелей в наборе": "1", "вес одной гантели, кг": "1.5", "общий вес всех гантелей в наборе, кг": "1.5", "форма дисков": "гексагональная", "длина грифа, мм": "150", "диаметр грифа, мм": "25", "регулируемая": "нет", "в кейсе": "нет", "материал": "металл", "покрытие дисков": "неопреновое", "покрытие грифа": "неопреновое", "цвет": "серый", "код товара": "100025816572"}, "formatted_text_length": 575}}
```

## Pattern 12

**Pattern:** Fishing hooks with different explicit hook sizes are negative variants even when the line and manufacturer are the same.

**Concrete difference:** Volzhanka Chinu hooks are size 8 in one listing and size 3/0 in the other.

**Why this is a strict negative signal:** Hook gauge is the functional size of the product and cannot be treated as a cosmetic listing difference.

**Evidence pair id:** pair_0749

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0749", "source_row_index": 153544, "fold": 1, "category": "Спорт и отдых", "target": 0.0, "item1": {"id": 360777312742, "name": "volzhanka крючки \"волжанка\" chinu № 8, 10 шт", "attributes": {"тип": "крючок", "размер": "8", "вид крючка/вертлюжка": "одинарный", "количество в наборе": "10 шт.", "вид ловли": "карповая", "вид рыбы": "на карася, на леща, на амура, на сазана", "особенности": "бородки на цевье"}, "formatted_text_length": 267}, "item2": {"id": 438086712396, "name": "волжанка, крючки \"волжанка\" chinu № 3/0, 2 набора по 10 штук", "attributes": {"бренд": "волжанка", "страна-изготовитель": "россия", "вид крючка": "одинарный", "размер крючка, n": "3/0", "количество упаковок, шт": "10", "тип": "крючок рыболовный"}, "formatted_text_length": 237}}
```

## Annotation caveats

- `pair_0714` is positive despite “для сайга-308” versus “на вепрь-308”; this may be a label error or an insufficiently captured compatibility relationship, so platform wording alone is not used as a general negative rule.
- `pair_0725` is positive although the first title lacks the `REV` model/3 W detail present in the other listing; sparse titles should not be used to infer a negative.
- `pair_0727` is positive for closely described Orzax gummies but gives 180 g versus 250 g, so package-weight discrepancies are not treated as universally decisive without clear variant evidence.
- Several negatives look potentially under-specified rather than clearly different: `pair_0705`, `pair_0721`, `pair_0724`, `pair_0735`, and `pair_0739`. They are not used as evidence for broad rules.
- `pair_0746` contains kiln/heat-exchanger tubing in this sport category; it is likely a category anomaly and is not used.
