## Pattern 1

**Pattern:** For otherwise comparable storage baskets, a different stated capacity together with different dimensions is a strict negative signal.

**Concrete difference:** The first basket is 7 L with 30×25 cm top dimensions, whereas the second is 15 L and 43×37×33 cm.

**Why this is a strict negative signal:** Capacity and physical dimensions specify different basket variants, not merely alternate listing text.

**Evidence pair id:** pair_0300

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0300", "source_row_index": 58251, "fold": 1, "category": "Дом и сад", "target": 0.0, "item1": {"id": 137438975655, "name": "корзина \"дачная\", 7 литров, дно 25х17 см, верх 30х25 см, высота 30 см", "attributes": {"артикул": "sl-2907332-1-sl212p1-33", "особенности емкостей для хранения": "жесткая конструкция", "бренд": "нет бренда", "цвет товара": "бежевый", "страна-изготовитель": "россия", "высота, см": "30", "длина, см": "30", "объем главного отделения, л": "7", "вид выпуска товара": "ручная, авторская работа", "материал": "лоза", "комплектация": "корзина \"дачная\", 7 литров, дно 25х17 см, верх 30х25 см, высота 30 см, 1 штука", "единиц в одном товаре": "1", "тип": "корзина плетеная", "ширина, см": "25", "число секций, шт": "1"}, "formatted_text_length": 586}, "item2": {"id": 532575999226, "name": "nobrand корзина «боровичок», 15 л, дно - 34х21, верх - 43х33, h= 19(37) см", "attributes": {"бренд": "nobrand", "страна-производитель": "россия", "коллекция": "боровичок", "цвет производителя": "бежевый", "тип товара": "корзина", "количество контейнеров в комплекте, шт.": "1", "назначение": "для мелочей", "количество отделений": "1", "ширина, см": "33", "высота, см": "37", "глубина, см": "43", "размеры (шxвxг), см": "43х37х33", "размеры отделений (шxвxг), см": "43х37х33", "объем, л": "15", "складной": "нет", "жесткость": "каркас", "с крышкой": "нет", "с ручками": "да", "застежка": "нет", "съемные разделители": "нет", "колеса": "нет", "материал корпуса": "ивовая лоза", "стиль": "кантри", "цвет корпуса": "бежевый", "код товара": "100049821712"}, "formatted_text_length": 705}}
```

## Pattern 2

**Pattern:** Two entrance mats with different stated width and length are negative even when their brand, colour, and general construction match.

**Concrete difference:** One Premium Grass mat is 150×300 cm, while the other is 120×250 cm.

**Why this is a strict negative signal:** The dimensions identify different purchasable size variants.

**Evidence pair id:** pair_0301

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0301", "source_row_index": 210300, "fold": 4, "category": "Дом и сад", "target": 0.0, "item1": {"id": 489626383890, "name": "коврик входной грязезащитный 150х300 см premium grass komet, высота ворса 5 мм, черный. коврик придверный", "attributes": {"артикул": "4821349-150х300", "бренд": "premium grass", "основа ковра": "резина", "цвет товара": "черный", "название цвета": "черный", "страна-изготовитель": "россия", "форма ковра": "прямоугольник", "длина, см": "300", "длина, м": "3", "вид выпуска товара": "фабричное производство", "материал": "полиэстер", "комплектация": "коврик входной грязезащитный 150х300 см premium grass komet, высота ворса 5 мм, черный. коврик придверный - 1 шт.", "вид принта": "без рисунка", "единиц в одном товаре": "1", "комната": "балкон", "стиль дизайна": "классический", "размер упаковки (длина х ширина х высота), см": "150х12,5х12,5", "тип": "коврик придверный", "гарантийный срок": "10 лет", "ширина, м": "1.5", "высота ворса, мм": "8"}, "formatted_text_length": 808}, "item2": {"id": 128849054941, "name": "premium grass / коврик входной грязезащитный 120х250 см, в рулоне, ворс 5 мм", "attributes": {"размер коврика": "120х250 см", "форма коврика": "прямоугольная", "материал изделия": "резина; полиэстер", "ширина предмета": "120 см", "высота предмета": "0.82 см", "вес без упаковки (кг)": "6.6 кг", "вес с упаковкой (кг)": "6.7 кг", "цвет": "черный", "страна производства": "россия", "комплектация": "коврик входной грязезащитный 120х250 см, в рулоне, ворс 5 мм - 1 шт.", "длина упаковки": "120 см", "высота упаковки": "13 см", "ширина упаковки": "13 см", "валюта": "rub"}, "formatted_text_length": 543}}
```

## Pattern 3

**Pattern:** A single piece and a multi-piece tableware set are negative when the advertised item count differs.

**Concrete difference:** The first Bonna listing is one 22 cm plate, while the other explicitly contains two 22 cm plates.

**Why this is a strict negative signal:** Pack count changes the sold offer, despite equal brand, material, and diameter.

**Evidence pair id:** pair_0302

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0302", "source_row_index": 320253, "fold": 1, "category": "Дом и сад", "target": 0.0, "item1": {"id": 747324392246, "name": "bonna тарелка, 1 шт, фарфор, диаметр 22 см", "attributes": {"артикул": "00000066905/1/мг", "бренд": "bonna", "цвет товара": "бежевый", "страна-изготовитель": "турция", "диаметр, см": "22", "размеры, мм": "220", "вид выпуска товара": "фабричное производство", "материал": "фарфор", "комплектация": "тарелка - 1 шт.", "количество предметов": "1", "тип": "тарелка"}, "formatted_text_length": 343}, "item2": {"id": 549755868832, "name": "bonna / обеденные тарелки 22 см фарфор 2 шт.", "attributes": {"ширина предмета": "22 см", "количество предметов в упаковке": "2 шт.", "цвет": "бежевый, светло-бежевый, охра, черный", "материал посуды": "фарфор; упрочненный фарфор ресторанного класса; ресторанный фарфор", "глубина предмета": "1.6 см", "особенности посуды": "матовая поверхность; разрешено для микроволновой печи; разрешено для посудомоечной машины", "декоративные элементы": "мозаика", "форма посуды": "круглая", "страна производства": "турция", "комплектация": "набор из 2 основных тарелок 22 см bonna mesopotamia; тарелки и блюда; набор тарелок; красивая посуда; посуда для подачи", "вес товара с упаковкой (г)": "1280 г", "длина упаковки": "25 см", "высота упаковки": "12 см", "ширина упаковки": "25 см", "валюта": "rub"}, "formatted_text_length": 764}}
```

## Pattern 4

**Pattern:** Stovetop kettles with explicitly different capacities are negative variants.

**Concrete difference:** The Mayer & Boch kettle is stated as 3 L in one listing and 4.3 L in the other.

**Why this is a strict negative signal:** Capacity is a core purchasable specification for this product type.

**Evidence pair id:** pair_0303

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0303", "source_row_index": 359712, "fold": 2, "category": "Дом и сад", "target": 0.0, "item1": {"id": 841813651120, "name": "mayer & boch mb 30964 чайник со свистком 3 л", "attributes": {"артикул": "111496", "бренд": "mayer&boch", "цвет товара": "серый", "диаметр, см": "19", "вид выпуска товара": "фабричное производство", "материал": "нержавеющая сталь", "тип": "чайник", "объем, л": "3"}, "formatted_text_length": 252}, "item2": {"id": 249108162839, "name": "mayer&boch / чайник со свистком 4,3 л", "attributes": {"цвет": "серебристый", "высота предмета": "25.5 см", "объем (л)": "4.3 л", "страна производства": "китай", "комплектация": "крышка; чайник", "вес с упаковкой (кг)": "1.22 кг", "длина упаковки": "24 см", "высота упаковки": "26 см", "ширина упаковки": "24 см", "валюта": "rub"}, "formatted_text_length": 313}}
```

## Pattern 5

**Pattern:** Tablecloths with a different stated finished length are negative, even if brand, colour, density, and shape agree.

**Concrete difference:** The Dolyana tablecloths are 145×220 cm and 144×180 cm respectively.

**Why this is a strict negative signal:** Their lengths differ by 40 cm, defining distinct size variants.

**Evidence pair id:** pair_0304

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0304", "source_row_index": 69678, "fold": 1, "category": "Дом и сад", "target": 0.0, "item1": {"id": 163208798050, "name": "доляна скатерть рогожка 145x220см", "attributes": {"артикул": "7533940", "бренд": "доляна", "цвет товара": "бежевый", "название цвета": "бежевый", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-5/6272715701.jpg", "страна-изготовитель": "россия", "плотность ткани, г/кв.м": "164", "форма": "прямоугольник", "длина, см": "220", "вид выпуска товара": "фабричное производство", "материал": "рогожка", "комплектация": "скатерть - 1 шт.", "вид принта": "геометрический", "размер, см": "145x220", "тип": "скатерть", "ширина, см": "144"}, "formatted_text_length": 507}, "item2": {"id": 850403607497, "name": "доляна / скатерть geometry 144х180см, 100% хлопок, 164 г/м2", "attributes": {"длина предмета": "180 см", "ширина предмета": "144 см", "форма салфетки/скатерти": "прямоугольная", "место применения скатерти": "на стол", "декоративные элементы": "без элементов", "страна производства": "россия", "цвет": "бежевый", "комплектация": "товар в фирменной упаковке", "вес товара с упаковкой (г)": "505 г", "длина упаковки": "26 см", "высота упаковки": "3 см", "ширина упаковки": "24 см", "валюта": "rub"}, "formatted_text_length": 472}}
```

## Pattern 6

**Pattern:** Music posters with different explicitly named tracks are negative even when their brand, format, medium, and presentation are the same.

**Concrete difference:** One Bulbacraft poster is for «pharaoh - на луне», while the other is for «полматери - ричледи».

**Why this is a strict negative signal:** The named print is the defining content of a decorative poster.

**Evidence pair id:** pair_0311

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0311", "source_row_index": 63769, "fold": 4, "category": "Дом и сад", "target": 0.0, "item1": {"id": 146029001913, "name": "не определен \"pharaoh - на луне\" spotify постер, музыкальная рамка, плакат, пластинка подарок bulbacraft (10х20см)", "attributes": {"бренд": "bulbacraft", "тип": "постер", "высота": "20.5 см", "ширина": "10.5 см", "форма": "прямоугольник", "поверхность": "полуматовая", "вид искусства": "принт", "жанр": "музыка", "техника": "цифровая печать", "материал основы": "акрил", "тип холста": "синтетический", "оформление": "без рамы", "толщина": "0.5 см", "комплектация": "1 музыкальная рамка, 2 ножки-подставки, крафтовая упаковка.", "дополнительная информация": "пожалуйста, не забывайте снимать защитную пленку!", "габариты транспортной упаковки": "10.50х20.50х0.50 см", "вес в транспортной упаковке": "0.02 кг"}, "formatted_text_length": 678}, "item2": {"id": 68719555370, "name": "\"полматери - ричледи\" spotify постер, музыкальная рамка, плакат, пластинка подарок bulbacraft. (10х20см)", "attributes": {"артикул": "yan0110", "бренд": "bulbacraft", "цвет товара": "прозрачный", "название цвета": "ричледи", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-5/6618438509.jpg", "страна-изготовитель": "беларусь", "жанр картины": "музыка", "для кого": "универсально", "праздник": "универсальный", "признак 18+": "false", "длина большей стороны, см": "20", "вид выпуска товара": "фабричное производство", "материал": "акрил", "комплектация": "музыкальная рамка, ножки, крафтовая упаковка", "оформление": "без рамы", "комплектация постера": "без паспарту", "название принта": "ричледи", "длина меньшей стороны, см": "10", "тип": "постер"}, "formatted_text_length": 720}}
```

## Pattern 7

**Pattern:** A conventional saucepan and a pilaf kazan are negative when the product type and intended cooking vessel differ.

**Concrete difference:** The first offer is a 4 L saucepan, whereas the second is a thick-walled kazan for pilaf.

**Why this is a strict negative signal:** Both are aluminium cookware, but their declared vessel types and use are different.

**Evidence pair id:** pair_0312

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0312", "source_row_index": 243169, "fold": 2, "category": "Дом и сад", "target": 0.0, "item1": {"id": 566935789774, "name": "кастрюля, алюминий, 4 л", "attributes": {"артикул": "087815", "бренд": "нет бренда", "цвет товара": "черный", "вид выпуска товара": "фабричное производство", "материал": "алюминий", "антипригарное покрытие": "да", "комплектация": "кастрюля ап 4л стеклянная крышка престиж - 1 шт", "особенности посуды": "жаропрочная", "размер крышки, см": "20", "совместимые плиты": "для газовых плит", "тип": "кастрюля", "объем, л": "4 л", "вес товара, г": "1500"}, "formatted_text_length": 425}, "item2": {"id": 850403529671, "name": "мечта / казан для плова с крышкой литой толстостенный", "attributes": {"объем казана": "3 л; 4 л; 5л", "цвет": "черный", "материал посуды": "алюминий", "ширина предмета": "26 см", "глубина предмета": "26 см", "толщина стенок": "4 мм", "внутреннее покрытие": "антипригарное; антипригарное покрытие; устойчивое к царапинам", "страна производства": "россия", "комплектация": "крышка; казан", "вес с упаковкой (кг)": "2 кг", "длина упаковки": "26 см", "высота упаковки": "15 см", "ширина упаковки": "26 см", "валюта": "rub"}, "formatted_text_length": 495}}
```

## Pattern 8

**Pattern:** A fitted sheet and a duvet cover are negative when the declared bedding component differs, even if both use percale.

**Concrete difference:** One item is a 200×200 fitted sheet, and the other is a 143×215 1.5-bed duvet cover.

**Why this is a strict negative signal:** These are non-interchangeable components of bedding sets.

**Evidence pair id:** pair_0319

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0319", "source_row_index": 254288, "fold": 3, "category": "Дом и сад", "target": 0.0, "item1": {"id": 592705594839, "name": "простыня ивановотекстиль 200х200 на резинке перкаль мадонна (борт 25 см)", "attributes": {"артикул производителя": "242824", "модель": "мадонна", "бренд": "ивановотекстиль", "страна-производитель": "россия", "произведено в ивановской области": "да", "цвет производителя": "сиреневый", "тип": "простыня", "материал": "перкаль", "состав ткани": "100% хлопок", "количество предметов, шт.": "1", "размер, см": "200x200", "непромокаемая": "нет", "на резинке": "да", "основной цвет": "фиолетовый", "декоративная отделка": "без отделки", "код товара": "600011799886", "цвет": "сиреневый"}, "formatted_text_length": 550}, "item2": {"id": 523986027660, "name": "королевское искушение пододеяльник перкаль, 1,5 спальный, 143x215", "attributes": {"артикул": "tkd704081", "бренд": "королевское искушение", "застежка пододеяльника": "с прорезью", "спальность": "1,5 спальный", "размер пододеяльника": "143x215", "состав материала": "хлопок", "название материала": "перкаль", "вид выпуска товара": "фабричное производство", "тип": "пододеяльник"}, "formatted_text_length": 364}}
```

## Pattern 9

**Pattern:** Bathroom mirrors with different width and materially different feature sets are negative variants.

**Concrete difference:** One Maskota mirror is 60×80 cm with contour lighting, while the other is 40×80 cm and adds front lighting, clock, heating, and brightness control.

**Why this is a strict negative signal:** Both width and included functionality identify a different mirror configuration.

**Evidence pair id:** pair_0325

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0325", "source_row_index": 186326, "fold": 1, "category": "Дом и сад", "target": 0.0, "item1": {"id": 438086685204, "name": "maskota зеркало для ванной \"lexa с управлением взмахом руки, нейтральный свет 4500k, контурная подсветка на стену\", 60 см х 80 см", "attributes": {"артикул": "lexa-st-80/60/4500k/vzmah", "бренд": "maskota", "цвет товара": "белый", "название цвета": "нейтральная подсветка 4500к", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-u/6739390974.jpg", "страна-изготовитель": "россия", "высота, см": "80", "вид выпуска товара": "фабричное производство", "материал корпуса": "пластик", "особенности зеркала": "без рамы", "форма зеркала": "прямоугольное", "комплектация": "зеркало, крепление.", "единиц в одном товаре": "1", "тип": "зеркало для ванной", "вид зеркала": "напольное", "гарантийный срок": "1 год", "ширина, см": "60"}, "formatted_text_length": 697}, "item2": {"id": 618475382466, "name": "maskota зеркало для ванной прямоугольное, фронтальная подсветка по краю, нейтральный свет 4500k, часы, подогрев, управление взмахом руки, 80х40 см.", "attributes": {"ширина": "40", "высота": "80", "цвет подсветки": "нейтральный белый", "назначение": "для ванной", "размещение": "настенное", "особенности": "без рамы, подогрев, подсветка, регулировка яркости подсветки, с часами", "форма": "прямоугольная", "материал рамы": "пластик", "комплектация": "зеркало, крепление", "дополнительная информация": "нейтральная подсветка 4500k, управление взмахом, часы, подогрев"}, "formatted_text_length": 550}}
```

## Pattern 10

**Pattern:** Ready-made tulle curtains with different finished width and height are negative size variants.

**Concrete difference:** The first tulle is 350×220 cm, while the second is 600×190 cm.

**Why this is a strict negative signal:** Both dimensions change substantially, so the items fit different window sizes.

**Evidence pair id:** pair_0327

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0327", "source_row_index": 129848, "fold": 2, "category": "Дом и сад", "target": 0.0, "item1": {"id": 300647854530, "name": "solleh&home / тюль штора белая 350х220 с утяжелителем под лен в спальню", "attributes": {"цвет": "белый, серебристо-белый", "плотность ткани": "90 г/кв.м", "материал изделия": "полиэстер", "ширина штор": "350 см; 3.5 м; 3.5 метра", "высота штор": "220 см; 2.2 м; 2.2 метра", "страна производства": "россия", "комплектация": "занавеска белая лен с утяжелителем 1 шт шторы", "вес с упаковкой (кг)": "0.3 кг", "валюта": "rub"}, "formatted_text_length": 408}, "item2": {"id": 249108113087, "name": "nusayba houm тюль высота 190 см, ширина 600 см, крепление - лента, белый;серебристо-белый", "attributes": {"артикул": "лен-рогожка-серебристый - 600х190", "бренд": "nusayba houm", "цвет товара": "белый", "название цвета": "белый;серебристо-белый", "образец цвета": "https://cdn1.ozone.ru/s3/multimedia-1-v/6907656415.jpg", "страна-изготовитель": "россия", "плотность ткани, г/кв.м": "75", "высота, см": "190", "вид крепления штор": "лента", "длина карниза, см": "600", "вид выпуска товара": "фабричное производство", "материал": "полиэстер", "комплектация": "1 штук готовый тюль ", "вид принта": "без рисунка", "количество в упаковке, шт.": "1", "единиц в одном товаре": "1", "состав комплекта": "1 штора", "тип": "тюль", "гарантийный срок": "3", "ширина, см": "600"}, "formatted_text_length": 730}}
```

## Pattern 11

**Pattern:** Decorative cutting boards with different printed design and dimensions are negative variants.

**Concrete difference:** One board is a 27.5×19.5 cm «с новым годом. дети и снеговик» board, whereas the other is a 36×28 cm «хохлома» board-tray.

**Why this is a strict negative signal:** The product dimensions and the decorative print both differ.

**Evidence pair id:** pair_0336

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0336", "source_row_index": 361321, "fold": 4, "category": "Дом и сад", "target": 0.0, "item1": {"id": 841813722423, "name": "доска разделочная сувенирная \"с новым годом. дети и снеговик\", 27,5х19,5см", "attributes": {"артикул": "3664887", "бренд": "нет бренда", "вид выпуска товара": "фабричное производство", "тип": "разделочная доска"}, "formatted_text_length": 207}, "item2": {"id": 266288047618, "name": "дарим красиво разделочная доска-поднос \"хохлома\" 36х28 см", "attributes": {"длина": "36 см", "ширина": "28 см", "тип": "разделочная доска", "форма": "прямоугольная"}, "formatted_text_length": 160}}
```

## Pattern 12

**Pattern:** Seeds of the same crop are negative when their named cultivar differs.

**Concrete difference:** Both offers are melon seeds from Aelita, but one is «медовые соты» and the other «нектарин».

**Why this is a strict negative signal:** A cultivar is the identity of the planted variety, not a cosmetic listing distinction.

**Evidence pair id:** pair_0340

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0340", "source_row_index": 300999, "fold": 1, "category": "Дом и сад", "target": 0.0, "item1": {"id": 704374686725, "name": "семена дыня медовые соты (1г) - аэлита", "attributes": {"артикул": "дыня-106", "бренд": "агрофирма аэлита", "страна-изготовитель": "россия", "место для выращивания": "улица", "регион выращивания": "для всех регионов", "количество заводских упаковок": "1", "сорт семян": "дыня медовые соты", "тип": "семена", "вид посадочного материала": "дыни"}, "formatted_text_length": 329}, "item2": {"id": 403726951829, "name": "семена дыня аэлита нектарин 1 уп.", "attributes": {"бренд": "аэлита", "тип": "семена фруктовых деревьев и кустарников", "вид": "дыня", "сорт": "нектарин", "лекарственное": "нет", "самоопыляемый": "нет", "корневая система саженцев": "закрытая", "количество пакетов семян, клубней, луковиц, саженцев, шт.": "1", "набор": "нет", "выращивание": "открытый грунт; теплица", "почва": "нейтральная", "максимальная высота, см": "50", "окраска плода": "желтая", "посев на рассаду": "апрель", "вес плода, г": "1500", "посадка в грунт": "апрель; май", "холодоустойчивый": "да", "теневыносливый": "да", "уход": "подкормка; полив", "цветение": "август; октябрь; сентябрь", "сроки созревания": "раннеспелый", "сбор урожая": "июнь; июль; август; сентябрь", "форма плода": "округлая", "код товара": "100028070196"}, "formatted_text_length": 752}}
```

## Pattern 13

**Pattern:** A general flower fertiliser and a fertiliser targeted to named plant groups are negative when their declared application differs.

**Concrete difference:** One Bona Forte fertiliser is for flower crops generally, while the other is specifically for camellia, azalea, rhododendron, and heather.

**Why this is a strict negative signal:** The stated plant applicability defines a more specific fertiliser formulation/use case.

**Evidence pair id:** pair_0343

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0343", "source_row_index": 327721, "fold": 2, "category": "Дом и сад", "target": 0.0, "item1": {"id": 764504278107, "name": "bona forte удобрение,285мл", "attributes": {"артикул": "o1157563", "бренд": "bona forte", "страна-изготовитель": "россия", "форма выпуска средства": "жидкость", "назначение удобрения": "для цветочных культур", "количество в упаковке, шт.": "1", "область применения": "для стимуляции роста и питания", "тип": "удобрение", "объем, мл": "285", "вес товара, г": "380"}, "formatted_text_length": 347}, "item2": {"id": 257698166111, "name": "bona forte удобрение для камелии, азалии, рододендронов и вересковых bona forte, флакон, 285 мл", "attributes": {"объем": "0.285 л", "тип упаковки": "бутылка"}, "formatted_text_length": 158}}
```

## Pattern 14

**Pattern:** Afghan kazans with different stated capacities are negative variants even when brand family, material, colour, and origin agree.

**Concrete difference:** The Rashko Baba kazan is 10 L in one offer and 8 L in the other.

**Why this is a strict negative signal:** Capacity is a central, explicit configuration of a cooking vessel.

**Evidence pair id:** pair_0348

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0348", "source_row_index": 101893, "fold": 2, "category": "Дом и сад", "target": 0.0, "item1": {"id": 231928385489, "name": "rashko baba / афганский казан 10 литров с доп прокл.", "attributes": {"объем казана": "10 л; 10л; 10", "цвет": "черный", "материал посуды": "алюминий; алюминий пищевой; алюминиевый слав", "ширина предмета": "26 см", "глубина предмета": "23.5 см", "толщина стенок": "11 мм", "внутреннее покрытие": "алюминий; антипригарное покрытие; алюминий пищевой", "страна производства": "афганистан", "комплектация": "крышка - 1 шт; афганский казан - 1 шт; клапан - 2 шт; прокладка - 2 шт; аварийный клапан - 1 шт", "вес с упаковкой (кг)": "4 кг", "длина упаковки": "35 см", "высота упаковки": "30 см", "ширина упаковки": "35 см", "валюта": "rub"}, "formatted_text_length": 609}, "item2": {"id": 850403550546, "name": "rashko baba ltd казан, 8 л", "attributes": {"артикул": "afgan-sb-8l", "дно казана": "плоское", "толщина дна, мм": "11", "бренд": "rashko baba ltd", "материал казана": "алюминий", "вид казана": "афганский", "цвет товара": "черный", "название цвета": "черный", "страна-изготовитель": "афганистан", "диаметр дна, см": "20", "вид выпуска товара": "фабричное производство", "вид блюда": "жаркое", "антипригарное покрытие": "да", "тип антипригарного покрытия": "эмалевое", "комплектация": "казан - 1 шт,\nбоковые ручки - 2 шт,\nверхняя ручка - 1 шт, \nкрышка - 1 шт,\nпредохранительный клапан - 1 шт,\nклапан для отвода пара - 2 шт,\nкоробка.\nв дополнении получите три подарка!", "упаковка": "картонная коробка", "особенности посуды": "крышка в комплекте", "размер крышки, см": "18", "совместимые плиты": "для газовых плит", "тип": "казан", "объем, л": "8", "гарантийный срок": "5 лет", "вес товара, г": "4100"}, "formatted_text_length": 863}}
```

## Annotation caveats

- `pair_0310` is labelled negative although both titles describe the same one-piece daylily root and the available attributes are strongly aligned; it is not used as a pattern.
- `pair_0321` has the same mini-bar title on both sides, but one side is typed as a shot-glass set and the other as a decorative bar; the label/type discrepancy is suspicious.
- `pair_0323` appears to describe the same 3 m, 80-LED, eight-mode indoor string light; only the named decorative attachments/light description may differ, so it is not used.
- `pair_0333`, `pair_0346`, and `pair_0349` lack enough decisive, conflicting detail for safe generalisation despite their negative labels.
- Positive pairs show that different brands, countries, material fields, or coarse product-type wording alone are not strict negative signals: `pair_0316` (garlic press), `pair_0324` (33 cm magnetic knife holder), `pair_0334` (bath mat), `pair_0342` (carpet versus runner), and `pair_0347` (5 L Afghan kazan) are labelled positive.
