## Pattern 1

**Pattern:** For ready-made corrective glasses, a different stated spherical optical power is a negative signal even when the brand, lens family, and general frame style are the same.

**Concrete difference:** The two Moretti glasses specify -8.00 versus -6.50 dioptres.

**Why this is a strict negative signal:** The listed dioptre is the prescribed optical correction; these are distinct corrective variants rather than interchangeable copies.

**Evidence pair id:** pair_0050

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0050", "source_row_index": 331309, "fold": 1, "category": "Аптека", "target": 0.0, "item1": {"id": 773094188219, "name": "очки с футляром на магните moretti мод. 81274 цвет 1 с флагманскими линзами cryol 1.56 hmc -8.00 рц 60-62", "attributes": {"диоптрии": "- 8.00", "цвет оправы": "золотистый, черный", "назначение": "для водителей", "конструкция": "ободковая", "материал оправы": "металл", "материал линз": "пластик", "форма оправы": "прямоугольная", "покрытие линз": "упрочняющее, водоотталкивающее и антистатическое, металлизированное (защита от электромагнитных волн) uv-380, просветляющее", "пол": "мужские", "ширина линзы": "54 мм", "высота линзы": "35 мм", "межцентровое расстояние": "60-62", "длина заушника": "140 мм"}, "formatted_text_length": 583}, "item2": {"id": 326417553472, "name": "очки с футляром на магните moretti мод. 81163 цвет 1 с флагманскими линзами cryol 1.56 hmc -6.50 рц 62-64", "attributes": {"наличие флексов": "нет", "бренд": "moretti", "цвет товара": "золотой", "название цвета": "black & gold", "страна-изготовитель": "россия", "размер заушника, мм": "146", "коллекция (год)": "2022.0", "конструкция очков": "ободковые", "форма оправы": "прямоугольная", "материал оправы очков": "комбинированный", "материал линз": "пластик", "стиль оправы": "офисный (бизнес)", "форма лица": "квадратная", "покрытие линз": "упрочняющее, просветляющее, металлизированное (защита от электромагнитных волн) uv-380, водоотталкивающее и антистатическое", "высота линзы, мм": "36", "ширина линзы, мм": "56", "размер моста на переносице, мм": "18", "расстояние между оптическими центрами": "62-64", "оптическая сила": "-6.50", "комплектация": "очки moretti с футляром с полимерными линзами cryol (корея), микрофибровая салфетка для ухода, футляр на магните.", "пол ребенка": "унисекс", "пол": "мужской", "общая ширина, мм": "142", "тип": "очки для дали"}, "formatted_text_length": 1020}}
```

## Pattern 2

**Pattern:** Different manufacturer frame model codes identify different ready-made glasses even if brand, lens family, dioptres, and PD range coincide.

**Concrete difference:** The two Dacchi listings are models 33644 and 33623, with different rim construction and frame geometry despite both being +2.25 Biomax glasses with PD 60-62.

**Why this is a strict negative signal:** A precise frame model code denotes the physical frame SKU; shared lens parameters do not make distinct frame models the same product.

**Evidence pair id:** pair_0053

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0053", "source_row_index": 121958, "fold": 4, "category": "Аптека", "target": 0.0, "item1": {"id": 283467949112, "name": "очки dacchi с линзами biomax +2.25 рц 60-62 мод. 33644 цвет 1", "attributes": {"диоптрии": "+ 2.25", "цвет оправы": "золотистый", "назначение": "корректирующие", "конструкция": "ободковая", "материал оправы": "металл", "материал линз": "пластик", "форма оправы": "классическая", "покрытие линз": "упрочняющее, водоотталкивающее и антистатическое, металлизированное (защита от электромагнитных волн) uv-380, просветляющее", "пол": "женские", "особенности": "с защитой от уф, салфетка в комплекте", "ширина моста": "17 мм", "ширина линзы": "53 мм", "высота линзы": "48 мм", "межцентровое расстояние": "60-62", "длина заушника": "140 мм"}, "formatted_text_length": 608}, "item2": {"id": 292057892836, "name": "очки dacchi с линзами biomax +2.25 рц 60-62 мод. 33623 цвет 1", "attributes": {"наличие флексов": "да", "бренд": "dacchi", "цвет товара": "черный", "название цвета": "black & red", "страна-изготовитель": "россия", "размер заушника, мм": "143", "коллекция (год)": "2022", "конструкция очков": "полуободковые", "форма оправы": "прямоугольная", "материал оправы очков": "пластик", "материал линз": "пластик", "стиль оправы": "повседневный", "форма лица": "квадратная", "покрытие линз": "упрочняющее, просветляющее, металлизированное (защита от электромагнитных волн) uv-380, водоотталкивающее и антистатическое", "высота линзы, мм": "36", "ширина линзы, мм": "54", "размер моста на переносице, мм": "18", "расстояние между оптическими центрами": "60-62", "оптическая сила": "+2.25", "комплектация": "очки dacchi с полимерными линзами biomax (китай), микрофибровая салфетка для ухода", "пол": "мужской", "общая ширина, мм": "140", "тип": "очки для чтения"}, "formatted_text_length": 909}}
```

## Pattern 3

**Pattern:** A photochromic lens variant and a fixed tinted lens variant are distinct glasses products even at the same signed optical power and brand.

**Concrete difference:** One Salivio pair has brown 15% gradient tint, while the other is described as brown 80% photochromic; both are -3.50.

**Why this is a strict negative signal:** The lens's light-response behaviour is an explicit product specification and changes the optical product rather than merely its listing text.

**Evidence pair id:** pair_0051

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0051", "source_row_index": 194383, "fold": 3, "category": "Аптека", "target": 0.0, "item1": {"id": 455266591300, "name": "тонированные очки с футляром-змейка salivio мод. 9044 цвет 4 с линзами nikita 1.56 gradient brown, hma/emi -3.50 рц 58-60", "attributes": {"наличие флексов": "да", "затемнение": "коричневый градиент 15%", "бренд": "salivio", "цвет товара": "серый металлик", "название цвета": "gray & black", "страна-изготовитель": "россия", "размер заушника, мм": "140", "коллекция (год)": "2022.0", "конструкция очков": "ободковые", "форма оправы": "круглая", "материал оправы очков": "комбинированный", "материал линз": "пластик", "стиль оправы": "повседневный", "форма лица": "квадратная", "покрытие линз": "упрочняющее, просветляющее, металлизированное (защита от электромагнитных волн) uv-350, водоотталкивающее и антистатическое", "высота линзы, мм": "47", "ширина линзы, мм": "50", "размер моста на переносице, мм": "21", "расстояние между оптическими центрами": "58-60", "оптическая сила": "-3.50", "комплектация": "очки salivio с полимерными линзами nikita 1.56 gradient brown, hma/emi, микрофибровая салфетка для ухода, футляр-змейка.", "пол ребенка": "унисекс", "пол": "мужской", "общая ширина, мм": "133", "тип": "очки для дали"}, "formatted_text_length": 1078}, "item2": {"id": 352187322393, "name": "фотохромные очки с футляром-змейка salivio мод. 9037 цвет 1 с линзами nikita 1.56 colophony gray, hmc+ -3.50 рц 62-64", "attributes": {"диоптрии": "- 3.50", "цвет оправы": "золотистый", "назначение": "корректирующие", "конструкция": "ободковая", "материал оправы": "металл+пластик", "материал линз": "пластик", "форма оправы": "прямоугольная", "покрытие линз": "антибликовое, антистатическое, блокирующее ультрафиолет, защитное, упрочняющее, водоотталкивающее и антистатическое, металлизированное (защита от электромагнитных волн) emi, просветляющее", "пол": "мужские", "особенности": "салфетка в комплекте, футляр в комплекте", "ширина моста": "17 мм", "ширина линзы": "54 мм", "высота линзы": "36 мм", "межцентровое расстояние": "62-64", "длина заушника": "142 мм"}, "formatted_text_length": 740}}
```

## Pattern 4

**Pattern:** Ready-made glasses with the same optical power are negative when their explicit frame geometry and frame material differ.

**Concrete difference:** Both listings are +2.0 women’s glasses with PD 62-64, but one has a pink plastic cat-eye frame and the other a white metal hexagonal frame.

**Why this is a strict negative signal:** Frame form and material describe the physical item; equal optical correction alone is insufficient for identity.

**Evidence pair id:** pair_0067

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0067", "source_row_index": 19213, "fold": 4, "category": "Аптека", "target": 0.0, "item1": {"id": 42949698619, "name": "многоочков / очки для зрения женские с диоптриями +2.0", "attributes": {"цвет линз": "нет", "цвет": "розовый", "вид оправы": "кошачий глаз", "материал оправы": "пластик", "высота линзы": "43 см", "ширина линзы": "53 см", "ширина носового моста": "16 мм", "длина дужки": "133 см", "материал линзы": "пластик", "оптическая сила": "+2.0", "расстояние между зрачками (pd)": "62-64 мм", "страна производства": "китай", "комплектация": "очки для зрения 1 -шт; футляр для очков 1-шт; салфетка для очков 1 - шт; фирменная упаковка.", "пол": "женский", "валюта": "rub"}, "formatted_text_length": 533}, "item2": {"id": 841813753216, "name": "очки для зрения женские с диоптриями +2.0", "attributes": {"наличие флексов": "да", "затемнение": "нет", "бренд": "salivio", "цилиндр (cyl)": "0", "цвет товара": "белый", "название цвета": "c9", "страна-изготовитель": "китай", "размер заушника, мм": "140", "коллекция (год)": "2023", "конструкция очков": "ободковые", "форма оправы": "гексагональная", "материал оправы очков": "металл", "материал линз": "пластик", "стиль оправы": "повседневный", "форма лица": "прямоугольная", "покрытие линз": "без покрытия", "высота линзы, мм": "40", "ширина линзы, мм": "53", "размер моста на переносице, мм": "19", "расстояние между оптическими центрами": "62-64", "оптическая сила": "+2.00", "комплектация": "очки корригирующие - 1 шт. мешочек для хранения - 1 шт.", "пол ребенка": "девочка", "пол": "женский", "общая ширина, мм": "135", "тип": "очки для чтения"}, "formatted_text_length": 803}}
```

## Pattern 5

**Pattern:** For toric contact lenses, different spherical power (SPH) is a strict negative signal even within the same Acuvue Oasys for Astigmatism line.

**Concrete difference:** The two Acuvue toric lens variants state +0.75 and +6.00 spherical power.

**Why this is a strict negative signal:** SPH is a dispensing parameter of a corrective contact lens, so a different value identifies a different prescription variant.

**Evidence pair id:** pair_0063

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0063", "source_row_index": 347436, "fold": 2, "category": "Аптека", "target": 0.0, "item1": {"id": 807454004989, "name": "acuvue oasys for astigmatism (6 линз) (+0.75/-1.75/70/8.6)", "attributes": {"ось (axis)": "70", "бренд": "acuvue", "цилиндр (cyl)": "-1.75", "цвет товара": "прозрачный", "страна-изготовитель": "ирландия", "диаметр, мм": "14.5", "особенности контактных линз": "силикон-гидрогель", "радиус кривизны": "8.6", "материал контактных линз": "сенофилкон а", "режим ношения": "дневной", "количество в упаковке": "6", "оптическая сила": "+0.75", "кислородопропускание, dk/t": "129", "дней до замены": "2 недели", "тип": "астигматические линзы", "влагосодержание, %": "38", "вес товара, г": "17"}, "formatted_text_length": 554}, "item2": {"id": 206158591546, "name": "acuvue / линзы астигматические oasys for astigmatism +6.00 -0.75 70", "attributes": {"состав": "силикон-гидрогель", "тип линз": "торические", "количество пар в упаковке": "3 шт.", "цвет": "прозрачный", "режим ношения": "дневной", "оптическая сила цилиндра": "-0.75", "ось линзы": "70", "коэффициент пропускания кислорода": "129 dk/t", "страна производства": "сша", "влагосодержание": "38 %", "количество предметов в упаковке": "6 шт.", "режим замены": "2 недели", "радиус кривизны": "8.6", "оптическая сила": "+6.0", "диаметр мкл": "14.5 мм", "комплектация": "контактные линзы acuvue oasys астигматизм +6.0 -0.75 70 - 6 шт.", "валюта": "rub"}, "formatted_text_length": 611}}
```

## Pattern 6

**Pattern:** For toric contact lenses, different cylindrical correction (CYL) is a strict negative signal within the same Biofinity Toric series.

**Concrete difference:** The Biofinity Toric variants specify cylinder -0.75 and -2.25.

**Why this is a strict negative signal:** CYL is an explicit prescription component for astigmatism; differing correction values designate different lens variants.

**Evidence pair id:** pair_0060

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0060", "source_row_index": 218717, "fold": 3, "category": "Аптека", "target": 0.0, "item1": {"id": 506806285596, "name": "контактные линзы biofinity toric 3 линзы r 8,7 -3,75/-0,75/140", "attributes": {"бренд": "biofinity", "серия": "biofinity toric", "артикул производителя": "66081077", "тип": "торические", "оптическая сила(sph)": "-3,75", "оптическая сила цилиндра(cyl)": "-0.75", "ось(ax)": "140", "радиус кривизны": "8.7", "количество штук в упаковке": "3", "количество в упаковке": "3", "частота замены": "ежемесячные", "производитель": "coopervision", "режим ношения": "дневной; пролонгированный", "тонировка": "прозрачные", "материал": "силикон-гидрогель", "вид": "асферические", "диаметр": "14.5", "влажность, %": "48", "уф-фильтр": "нет", "бьюти-линзы": "нет", "коэффициент пропускания кислорода, dk/t": "116", "толщина в центре,в миллиметрах": "0.11", "метод дезинфекции": "химический", "тип упаковки": "блистер", "страна-производитель": "сша", "код товара": "100025967744"}, "formatted_text_length": 815}, "item2": {"id": 446676632063, "name": "biofinity астигматические линзы toric 3 линзы 3 линзы ось 70 оптическая сила цилиндра -2.25, 3 шт., -2.75 / 8.7/-2.25/70, 1 месяц", "attributes": {"ось (axis)": "70", "бренд": "biofinity", "цилиндр (cyl)": "-2.25", "название цвета": "бесцветные", "страна-изготовитель": "сша", "диаметр, мм": "14.5", "радиус кривизны": "8.7", "режим ношения": "дневной", "количество в упаковке": "3", "оптическая сила": "-2.75", "дней до замены": "1 месяц", "тип": "астигматические линзы", "влагосодержание, %": "48"}, "formatted_text_length": 477}}
```

## Pattern 7

**Pattern:** For toric contact lenses, a different stated axis (AX) is a strict negative signal when the named product family and other central prescription values coincide.

**Concrete difference:** The Miru 1 Month for Astigmatism listings both state SPH -0.75, CYL -1.25, BC 8.6, diameter 14, and six lenses, but their axes are 130 and 180.

**Why this is a strict negative signal:** Axis fixes the orientation of astigmatic correction; an axis mismatch changes the prescription variant.

**Evidence pair id:** pair_0089

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0089", "source_row_index": 276602, "fold": 2, "category": "Аптека", "target": 0.0, "item1": {"id": 644245217032, "name": "астигматические линзы miru 1 month for astigmatism 6 линз sph -0.75 cyl -1.25 axis 130", "attributes": {"бренд": "miru", "артикул производителя": "miru(tor)-0.75-1.25x130", "тип": "торические", "оптическая сила(sph)": "-0,75", "оптическая сила цилиндра(cyl)": "-1.25", "ось(ax)": "130", "радиус кривизны": "8.6", "количество штук в упаковке": "6", "количество в упаковке": "6", "частота замены": "ежемесячные", "производитель": "menicon", "режим ношения": "дневной", "тонировка": "прозрачные", "материал": "силикон-гидрогель", "диаметр": "14", "влажность, %": "40", "уф-фильтр": "нет", "бьюти-линзы": "нет", "коэффициент пропускания кислорода, dk/t": "160", "тип упаковки": "блистер", "страна-производитель": "япония", "код товара": "600014951186"}, "formatted_text_length": 707}, "item2": {"id": 635655166595, "name": "miru 1-month for astigmatism sph-0.75 cyl -1.25 ax180 8.6/14.0 (6шт)", "attributes": {"ось (axis)": "180", "бренд": "miru", "цилиндр (cyl)": "-1.25", "цвет товара": "прозрачный", "страна-изготовитель": "япония", "диаметр, мм": "14.0", "радиус кривизны": "8.6", "материал контактных линз": "асмофилкон а", "режим ношения": "дневной", "количество в упаковке": "6", "оптическая сила": "-0.75", "кислородопропускание, dk/t": "161", "дней до замены": "1 месяц", "тип": "астигматические линзы", "влагосодержание, %": "40"}, "formatted_text_length": 489}}
```

## Annotation caveats

- `pair_0054` is positive although one listing is a generic Wet lubricant and the other explicitly names a cherry, warming, anal-use Wet product; it is a suspiciously broad positive and should not support treating flavour/effect omissions as a negative signal.
- `pair_0065` is positive despite PD ranges 64-66 and 60-62 and different lens/coating wording. Therefore PD range or coating text alone must not be used as a categorical negative rule.
- `pair_0059` is a negative pair with the same named Acuvue Oasys for Astigmatism family but conflicting replacement schedules (one month versus two weeks); this looks like potentially unreliable listing metadata, so replacement schedule was not extracted as a standalone rule.
- `pair_0089` also has a material and oxygen-permeability discrepancy in addition to the axis difference; the axis rule is supported by the explicit named prescription values, while the metadata discrepancy should be treated cautiously.
