## Pattern 1

**Pattern:** Pet food explicitly intended for cats and pet food explicitly intended for dogs are different products, even when the veterinary line, flavour, and weight coincide.

**Concrete difference:** One Farmina Vet Life Diabetic item is for cats, while the other is explicitly for dogs.

**Why this is a strict negative signal:** The animal species is a stated intended-use constraint of the food; the positive examples only support close descriptions of the same intended product, not cross-species food.

**Evidence pair id:** pair_0805

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0805", "source_row_index": 212998, "fold": 1, "category": "Товары для животных", "target": 0.0, "item1": {"id": 498216258444, "name": "farmina / vet life diabetic для кошек с курицей 2 кг", "attributes": {"количество предметов в упаковке": "1 шт.", "вес товара без упаковки (г)": "2000 г", "назначение корма": "с сахарным диабетом", "вид вкуса корма": "птица", "основной вкус корма": "курица", "вид животного": "для кошек", "порода кошки": "все породы кошек", "размер животного": "для всех пород", "возраст животного": "взрослые", "класс корма": "супер-премиум", "срок годности": "18 месяцев", "страна производства": "сербия", "вес товара с упаковкой (г)": "2000 г", "длина упаковки": "30 см", "высота упаковки": "10 см", "ширина упаковки": "15 см", "валюта": "rub"}, "formatted_text_length": 600}, "item2": {"id": 120259206415, "name": "сухой диетический корм farmina vet life diabetic, для собак с сахарным диабетом, курица, 2кг", "attributes": {"возраст животного": "для взрослых", "размер животного": "средний", "артикул": "127341", "бренд": "farmina", "состав": "дегидратированное куриное мясо, спельта, гидролизат животного белка, дегидратированные цельные яйца, овес, пульпа свеклы (без сахаров), волокна гороха, дегидратированная рыбная мука, животный жир, семя льна, рыбий жир, фруктоолигосахариды, маннанолигосахариды, калия хлорид, натрия хлорид, карбонат кальция, монодикальция фосфат, подорожник, глюкозамин, хондроитина сульфат, экстракт календулы (источник лютеина). источник углеводов: спельта, овес.сырой белок 33,00%, сырой жир и масла 10,50%, сырая клетчатка 7,90%, сырая зола 7,90%, кальций 1,00%, фосфор 0,90%, натрий 0,35%, калий 0,70%, магний 0,12%, крахмала 17,50%, общего сахара 1,30%, омега-3 0,35%, омега-6 2,30%, epa 1,15%, dha 0,20%.", "страна-изготовитель": "сербия", "порода собаки": "для любой породы", "особенности": "без глютена", "основной ингредиент": "птица", "вкус корма для животных": "курица", "минеральные вещества": "зола;калий;кальций;клетчатка;магний;натрий;фосфор", "упаковка": "пластиковый пакет", "предназначено для": "для собак", "особые потребности": "с избыточным весом", "срок годности в днях": "540", "условия хранения": "в сухом прохладном месте", "размер упаковки (длина х ширина х высота), см": "35 x 23 x 10", "тип": "корм сухой", "ветеринарная диета": "заболевания печени", "вес товара, г": "2000"}, "formatted_text_length": 1480}}
```

## Pattern 2

**Pattern:** Different explicitly named veterinary or functional food formulas are negative, even within the same brand, food type, species, protein family, and nominal weight.

**Concrete difference:** Farmina Vet Life Ultrahypo is stated for allergy reactions, whereas Farmina Vet Life Cat Hairball is for removing hairballs.

**Why this is a strict negative signal:** These are separately named formulations with different stated dietary purposes, rather than alternate descriptions of one formula.

**Evidence pair id:** pair_0801

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0801", "source_row_index": 16392, "fold": 1, "category": "Товары для животных", "target": 0.0, "item1": {"id": 34359783593, "name": "farmina / сухой корм для кошек farmina vet life ultrahypo 2 кг", "attributes": {"вес товара без упаковки (г)": "2000 г", "назначение корма": "при аллергии; при аллергических реакциях", "вид вкуса корма": "с рыбой", "основной вкус корма": "рыба", "вид животного": "для кошек", "порода кошки": "для всех кошек", "размер животного": "все размеры", "возраст животного": "для всех", "срок годности": "540 дней", "страна производства": "сербия", "вес товара с упаковкой (г)": "2000 г", "комплектация": "1 пакет.", "длина упаковки": "8 см", "высота упаковки": "40 см", "ширина упаковки": "25 см", "валюта": "rub"}, "formatted_text_length": 577}, "item2": {"id": 704374774485, "name": "farmina vet life cat hairball сухой корм для взрослых кошек для выведения шерстяных комочков - 2 кг x 2 шт.", "attributes": {"возраст животного": "для взрослых", "артикул": "zoo21971", "бренд": "farmina", "порода кошки": "для любой породы", "страна-изготовитель": "италия", "порода собаки": "для любой породы", "основной ингредиент": "рыба", "вкус корма для животных": "сардины", "комплектация": "farmina vet life cat hairball сухой корм для взрослых кошек для выведения шерстяных комочков - 2 кг x 2 шт.", "упаковка": "пакет", "предназначено для": "для кошек", "единиц в одном товаре": "2", "срок годности в днях": "365", "условия хранения": "в сухом месте", "тип": "корм сухой", "вес товара, г": "2000"}, "formatted_text_length": 676}}
```

## Pattern 3

**Pattern:** A food for adult cats and a food explicitly sold for puppies are negative when the product titles state those incompatible life-stage/species uses.

**Concrete difference:** Gemon Cat is for adult cats with chicken and turkey, while the other Gemon item is food for puppies with chicken.

**Why this is a strict negative signal:** The intended animal and life stage are both expressly different; shared brand and poultry flavour do not identify one SKU.

**Evidence pair id:** pair_0812

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0812", "source_row_index": 226907, "fold": 2, "category": "Товары для животных", "target": 0.0, "item1": {"id": 532575994810, "name": "gemon cat для взрослых кошек, с курицей и индейкой 400гр", "attributes": {"возраст животного": "для взрослых", "размер животного": "крупный", "артикул": "70297004", "бренд": "gemon", "порода кошки": "для любой породы", "состав": "злаки, мясо и мясные продукты (свежее мясо курицы мин. 8%, свежее мясо индейки мин. 8%), масла и жиры (лососевое масло 0,5%), овощи, рыба и рыбные субпродукты, минералы, пивные дрожжи (маннаноолигосахариды мос 500 мг/кг), фос (фруктоолигосахариды 500 мг/кг), юкка шидигера 1.000 мг/кг.\n\nпитательная ценность: сырой белок 32%, сырые масла и жиры 13%, сырая клетчатка 2,5%, сырая зола 8,5%, магний 0,12%, кальций 2,5%, фосфор 1,5%, омега-6 незаменимые жирные кислоты 2,38%, омега-3 незаменимые жирные кислоты 0,4%.", "страна-изготовитель": "италия", "особенности": "без гмо", "класс корма": "премиум", "вкус корма для животных": "птица", "минеральные вещества": "питательные добавки в 1 кг: витамин а (ретинола ацетат) 25 000 ме, витамин d3 (холекальциферол) 1 600 ме, витамин е (альфа-токоферола ацетат) 150 мг, таурин 1 950 мг, холина хлорид 2 200 мг, сульфат марганца моногидрат 60 мг (марганец 20 мг), оксид цинка", "комплектация": "1шт", "упаковка": "пакет", "предназначено для": "для кошек", "единиц в одном товаре": "1", "особые потребности": "профилактика зубного налета", "срок годности в днях": "540", "условия хранения": "в сухом прохладном месте", "размер упаковки (длина х ширина х высота), см": "15х13х7", "тип": "корм сухой", "вес товара, г": "400"}, "formatted_text_length": 1455}, "item2": {"id": 747324421225, "name": "gemon / корм для щенков 1 кг", "attributes": {"вес товара без упаковки (г)": "1000 г", "назначение корма": "для собак мелких пород", "вид вкуса корма": "птица", "основной вкус корма": "курица", "возраст животного": "для маленьких", "срок годности": "547 дней", "страна производства": "италия", "вес товара с упаковкой (г)": "1000 г", "комплектация": "корм - 1 шт", "длина упаковки": "20 см", "высота упаковки": "24 см", "ширина упаковки": "10 см", "валюта": "rub"}, "formatted_text_length": 441}}
```

## Pattern 4

**Pattern:** Foods with distinct named recipes and product numbers are negative even if their brand, species, food type, texture, and broad protein family overlap.

**Concrete difference:** Berkley №7 is chicken and beef in a two-unit pack, whereas Berkley №6 is chicken with duck in a one-unit pack.

**Why this is a strict negative signal:** The distinct numbered recipe and second protein identify different product offerings; the comparison is not based on pack size alone.

**Evidence pair id:** pair_0833

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0833", "source_row_index": 75518, "fold": 4, "category": "Товары для животных", "target": 0.0, "item1": {"id": 171798836835, "name": "беркли / корм влажный для кошек №7 курицаговядина 85гр, 2шт", "attributes": {"состав": "курица; соус; говядина; соевое масло; крахмал тапиоки; витамины и минералы", "количество предметов в упаковке": "2 шт.", "основной вкус корма": "курица; говядина", "вид вкуса корма": "птица; мясо", "назначение корма": "для здорового пищеварения; уход за кожей и шерстью; для здоровой и активной жизни", "вид животного": "для кошек", "порода кошки": "для всех пород", "консистенция корма": "кусочки в соусе", "класс корма": "супер-премиум", "возраст животного": "взрослые", "размер животного": "для крупных пород; для средних пород; для мелких пород", "срок годности": "3 года", "вес одного предмета (г)": "85 г", "страна производства": "таиланд", "комплектация": "2 шт.", "длина упаковки": "10 см", "высота упаковки": "4 см", "ширина упаковки": "10 см", "валюта": "rub"}, "formatted_text_length": 823}, "item2": {"id": 670014934475, "name": "консервы беркли №6 для взрослых кошек курица с уткой 85г х 1шт", "attributes": {"возраст животного": "для взрослых", "размер животного": "малый", "бренд": "berkley", "порода кошки": "абиссинская", "состав": "курица 42,5%, соус 48,2%, утиная грудка 4%, соевое масло 3,5%, крахмал тапиоки (натуральный загуститель) 1,1%, витамины и минералы 0,7%.", "страна-изготовитель": "германия", "особенности": "без ароматизаторов", "класс корма": "супер-премиум", "консистенция корма для животных": "кусочки в соусе", "основной ингредиент": "птица", "вкус корма для животных": "курица", "содержание мяса, субпродуктов, %": "47", "минеральные вещества": "протеин 11,0г; жир 3,5г; клетчатка 0г; зола 1,1г; влажность 83%; кальций 0,07г; фосфор 0,07г. добавки на 1кг продукта: витамин а 10611ме, витамин d3 225ме, витамин е 28ме, витамин в1 15,27мг, витамин в2 1,65мг, ниацин 22,57мг, пантотеновая кислота 2,35мг", "комплектация": "консервы беркли №6 для взрослых кошек курица с уткой 85г: 1 шт.", "упаковка консервированных кормов": "консервная банка", "предназначено для": "для кошек", "единиц в одном товаре": "1", "особые потребности": "при плохом аппетите", "срок годности в днях": "730", "условия хранения": "в холодильнике", "размер упаковки (длина х ширина х высота), см": "15*10*10", "тип": "корм влажный", "вес товара, г": "85"}, "formatted_text_length": 1278}}
```

## Pattern 5

**Pattern:** Cat food variants that differ in both a named target condition and a stated principal flavour are negative within the same brand.

**Concrete difference:** Karmy Sterilized for sensitive digestion with turkey is paired with Karmy Adult chicken.

**Why this is a strict negative signal:** “Sterilized” and sensitive-digestion positioning versus Adult, together with turkey versus chicken, describe distinct formulations rather than incomplete metadata for one item.

**Evidence pair id:** pair_0825

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0825", "source_row_index": 3799, "fold": 2, "category": "Товары для животных", "target": 0.0, "item1": {"id": 164840, "name": "karmy сухой sterilized для кошек с чувствительным пищеварением индейка 400г", "attributes": {"артикул": "73314", "бренд": "karmy", "состав": "дегидрированное мясо индейки - 38%, бурый цельный рис, белый цельный рис, дегидрированное мясо лосося, жир лососевый (естественный источник омега-3 и омега-6), пивные дрожжи (источник маннанолигосахаридов), витаминно-минеральный комплекс, сушеное яблоко, пробиотики (bacillus subtilis, bacillus licheniformis), юкка шидигера, таурин, сохранено витамином с и смесью природных токоферолов (источники натурального витамина е)", "страна-изготовитель": "россия", "вкус корма для животных": "индейка", "предназначено для": "для кошек", "единиц в одном товаре": "1", "срок годности в днях": "540", "тип": "корм сухой", "вес товара, г": "400"}, "formatted_text_length": 760}, "item2": {"id": 455266652809, "name": "корм для кошек karmy 1.5кг adult курица", "attributes": {"тип": "полнорационный", "класс корма": "премиум", "консистенция": "сухой", "вкус корма": "курица", "возрастная группа": "для взрослых", "упаковка": "пакет", "вес, г": "1500", "длина упаковки, см": "20", "ширина упаковки, см": "10", "высота упаковки, см": "29.5", "вес упаковки, кг": "1.51", "страна производства": "россия"}, "formatted_text_length": 360}}
```

## Pattern 6

**Pattern:** A silica-gel cat litter and a clumping litter are different products when those filler mechanisms are explicitly stated.

**Concrete difference:** Filler Aloe Vera is silica gel, while Wonder White is clumping litter.

**Why this is a strict negative signal:** The product types describe different material/absorption mechanisms, not alternate wording for one litter.

**Evidence pair id:** pair_0843

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0843", "source_row_index": 318645, "fold": 4, "category": "Товары для животных", "target": 0.0, "item1": {"id": 747324324658, "name": "filler / силикагелевый наполнитель для кошек aloe vera, 5л", "attributes": {"тип наполнителя для животных": "силикагелевый", "вид животного": "для кошек", "особенности наполнителя для животных": "с ароматом", "страна производства": "китай", "валюта": "rub"}, "formatted_text_length": 250}, "item2": {"id": 429496801706, "name": "наполнитель wonder white комкующийся 10000г.", "attributes": {"бренд": "wonder white", "страна-изготовитель": "турция", "особенности наполнителя": "с ароматизатором", "тип наполнителя": "комкующийся", "вес товара, г": "10000"}, "formatted_text_length": 219}}
```

## Pattern 7

**Pattern:** Animal shampoos with explicitly different care purposes and named active botanical ingredients are negative, even when they share a brand and the same general product type.

**Concrete difference:** Homepet hypoallergenic dog shampoo contains celandine extract, while the other Homepet shampoo is antiparasitic with geraniol for cats and dogs.

**Why this is a strict negative signal:** Hypoallergenic care and antiparasitic treatment are distinct stated uses, reinforced by different named ingredients.

**Evidence pair id:** pair_0809

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0809", "source_row_index": 186247, "fold": 4, "category": "Товары для животных", "target": 0.0, "item1": {"id": 438086681131, "name": "homepet / шампунь гипоаллергенный для собак, с экстрактом чистотела", "attributes": {"объем товара": "0.22 мл", "назначение косметики для животных": "уход за шерстью и кожей", "вид животного": "для собак", "срок годности": "547", "страна производства": "россия", "валюта": "rub"}, "formatted_text_length": 270}, "item2": {"id": 455266628716, "name": "homepet / шампунь хоумпет для кошек и собак антипаразитарный универсальный с гераниолом 220 мл", "attributes": {"вид ухода": "уход за шерстью и кожей", "артикул": "40415", "бренд": "homepet", "страна-изготовитель": "россия", "упаковка": "бутылка с дозатором", "предназначено для": "для собак", "единиц в одном товаре": "1", "срок годности в днях": "1095", "условия хранения": "при температуре от +4`с до +25`с", "тип": "шампунь для животных", "вес товара, г": "300"}, "formatted_text_length": 447}}
```

## Pattern 8

**Pattern:** Dog balls with different explicitly stated material constructions and diameters are negative variants, even under the same toy brand.

**Concrete difference:** The Trixie ball is 6 cm soft rubber, whereas Trixie Toy Balls is a 13 cm latex-and-fleece sounding ball.

**Why this is a strict negative signal:** Both the physical construction and diameter differ, so this is more than a spelling or listing-format variation.

**Evidence pair id:** pair_0802

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0802", "source_row_index": 358243, "fold": 4, "category": "Товары для животных", "target": 0.0, "item1": {"id": 833223820525, "name": "мяч для собак мягкая резина, 6 см", "attributes": {"вид игрушки для животных": "мяч", "бренд": "trixie", "размеры, мм": "60х60х60", "материал": "резина", "место для игр": "улица", "предназначено для": "для собак", "назначение игрушки": "для поддержания питомца в форме", "количество игрушек в одной упаковке": "1", "тип": "игрушка для животных"}, "formatted_text_length": 330}, "item2": {"id": 360777278388, "name": "мячик для собак trixie toy balls (3501)", "attributes": {"вид животного": "собаки", "тип": "мячик", "назначение": "аппортировка", "размер породы": "крупныйсредний", "материал": "латексфлис", "вид игрушки": "звуковая", "возраст животного": "для взрослыхдля маленькихдля пожилых", "диаметр": "13 см"}, "formatted_text_length": 285}}
```

## Pattern 9

**Pattern:** Two named harness size ranges are negative when their stated fit intervals are different product sizes.

**Concrete difference:** Trixie Soft XXS–XS is for a 25–35 cm neck, whereas Trixie Soft S–M is labelled 35–60 cm and 20 mm.

**Why this is a strict negative signal:** The explicit size labels and fit ranges designate separate harness variants; this does not rely on colour or an unreliable generic attribute.

**Evidence pair id:** pair_0815

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0815", "source_row_index": 320829, "fold": 2, "category": "Товары для животных", "target": 0.0, "item1": {"id": 747324419584, "name": "trixie, шлейка для собак soft, xxs-xs, обхват шеи 25-35см/15мм, фуксия", "attributes": {"бренд": "trixie", "страна-изготовитель": "китай", "предназначено для": "для собак", "тип": "шлейка"}, "formatted_text_length": 184}, "item2": {"id": 120259197289, "name": "trixie / 16288 шлейка soft s-m 35-60см 20мм фуксия", "attributes": {"цвет": "фуксия", "вид животного": "для собак", "материал изделия": "нейлон", "комплектация": "1 шт.", "валюта": "rub", "российский размер": "m"}, "formatted_text_length": 204}}
```

## Pattern 10

**Pattern:** Aquarium lights with distinct stated physical length and maximum power are negative variants, even if their titles use the same brand name, installation type, and light colour.

**Concrete difference:** One Jingye underwater light is 36 cm and 4 W; the other is 26 cm and 3.5 W.

**Why this is a strict negative signal:** Length and rated power are explicit model specifications, and both differ in the annotated negative pair.

**Evidence pair id:** pair_0848

**Full pair JSON:**
```json
{"sample_pair_id": "pair_0848", "source_row_index": 284091, "fold": 3, "category": "Товары для животных", "target": 0.0, "item1": {"id": 661425111276, "name": "jingye светильник подводный, jingye, 36 см, 4 вт, белый", "attributes": {"назначение": "фильтр для аквариума", "тип светильника": "подводная подсветка", "бренд": "jingye", "тип лампы": "светодиодная", "длина аквариума": "60 - 75 см", "размещение": "подводное", "максимальная мощность осветительного прибора": "4 вт", "длина": "360 мм", "цвет свечения": "белый", "цвет корпуса": "белый", "срок службы": "12 мес., использовать при температуре не ниже -10 градусов", "гарантийный срок": "1 г., гарантия на аккумулятор — 6 месяцев"}, "formatted_text_length": 507}, "item2": {"id": 549755872521, "name": "светильник подводный, jingye, 26 см, 3,5 вт, белый", "attributes": {"бренд": "нет бренда", "размеры, мм": "48 см x 3 см x 3 см", "единиц в одном товаре": "1", "размер упаковки (длина х ширина х высота), см": "48 см x 3 см x 3 см", "тип": "светильник для аквариума", "вес товара, г": "0.13"}, "formatted_text_length": 281}}
```

## Annotation caveats

- `pair_0811` is labelled negative although its titles describe the same Trixie double wooden swing with bell, same dimensions, and matching one-unit configuration; it is suspicious.
- `pair_0806` is labelled negative despite describing the same Mr.Kranch leather QR-tag collar with the same 25–30 cm range; only olive versus green wording and sparse metadata separate them.
- `pair_0807` is a near-identical six-layer charcoal gel-pads listing (60 × 60 cm, five pieces) labelled negative; the brand/source listing difference may be driving the label.
- `pair_0821` is labelled positive although the named Zооэкспресс litter models and stated dimensions differ, so dimensional differences alone are not promoted to a general rule.
- `pair_0840` is labelled positive despite the title saying “до 5 кг” and the counterpart saying “до 50 кг”; the tenfold discrepancy looks like a listing or annotation error.
- `pair_0814` is labelled positive even though one title names Tetra and the other names Tetravital with brand Borinat; brand text by itself is therefore not a reliable negative signal.
- `pair_0845` has an attribute saying the dog collar is intended for cats, and `pair_0846` has a harness listing typed as a leash; these taxonomy attributes appear unreliable.
