from sample_name_attribute_overlap import token_set


assert token_set("Вес 0,23 кг; погрешность 0.1") == {"вес", "0.23", "кг", "погрешность", "0.1"}
