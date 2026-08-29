from sample_name_attribute_overlap import candidate_intersections, token_set, token_set_without_quotes, token_set_without_word_commas


assert token_set("A4 xgp5600p 15/4 (0,23)\n0.1") == {"a4", "xgp5600p", "15/4", "0.23", "0.1"}
assert {"40/2", "366м"}.issubset(token_set("нитки для шитья 40/2 полиэстер 366м"))
assert token_set("0,8мм 0.8мм") == {"0.8мм"}
assert token_set_without_word_commas("красный, синий 0,8мм") == {"красный", "синий", "0.8мм"}
assert token_set_without_quotes('«Parker» "IM"') == {"parker", "im"}
assert candidate_intersections('красный, 0,8мм', 'красный 0.8мм') == {
    "current": ["0.8мм"],
    "without_word_commas": ["0.8мм", "красный"],
    "without_quotes": ["0.8мм"],
}
