from sample_name_attribute_overlap import token_set


assert token_set("A4 xgp5600p 15/4 (0,23)\n0.1") == {"a4", "xgp5600p", "15/4", "0.23", "0.1"}
assert {"40/2", "366м"}.issubset(token_set("нитки для шитья 40/2 полиэстер 366м"))
assert token_set("0,8мм 0.8мм") == {"0.8мм"}
