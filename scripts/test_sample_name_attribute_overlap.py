from sample_name_attribute_overlap import token_set


assert token_set("A4 xgp5600p 15/4 (0,23)\n0.1") == {"a4", "xgp5600p", "15/4", "0.23", "0.1"}
