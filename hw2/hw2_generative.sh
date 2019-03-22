test_data = $5
output = $6
python gen_model/extract_feature.py test $test_data
python gen_model/test_gen.py $6