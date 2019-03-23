test_data = $5
output = $6
python logistic_model/extract_feature.py test $test_data
python logistic_model/test.py $6