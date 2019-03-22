test_data = $5
output = $6
python best_model/extract_feature.py test $test_data
python best_model/test_keras.py $6