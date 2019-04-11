wget 'https://www.dropbox.com/s/0nqf7olne53oxr9/model.zip?dl=1' -O model.zip
unzip model.zip
python extract_feature.py test $1
python test.py $2 modellist

