cat news.2019.en.shuffled.deduped | awk 'NR%1000==1' > valid.en 
cat news.2019.en.shuffled.deduped | awk 'NR%1000==2' > test.en
cat news.2019.en.shuffled.deduped | awk 'NR%1000!=1&&NR%1000!=2' > train.en