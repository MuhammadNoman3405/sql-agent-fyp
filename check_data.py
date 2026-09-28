import json

bird = json.load(open('data/bird/dev/dev_20240627/dev.json'))
print('BIRD dev examples:', len(bird))
print('Sample:', bird[0]['question'], '->', bird[0]['SQL'])

spider = json.load(open('data/spider/spider_data/spider_data/dev.json'))
print('Spider dev examples:', len(spider))
print('Sample:', spider[0]['question'], '->', spider[0]['query'])