import gzip
from collections import defaultdict
import argparse
import json


def parse(path):
    g = open(path, 'r', encoding='utf-8')
    for l in g:
        yield json.loads(l)

countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

parser = argparse.ArgumentParser(description='Generate dataset txt with item names')
parser.add_argument('--dataset', default='Musical_Instruments', help='Dataset name, e.g., Beauty, Electronics')
args = parser.parse_args()
dataset_name = args.dataset
# Load meta file to get asin -> title mapping
asin_to_title = dict()
meta_path = 'meta_' + dataset_name + '.json'
try:
    for m in parse(meta_path):
        if 'asin' in m:
            asin = m['asin']
            title = m.get('title', '')
            try:
                title_clean = (title if isinstance(title, str) else str(title)).replace('\n', ' ').strip()
            except:
                title_clean = ''
            asin_to_title[asin] = title_clean
except:
    asin_to_title = dict()

# Count interactions directly from source json; no intermediate file
for l in parse('reviews_' + dataset_name + '_5.json'):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    countU[rev] += 1
    countP[asin] += 1

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
# Record the mapping from itemid to item title
itemid_to_name = dict()
User = dict()
for l in parse('reviews_' + dataset_name + '_5.json'):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
        # When first seeing an asin, record its title (fallback to ASIN if meta missing to ensure the 3rd column is non-empty)
        itemid_to_name[itemid] = asin_to_title.get(asin, asin)
    User[userid].append([time, itemid])
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

f = open(dataset_name + '.txt', 'w')
for user in User.keys():
    for i in User[user]:
        name = itemid_to_name.get(i[1], '')
        f.write('%d %d %s\n' % (user, i[1], name))
f.close()
