import json
def read_from_dat(fn):
    # format:
    # "padding":0
    # "~":1
    # "en-us_oy2":38
    # "punc>":39
    # "punc?":69
    # "en-us_aw2":112
    # "punc：":4486
    res = {}
    with open(fn, 'r') as f:
        for line in f:
            line = line.strip()
            # : may in key
            all= line.split(':')
            key = ':'.join(all[:-1])
            val = all[-1]
            key = key.strip('"')
            val = int(val)
            res[val] = key
    return res

if __name__ == '__main__':
    fn = 'phonememap.dat'
    mappping = read_from_dat(fn)
    phone_set = list()
    phone_num = 5000
    placeholder_idx = 0
    for idx in range(phone_num):
        if idx in mappping:
            phone_set.append(mappping[idx])
        else:
            phone_set.append(f'placeholder_{placeholder_idx}')
            placeholder_idx += 1


    with open('phone_set.json', 'w') as f:
        json.dump(phone_set, f)