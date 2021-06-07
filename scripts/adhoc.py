# import shelve
# import json
import time
# from tqdm.auto import tqdm
import numpy as np
import math

def find_factor(n):
    for i in range(2, int(math.sqrt(n))+1):
        if not i%n:
            return i
    return n


start = time.time()
print(find_factor(1073741741))
print('time: ', time.time() - start)




# # data_path = 'data/2^16.json'
# # with open(data_path, 'r') as f:
#     # data = json.load(f)

# # print('Finished loading data!')

# s = shelve.open('test')
# # print('Ive opened!')
# # for k, v in tqdm(data['train'].items(), total=len(data['train'])):
# #     s[k] = v
# #     time.sleep(0.0000001)
# # s.close()
# print(s['1234'])
# print(s['12345'])
# # s['train'] = data['train']
# # s['test'] = data['test']
# # s.close()

# # for k, v in data['train'].items():
# #     print(k)
# #     print(v)
# #     break
