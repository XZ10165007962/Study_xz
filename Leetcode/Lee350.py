#给定两个数组，编写一个函数来计算它们的交集。
#利用字典统计nums1和nums2每个元素的数量，然后取得两个字典相同键的最小值，返回结果。实际上是对原数据做了hash映射
from collections import defaultdict


def intersect( nums1, nums2):
    dct1 = defaultdict(int)
    for i in nums1:
        dct1[i] += 1
    dct2 = defaultdict(int)
    for i in nums2:
        dct2[i] += 1
    dct3 = {i: min(dct1[i], dct2[i]) for i in set(dct1) & set(dct2)}
    return sum([[key] * val for key, val in dct3.items()], [])

print(intersect([1,2,2,1],[2,1,1]))

'''a = {1:1,2:2}
b = {3:3,4:4,5:6,2:2}
print(a,b)
print(set(a),set(b))
print(set(a)&set(b))
print(set(a)|set(b))
print(set(a)-set(b))'''