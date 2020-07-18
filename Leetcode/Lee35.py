
'''
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

'''

def searchInsert1( nums, target):
    for i in range(0,nums[-1]+1):
        if target in nums:
            print('--')
            return nums.index(target)
        elif (target + i) in nums:
            print('---')
            return nums.index((target + i))
        elif (target - i) in nums:
            print('----')
            return nums.index((target - i))+1

def searchInsert( nums, target):
    n = len(nums)
    left = 0
    right = n
    while left < right:
        mid = (left + right) // 2
        val = nums[mid]
        if val > target:
            right = mid-1
        elif val < target:
            left = mid+1
        else:
            return mid
    return left

print(searchInsert([1,3,5,6], 7))



