'''
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。'''

def twoSum(nums,target):

    indexs = []
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if nums[i] + nums[j] == target:
                if i not in indexs and j not in indexs:
                    indexs.extend([i,j])
                    break
    return indexs

'''class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic={}
        for k,v in enumerate(nums):
            if target-v in dic: #写之前判断，避免了重复元素的覆盖
                return [dic[target-v],k]
            dic[v]=k'''

#直接用target 减去 取出的数字，看结果有没有在数组里
class Solution:
	def twoSum(self,nums,target):
		n = len(nums)
		for x in range(n):
			a = target - nums[x]
			if a in nums: # 判断a有没有在nums数组里
				y = nums.index(a) # 有的话，那么用index获取到该数字的下标
				if x == y:
					continue # 同样的数字不能重复用，所以这里如果是一样的数字，那么就不满足条件，跳过
				else: # 否则就返回结果
					return x,y
					break
			else:
				continue # 上面的条件都不满足就跳过，进行下一次循环

#原先的数组转化成字典，通过字典去查询速度就会快很多。下面的代码我变更了顺序，好理解多了，速度也快了一些。

def twoSum1(nums,target):
	d = {}
	n = len(nums)
	for x in range(n):
		if target - nums[x] in d:
			return d[target-nums[x]],x
		else:
			d[nums[x]] = x;print(d)

print(twoSum1( [2,2,7,11,15],9))