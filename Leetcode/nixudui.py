
def InversionNum(lst):
    if len(lst) == 1:
        return lst,0
    else:
        n = len(lst) // 2
        lst1,count1 = InversionNum(lst[0:n])
        lst2,count2 = InversionNum(lst[n:len(lst)])
        lst,count = Count(lst1,lst2,0)
        print('lst',lst)
        print('lst1',lst1)
        print('lst2',lst2)
        return lst,count1+count2+count

def Count(lst1,lst2,count):
    i = 0
    j = 0
    res = []
    while i < len(lst1) and j < len(lst2):
        if lst1[i] <= lst2[j]:
            res.append(lst1[i])
            i += 1
        else:
            res.append(lst2[j])
            count += len(lst1) - i
            j += 1
    res += lst2[j:]
    res  += lst1[i:]

    return res,count


print(InversionNum([1, 2, 3, 4, 6, 7, 8, 5]))