def bubble_sort(origin_items, comp=lambda x, y: x > y):
    items = origin_items[:]
    for i in range(len(items)-1):
        swapped = False
        for j in range(i, len(items)-1-i):
            if comp(items[j], items[j+1]):
                items[j], items[j+1] = items[j+1], items[j]
                swapped = True
        if swapped:
            swapped = False
            for j in range(len(items)-2-i, i, -1):
                if comp(items[j-1], items[j]):
                    items[j], items[j-1]=items[j-1], items[j]
                    swapped = True
        if not swapped:
            break
    return items

def merge(items1, items2, comp=lambda x, y: x <= y):
    items = []
    index1, index2 = 0, 0
    while index1 < len(items1) and index2< len(items2):
        if comp(items1[index1], items2[index2]):
            items.append(items1[index1])
            index1 +=1
        else:
            items.append(items2[index2])
            index2 +=1
    items += items1[index1:]
    items += items2[index2]
    return items

items = [1, 4, 5, 6, 7, 9, 13, 3, 2, 7, 4]

items1 = [34, 5, 4]


prices = {
    'AAPL': 191.88,
    'GOOG': 1186.96,
    'IBM': 149.24,
    'ORCL': 48.44,
    'ACN': 166.89,
    'FB': 208.09,
    'SYMC': 21.29
}


price2 = {key: value for key, value in prices.items() if value >100}
print(price2)

names = ['关羽', '张飞', '赵云', '马超', '黄忠']
courses = ['语文', '数学', '英语']