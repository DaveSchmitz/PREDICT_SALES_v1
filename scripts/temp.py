# lst = input().split(",")
lst = ['red','orange','yellow','green','blue','purple','pink','brown','black','white','gray','violet','indigo','maroon','navy']

# Write your code below
print(lst[1::3])
print(lst[5::-1])
middle = (len(lst) // 2)
print(lst[middle::2])