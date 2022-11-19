with open("./res.txt", "r") as f:
    ls = f.read().split('\n')
for line in ls:
    data, anno = line.rsplit(' ', 1)
    tp = eval(data)
    tp = [[-i[0], i[1]] for i in tp]
    print(tp,anno)

