
def merge(file1, file2):
    f1 = open(file1, 'a+', encoding='utf-8')
    with open(file2, 'r', encoding='utf-8') as f2:
        f1.write('\n')
        for i in f2:
            f1.write(i)

def consis_node_edge(path):

    #获取txt文件中所有的结点序号
    node_num_list = []
    node_num_list_edg=[]
    file1='MG_datasets3/consis_newADG_edg.txt'
    file2='MG_datasets3/consis_newADG_node.txt'
    with open(file1, 'w', encoding='UTF-8') as f1:
        f1.write('')
    with open(file2, 'w', encoding='UTF-8') as f1:
        f1.write('')
    f1 = open(path, 'r', encoding='utf-8')
    for line1 in f1:

        if line1.find('->') == -1:

            n3 = line1.find(':')
            print(type(line1))
            # line1 = line1.replace(':', '[')
            # print(line1)
            i3 = line1[0:n3]
            node_num_list.append(i3)



    # 获取边的两端结点序号
    lines_seen = set()
    f = open(path, 'r', encoding='utf-8')
    for line in f:

        if line.find('->') != -1:
            n1=line.find('-')
            i1=line[0:n1]
            n2=line.find('>')
            n2=n2+1
            # n4=line.find('[')
            i2=line[n2:len(line)-1]

            #判断边的两端数字是否在结点的标号中找到
            if i1 in node_num_list and i2 in node_num_list:
                if i1 not in lines_seen:
                    lines_seen.add(i1)
                if i2 not in lines_seen:
                    lines_seen.add(i2)
                #如果找到，将边的信息存储到一个文件中
                with open(file1,'a',encoding='UTF-8') as f2:
                    f2.write(line)

    # exit()
    # print(lines_seen)
    f3 = open(path, 'r', encoding='utf-8')
    for line2 in f3:

        if line2.find('->') == -1:
            n5 = line2.find(':')
            i5 = line2[0:n5]
            print("标号：",i5)
            if i5 in lines_seen:
                print("程序是否执行到了这个地方")
                with open(file2,'a',encoding='UTF-8') as f4:
                    f4.write(line2)
    merge(file2,file1)

path1='MG_datasets3/newADG.txt'
# with open('af_newADG_1000.txt', 'w', encoding='UTF-8') as f1:
#     f1.write('ADG\n')
consis_node_edge(path1)