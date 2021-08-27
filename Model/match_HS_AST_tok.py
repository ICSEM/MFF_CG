# dict1={}
# dict2={}
# list=[]
# dict1['u'] = 1
# dict1['v'] = 1
# dict1['w'] = 1
# dict2['x'] = 1
# dict2['y'] = 1
# dict2['z'] = 1
# list.append(dict1)
# list.append(dict2)
# print(dict1)
# print(list)

def match_Tok_AST2():
    AST_num = 532
    several_AST_emb=[]

    # list_match_pos = []
    list_AST = []
    # list_leaf_pos = []
    for i1 in range(AST_num):
        #dic_AST_tok字典中的key表示源代码中的token位置，value表示对应AST中叶子结点的标号
        dic_match = {}
        dic_AST_tok = {}
        i1_str = str(i1)
        file_name = i1_str + '.txt'
        AST_path = 'HS_AST/' + file_name
        with open(AST_path, 'r', encoding='UTF-8') as f:
            lines_AST = f.readlines()
        with open('HS_datasets/newHS_code_598.out', 'r', encoding='UTF-8-sig') as f1:
            lines_tok = f1.readlines()

        # tok_embedding = self.tgt_emb3(dec_inputs[][])
        for line in lines_AST:
            if line.find('[ter_') != -1 and line.find('[ter_None') == -1:
                line1 = line.split('[ter_')
                # 将AST中的叶子结点的标号和相应的信息提取出来
                leaf_order = line1[0]
                leaf_info = line1[1]
                leaf_info = leaf_info.split('\n')
                leaf_info = leaf_info[0]
                # print(line1)
                # print(leaf_info)
                # print(lines_tok[i1])
                # exit()
                # 找出AST中叶子结点信息中的每个tok对应于源代码中的位置
                # if lines_tok[i1].find(leaf_info) != -1:

                leaf_info_toks = leaf_info.split()
                for line_info_tok in leaf_info_toks:

                    # leaf_tok_startpos = lines_tok[i1].find(leaf_info)
                    # leaf_tok_endpos = leaf_tok_startpos + len(leaf_info)

                    # print(lines_tok)
                    # print(line_info_tok)
                    #对一行源代码中的token进行分割
                    lines_tok_sp = lines_tok[i1].split()
                    if line_info_tok not in dic_match:

                        # print(lines_tok_sp)
                        # print(line_info_tok)
                        tok_order1 = 0
                        for lines_tok_sp1 in lines_tok_sp:
                            if lines_tok_sp1.find(line_info_tok) != -1:
                                pos = tok_order1
                                break
                            tok_order1 = tok_order1 + 1
                        # print(lines_tok)
                        dic_AST_tok[pos] = leaf_order
                        dic_match[line_info_tok] = pos
                        # print("叶子结点中的" + line_info_tok + "位置为：" + str(pos))
                        # exit()
                    else:

                        pos1 = dic_match[line_info_tok]
                        pos1 = pos1+1
                        length_tok = len(lines_tok_sp)
                        lines_tok1 = lines_tok_sp[pos1:length_tok]
                        tok_order2 = 0
                        pos2 = 0
                        for lines_tok_sp1 in lines_tok1:
                            if lines_tok_sp1.find(line_info_tok) != -1:
                                pos2 = tok_order2
                                break
                            tok_order2 = tok_order2 + 1

                        # print(lines_tok1)
                        # pos2 = lines_tok1.index(line_info_tok)
                        pos3 = pos1 + pos2
                        # print("叶子结点中的"+line_info_tok+"位置为："+str(pos3))
                        dic_AST_tok[pos3] = leaf_order
                        dic_match[line_info_tok] = pos3
        list_AST.append([])
        list_AST[i1].append(dic_AST_tok)
    # for k,v in list_AST[0][0].items():
    #     print("key:",k)
    #     print("value:",v)
    # print(list_AST[0][0])
    # print(list_AST[0][0][128])
    print(len(list_AST))
    return list_AST

# match_Tok_AST1()