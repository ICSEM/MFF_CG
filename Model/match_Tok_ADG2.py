def match_Tok_ADG2():
    dic1 = {}
    dic_match = {}
    ADG_node_order = 0
    list1 = []
    with open('MG_datasets3/preADG2.txt', 'r', encoding='UTF-8') as f1:
        lines_ADG = f1.readlines()
    with open('MG_datasets3/preMG_code_1000.txt', 'r', encoding='UTF-8') as f2:
        lines_Tok = f2.readlines()

    for i in range(0,len(lines_Tok)):
        list1.append(lines_Tok[i])
    #主要以ADG中的结点个数为基准
    for line in lines_ADG:
        if line.find('[') != -1:
            line1 = line.split(':')
            line2 = line1[1].split('[')
            line3 = line2[1].split()
            ADG_node_info = line3[2]

            ADG_tok_order = line2[0]
            # print(ADG_tok_order)
            j = int(ADG_tok_order)
            # print("输出的源代码：",list1[j])

            str_line = list1[j]
            str_tok = str_line.split()

            # print("当前运行到的行数为：",ADG_node_order+1)
            if ADG_tok_order not in dic1:
                over_start_pos = str_tok.index('@Override')

                while str_tok[over_start_pos + 2] != ADG_node_info:
                    over_start_pos = over_start_pos + 1
                    str_tok3 = str_tok[over_start_pos:]

                    gap = str_tok3.index('@Override')
                    over_start_pos = over_start_pos + gap
                dic1[ADG_tok_order] = over_start_pos
                # print(dic1)
                # exit()
            else:
                pos = dic1[ADG_tok_order]
                pos = pos + 1
                str_tok1 = str_tok[pos:]
                over_pos_gap = str_tok1.index('@Override')
                over_start_pos = pos + over_pos_gap
                dic1[ADG_tok_order] = over_start_pos
                # print(dic1)
            str_tok2 = str_tok[over_start_pos:]
            over_gap = str_tok2.index('}')
            over_end_pos = over_start_pos +over_gap
            # print(str_tok)
            # print(over_start_pos)
            # print(over_end_pos)
            # print(type(ADG_tok_order))
            tok_match_str = ADG_tok_order+'+'+str(over_start_pos)+'+'+str(over_end_pos)
            dic_match[ADG_node_order] = tok_match_str
            ADG_node_order = ADG_node_order + 1
            # exit()
    # print(dic_match)
    return dic_match
            # ADG_node_order = ADG_node_order + 1

match_Tok_ADG2()