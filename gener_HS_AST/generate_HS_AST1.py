from collections import deque
from treelib import Tree,Node
from io import StringIO
import ast

def walk_dfs(node):
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extendleft(reversed( [i for i in ast.iter_child_nodes(node)]))
        yield node

class global_tree():
    prefix = 1
    codeNode_id = []
    codeNode_id_temp = []
    test = deque([])
    tree_ast = Tree()
    tree_ast.create_node(tag='root',identifier='root')
    @classmethod
    def clear_msg(cls):
        cls.prefix = 1
        cls.codeNode_id.clear()
        cls.codeNode_id_temp.clear()
        cls.test.clear()
    @classmethod
    def clear_tree(cls):
        cls.prefix = 1
        cls.codeNode_id.clear()
        cls.codeNode_id_temp.clear()
        cls.test.clear()
        cls.tree_ast = Tree()
        cls.tree_ast.create_node(tag='root',identifier='root')

def to_treelib(codeNode,codeNode_id):
    def isEmpty(codeNode):
        fields_sum = sum(1 for _ in ast.iter_fields(codeNode))
        return [False,True][fields_sum==0]
    def class_name(codeNode):
        return codeNode.__class__.__name__
    def str_to_Node(str):
        tag = str
        identifier = global_tree.prefix.__str__() +'_'+ tag
        temp_node = Node(tag=tag,identifier=identifier)
        global_tree.prefix+=1
        return temp_node
    def list_to_tree(root,list):
        tree = Tree()
        root_node = str_to_Node(root)
        tree.add_node(root_node)
        for each in list:
            if each.startswith('flag_'):
                each =each[len('flag_'):]
                child_node = str_to_Node(each)
                tree.add_node(child_node,tree.root)
                global_tree.codeNode_id.append(child_node.identifier)
            else:
                child_node = str_to_Node(each)
                tree.add_node(child_node,tree.root)
        return tree
    def codeNode_to_tree(id):
        tree = Tree()
        tag = id.split('_')[1]
        identifier = id
        tree.create_node(tag=tag,identifier=identifier)
        return tree
    if isEmpty(codeNode):
        return
    tree_local_list = []#存放子树
    tree_codeNode = codeNode_to_tree(codeNode_id)

    for field in ast.iter_fields(codeNode):

        node_list = []
        if isinstance(field[1],ast.AST):
            if  isEmpty(field[1]):
                continue
            node_list.append('flag_'+class_name(field[1]))

        if isinstance(field[1],list):
            if not field[1]:
                continue
            for each in field[1]:
                if isEmpty(each):continue
                node_list.append('flag_'+class_name(each))

        if isinstance(field[1],(str,int)):
            id = 'ter_'+str(field[1])
            node_list.append(id)
        if field[1] is None:#is None
            id = 'ter_None'
            node_list.append(id)
        if len(node_list):
            tree_local =list_to_tree(field[0],node_list)
            tree_local_list.append(tree_local)

    if len(tree_local_list):
        for each in tree_local_list:
            tree_codeNode.paste(tree_codeNode.root,each)

    return tree_codeNode

def create_tree(ast_model):
    for n,i in enumerate(walk_dfs(ast_model)):
        if n==0:
            global_tree.test.append(str(n)+'_'+i.__class__.__name__)
        if len(global_tree.test)==0:
            continue
        if i.__class__.__name__ not in global_tree.test[0] :
            continue
        fields_sum = sum(1 for _ in ast.iter_fields(i))
        if fields_sum == 0:continue
        global_tree.codeNode_id_temp.append(i.__class__.__name__)
        codeNode_root = global_tree.test.popleft()
        tree_loc = to_treelib(i,codeNode_root)
        global_tree.test.extendleft(reversed(global_tree.codeNode_id))
        global_tree.codeNode_id.clear()
        if n==0:
            global_tree.tree_ast.paste(global_tree.tree_ast.root,tree_loc)
            continue
        try:
            global_tree.tree_ast.merge(tree_loc.root,tree_loc)
        except AttributeError:
            print(codeNode_root)
            print(i)
            print(global_tree.codeNode_id)
            for j in ast.iter_fields(i):
                print(j)
    return None

def show_tree():
    import os
    global_tree.tree_ast.to_graphviz(filename='tree.gv', shape='ellipse')
    os.system('dot -Tpng tree.gv -o tree.gv.png')
    os.startfile('tree.gv.png')


def create_file():
    mz_node_list = []
    mz_edge_list = []
    mz_tree = global_tree.tree_ast
    for i in mz_tree.expand_tree():
        if 'ter_' in i:
            # temp = i.replace('ter_','')
            # mz_node_list.append(temp.replace('_','[',1))
            mz_node_list.append(i.replace('_', '[', 1))
        else:
            mz_node_list.append(i.replace('_','[',1))
        if mz_tree.parent(i):
            i_parent = mz_tree.parent(i).identifier.split('_')[0]
            mz_edge_list.append(i_parent+'->'+i.split('_')[0])
    return mz_node_list[1:],mz_edge_list[1:]

def del_Name():
    temp_tree = global_tree.tree_ast
    for i in temp_tree.expand_tree():
    #print(temp_tree.children(i))
        if len(temp_tree.children(i))==1:
            try:
                if temp_tree.children(i)[0].tag =='Name':
                    leaf = temp_tree.subtree(i).leaves()[0]
                    # print(leaf)
                    temp_tree.remove_node(temp_tree.children(i)[0].identifier)
                    # print(temp_tree.children(i))
                    # print(i)
                    temp_tree.add_node(leaf,parent=i)
            except Exception as e :
                print(e)

                global_tree.tree_ast.show(idhidden=False)
    # return temp_tree




buf = StringIO()
n = 0
error = dict()
success = []
with open('HS_datasets/HS_code_599.out', 'r', encoding='utf-8') as f:
    lines = f.readlines()
# print(lines[0])
for n,line in enumerate(lines):
    code_list = line.replace('§','\n')
    try:
        ast_model = ast.parse(code_list)
        create_tree(ast_model)
        del_Name()
        node,edge = create_file()

        node = sorted(node, key=lambda l: int(l.split('[')[0]))
        buf.write('AST\n')
        for i in node:
            buf.write(i)
            buf.write('\n')
        for i in edge:
            buf.write(i)
            buf.write('\n')
        success.append(n)
        print('finished',n,line.split('§')[0])
        global_tree.clear_tree()
    except SyntaxError:
        print(n,'SyntaxError',line.split()[1])
        error[n] = line.split()[1]+'....'
    finally:
        global_tree.clear_tree()

with open('HS_AST.txt', 'w') as f:
    f.write(buf.getvalue())
buf.close()
print('源文件行数:',len(lines))
print('解析成功条数:',len(success))
print('失败程序')
print(error)
with open('HS_AST.txt','r') as f:
    line = f.readlines()
n=0
for i in line:
    if i.startswith('AST'):
        n+=1
print(n)