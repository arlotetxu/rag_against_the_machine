import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from icecream import ic

ic.includeContext=True

def get_py_code(py_path: str):
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    try:
        with open(py_path, mode='r', encoding='utf8') as fd:
            data = fd.read()
        lines = data.splitlines()
        long_lines = [len(line) for line in lines]
        # ic(long_lines)
        data_bytes = data.encode('utf8')
        tree = parser.parse(data_bytes)
    except Exception as e:
        print("ERROR FLAG!!")
        print(e)

    # ic(tree)
    # ic(type(tree))

    root = tree.root_node
    # ic(root)
    # ic(type(root))
    # ic(root.type)
    # ic(root.start_point, root.end_point)
    # ic(root.start_byte, root.end_byte)


    childrens = root.children
    # ic(childrens)
    # ic(type(childrens))

    children_types = []
    last_char = 0
    for children in childrens:
        if children.type not in children_types:
            children_types.append(children.type)
        ic(children.type)
        ic(children.byte_range)  # inicio y fin de cada children en caracteres
        ic(children.start_byte)  # posicion de inicio del cada children
        print(children.text)  # bloque de codigo
    ic(children_types)




get_py_code("indexing/indexer.py")
