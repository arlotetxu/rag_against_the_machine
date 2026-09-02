import ast
from icecream import ic

data = ast.parse("indexing/indexer.py")
ic(ast.dump(data, indent=4))
ic(type(data))