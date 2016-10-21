#!/usr/bin/env python

import ast

import histogrammar
import histogrammar.hgawk_grammar as grammar

expectFunction = set([
    # fundamental aggregators
    ("Count", 0),
    ("Count", "quantity"),
    ("Sum", 0),
    ("Sum", "quantity"),
    ("Average", 0),
    ("Average", "quantity"),
    ("Deviate", 0),
    ("Deviate", "quantity"),
    ("Minimize", 0),
    ("Minimize", "quantity"),
    ("Maximize", 0),
    ("Maximize", "quantity"),
    ("Bag", 0),
    ("Bag", "quantity"),
    ("Bin", 3),
    ("Bin", "quantity"),
    ("SparselyBin", 1),
    ("SparselyBin", "quantity"),
    ("CentrallyBin", 1),
    ("CentrallyBin", "quantity"),
    ("IrregularlyBin", 1),
    ("IrregularlyBin", "quantity"),
    ("Categorize", 0),
    ("Categorize", "quantity"),
    ("Fraction", 0),
    ("Fraction", "quantity"),
    ("Stack", 1),
    ("Stack", "quantity"),
    ("Select", 0),
    ("Select", "quantity"),

    # convenience functions
    ("Histogram", 3),
    ("Histogram", 4),
    ("Histogram", "quantity"),
    ("Histogram", "selection"),
    ("SparselyHistogram", 1),
    ("SparselyHistogram", 2),
    ("SparselyHistogram", "quantity"),
    ("SparselyHistogram", "selection"),
    ("Profile", 3),
    ("Profile", 4),
    ("Profile", 5),
    ("Profile", "binnedQuantity"),
    ("Profile", "averagedQuantity"),
    ("Profile", "selection"),
    ("SparselyProfile", 1),
    ("SparselyProfile", 2),
    ("SparselyProfile", 3),
    ("SparselyProfile", "binnedQuantity"),
    ("SparselyProfile", "averagedQuantity"),
    ("SparselyProfile", "selection"),
    ("ProfileErr", 3),
    ("ProfileErr", 4),
    ("ProfileErr", 5),
    ("ProfileErr", "binnedQuantity"),
    ("ProfileErr", "averagedQuantity"),
    ("ProfileErr", "selection"),
    ("SparselyProfileErr", 1),
    ("SparselyProfileErr", 2),
    ("SparselyProfileErr", 3),
    ("SparselyProfileErr", "binnedQuantity"),
    ("SparselyProfileErr", "averagedQuantity"),
    ("SparselyProfileErr", "selection"),
    ("TwoDimensionallyHistogram", 3),
    ("TwoDimensionallyHistogram", 7),
    ("TwoDimensionallyHistogram", 8),
    ("TwoDimensionallyHistogram", "xquantity"),
    ("TwoDimensionallyHistogram", "yquantity"),
    ("TwoDimensionallyHistogram", "selection"),
    ("TwoDimensionallySparselyHistogram", 1),
    ("TwoDimensionallySparselyHistogram", 3),
    ("TwoDimensionallySparselyHistogram", 4),
    ("TwoDimensionallySparselyHistogram", "xquantity"),
    ("TwoDimensionallySparselyHistogram", "yquantity"),
    ("TwoDimensionallySparselyHistogram", "selection"),
    ])

def highestDollar(node):
    if isinstance(node, grammar.DollarNumber):
        return node.n
    elif isinstance(node, ast.AST):
        return max([0] + [highestDollar(getattr(node, field)) for field in node._fields])
    elif isinstance(node, list):
        return max([0] + [highestDollar(x) for x in node])
    else:
        return 0

def dollarToArg(node):
    if isinstance(node, grammar.DollarNumber):
        out = ast.Subscript(ast.Name("$args", ast.Load()), ast.Index(ast.Num(node.n - 1)), ast.Load())
        # ASTs need to have line numbers to be compilable by Python
        out.lineno,               out.col_offset               = node.lineno, node.col_offset
        out.value.lineno,         out.value.col_offset         = node.lineno, node.col_offset
        out.value.ctx.lineno,     out.value.ctx.col_offset     = node.lineno, node.col_offset
        out.slice.lineno,         out.slice.col_offset         = node.lineno, node.col_offset
        out.slice.value.lineno,   out.slice.value.col_offset   = node.lineno, node.col_offset
        out.ctx.lineno,           out.ctx.col_offset           = node.lineno, node.col_offset
        return out
    elif isinstance(node, ast.AST):
        for field in node._fields:
            setattr(node, field, dollarToArg(getattr(node, field)))
        return node
    elif isinstance(node, list):
        for i, x in node:
            node[i] = dollarToArg(x)
        return node
    else:
        return node
            
def functionalize(node):
    # changes node in-place, but returns it anyway
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        for i, x in enumerate(node.args):
            if (node.func.id, i) in expectFunction:
                numargs = highestDollar(x)
                if numargs > 0:
                    # the parameter name "$args" can't conflict with any valid Python names
                    out = ast.Lambda(ast.arguments([ast.Name("$args", ast.Param())], None, None, []), dollarToArg(x))
                    out.lineno,                  out.col_offset                  = x.lineno, x.col_offset
                    out.args.lineno,             out.args.col_offset             = x.lineno, x.col_offset
                    out.args.args[0].lineno,     out.args.args[0].col_offset     = x.lineno, x.col_offset
                    out.args.args[0].ctx.lineno, out.args.args[0].ctx.col_offset = x.lineno, x.col_offset
                    node.args[i] = out

    for field in node._fields:
        subnode = getattr(node, field)
        if isinstance(subnode, ast.AST):
            functionalize(subnode)
        elif isinstance(subnode, list):
            for x in subnode:
                if isinstance(x, ast.AST):
                    functionalize(x)

    return node

def assignLast(node):
    # also changes node in-place and returns it anyway
    assert isinstance(node, ast.Module)
    assert len(node.body) > 0
    last = node.body[-1]
    if not isinstance(last, ast.Expr):
        raise SyntaxError("last line must be an expression (the aggregator)")
    # the variable name "$h" can't conflict with any valid Python names
    out = ast.Assign([ast.Name("$h", ast.Store())], last.value)
    out.lineno,                out.col_offset                = last.lineno, last.col_offset
    out.targets[0].lineno,     out.targets[0].col_offset     = last.lineno, last.col_offset
    out.targets[0].ctx.lineno, out.targets[0].ctx.col_offset = last.lineno, last.col_offset
    node.body[-1] = out
    return node

def executeWithDollars(source):
    namespace = {}
    namespace.update(histogrammar.__dict__)
    exec(compile(assignLast(functionalize(grammar.parse(source))), "<string>", "exec")) in namespace
    return namespace["$h"]
    
if __name__ == "__main__":
    h = executeWithDollars("Bin(10, -5.0, 5.0, $1 + $2)")
    h.fill((1.0, 2.0))
    h.fill((-2.0, -1.0))
    print h.toJsonString()
