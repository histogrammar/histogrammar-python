#!/usr/bin/env python
# generated at 2016-10-20T23:27:42 by "python
# generate-grammars/python-awk/python2_hgawk.g
# generate-grammars/python-awk/python2_actions.py
# histogrammar.python23_lex histogrammar/hgawk_grammar.py"

import re
import ast
import inspect

from histogrammar.pycparser.ply import yacc
from histogrammar.python23_lex import PythonLexer


class DollarNumber(ast.expr):
    _fields = ("n",)

    def __init__(self, n, **kwds):
        self.n = n
        self.__dict__.update(kwds)


def inherit_lineno(p0, px, alt=True):
    if isinstance(px, dict):
        p0.lineno = px["lineno"]
        p0.col_offset = px["col_offset"]
    else:
        if alt and hasattr(px, "alt"):
            p0.lineno = px.alt["lineno"]
            p0.col_offset = px.alt["col_offset"]
        else:
            p0.lineno = px.lineno
            p0.col_offset = px.col_offset


def ctx_to_store(obj, store=ast.Store):
    if isinstance(obj, list):
        for i, x in enumerate(obj):
            obj[i] = ctx_to_store(x, store)
        return obj
    elif isinstance(obj, (ast.Attribute, ast.Subscript)):
        obj.ctx = store()
        return obj
    elif isinstance(obj, ast.AST):
        for attrib in obj._fields:
            value = getattr(obj, attrib)
            if isinstance(value, ast.Load):
                setattr(obj, attrib, store())
            elif isinstance(value, ast.Param):
                setattr(obj, attrib, store())
            elif isinstance(value, list):
                for i, x in enumerate(value):
                    value[i] = ctx_to_store(x, store)
            elif isinstance(value, ast.AST):
                setattr(obj, attrib, ctx_to_store(value, store))
        return obj
    else:
        return obj


def iskeyword(x):
    return isinstance(x, ast.keyword)


def notkeyword(x):
    return not isinstance(x, ast.keyword)


def unwrap_left_associative(args, rule, alt=False):
    out = ast.BinOp(args[0], args[1], args[2], rule=rule)
    inherit_lineno(out, args[0])
    args = args[3:]
    while len(args) > 0:
        out = ast.BinOp(out, args[0], args[1], rule=rule)
        inherit_lineno(out, out.left)
        if alt:
            out.alt = {"lineno": out.lineno, "col_offset": out.col_offset}
            inherit_lineno(out, out.op)
        args = args[2:]
    return out


def unpack_trailer(atom, power_star):
    out = atom
    for trailer in power_star:
        if isinstance(trailer, ast.Call):
            trailer.func = out
            inherit_lineno(trailer, out)
            out = trailer
        elif isinstance(trailer, ast.Attribute):
            trailer.value = out
            inherit_lineno(trailer, out, alt=False)
            if hasattr(out, "alt"):
                trailer.alt = out.alt
            out = trailer
        elif isinstance(trailer, ast.Subscript):
            trailer.value = out
            inherit_lineno(trailer, out)
            out = trailer
        else:
            assert False
    return out


# file_input: (NEWLINE | stmt)* ENDMARKER
def p_file_input_1(p):
    '''file_input : ENDMARKER'''
    #                       1
    p[0] = ast.Module([], rule=inspect.currentframe().f_code.co_name, lineno=0, col_offset=0)


def p_file_input_2(p):
    '''file_input : file_input_star ENDMARKER'''
    #                             1         2
    p[0] = ast.Module(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1][0])


def p_file_input_star_1(p):
    '''file_input_star : NEWLINE'''
    #                          1
    p[0] = ast.Module([], rule=inspect.currentframe().f_code.co_name, lineno=0, col_offset=0)


def p_file_input_star_2(p):
    '''file_input_star : stmt'''
    #                       1
    p[0] = p[1]


def p_file_input_star_3(p):
    '''file_input_star : file_input_star NEWLINE'''
    #                                  1       2
    p[0] = ast.Module(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1][0])


def p_file_input_star_4(p):
    '''file_input_star : file_input_star stmt'''
    #                                  1    2
    p[0] = p[1] + p[2]

# decorator: '@' dotted_name [ '(' [arglist] ')' ] NEWLINE


def p_decorator_1(p):
    '''decorator : AT dotted_name NEWLINE'''
    #               1           2       3
    p[0] = p[2]
    p[0].alt = p[1][1]


def p_decorator_2(p):
    '''decorator : AT dotted_name LPAR RPAR NEWLINE'''
    #               1           2    3    4       5
    p[0] = ast.Call(p[2], [], [], None, None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1][1])


def p_decorator_3(p):
    '''decorator : AT dotted_name LPAR arglist RPAR NEWLINE'''
    #               1           2    3       4    5       6
    p[4].func = p[2]
    p[0] = p[4]
    inherit_lineno(p[0], p[2])
    p[0].alt = p[1][1]

# decorators: decorator+


def p_decorators(p):
    '''decorators : decorators_plus'''
    #                             1
    p[0] = p[1]


def p_decorators_plus_1(p):
    '''decorators_plus : decorator'''
    #                            1
    p[0] = [p[1]]


def p_decorators_plus_2(p):
    '''decorators_plus : decorators_plus decorator'''
    #                                  1         2
    p[0] = p[1] + [p[2]]

# decorated: decorators (classdef | funcdef)


def p_decorated_1(p):
    '''decorated : decorators classdef'''
    #                       1        2
    p[2].decorator_list = p[1]
    p[0] = p[2]
    inherit_lineno(p[0], p[1][0])


def p_decorated_2(p):
    '''decorated : decorators funcdef'''
    #                       1       2
    p[2].decorator_list = p[1]
    p[0] = p[2]
    inherit_lineno(p[0], p[1][0])

# funcdef: 'def' NAME parameters ':' suite


def p_funcdef(p):
    '''funcdef : DEF NAME parameters COLON suite'''
    #              1    2          3     4     5
    p[0] = ast.FunctionDef(p[2][0], p[3], p[5], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# parameters: '(' [varargslist] ')'


def p_parameters_1(p):
    '''parameters : LPAR RPAR'''
    #                  1    2
    p[0] = ast.arguments([], None, None, [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_parameters_2(p):
    '''parameters : LPAR varargslist RPAR'''
    #                  1           2    3
    p[0] = p[2]


def p_varargslist_1(p):
    '''varargslist : fpdef COMMA STAR NAME'''
    #                    1     2    3    4
    p[0] = ast.arguments([p[1]], p[4][0], None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_2(p):
    '''varargslist : fpdef COMMA STAR NAME COMMA DOUBLESTAR NAME'''
    #                    1     2    3    4     5          6    7
    p[0] = ast.arguments([p[1]], p[4][0], p[7][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_3(p):
    '''varargslist : fpdef COMMA DOUBLESTAR NAME'''
    #                    1     2          3    4
    p[0] = ast.arguments([p[1]], None, p[4][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_4(p):
    '''varargslist : fpdef'''
    #                    1
    p[0] = ast.arguments([p[1]], None, None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_5(p):
    '''varargslist : fpdef COMMA'''
    #                    1     2
    p[0] = ast.arguments([p[1]], None, None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_6(p):
    '''varargslist : fpdef varargslist_star COMMA STAR NAME'''
    #                    1                2     3    4    5
    p[2].args.insert(0, p[1])
    p[2].vararg = p[5][0]
    p[0] = p[2]


def p_varargslist_7(p):
    '''varargslist : fpdef varargslist_star COMMA STAR NAME COMMA DOUBLESTAR NAME'''
    #                    1                2     3    4    5     6          7    8
    p[2].args.insert(0, p[1])
    p[2].vararg = p[5][0]
    p[2].kwarg = p[8][0]
    p[0] = p[2]


def p_varargslist_8(p):
    '''varargslist : fpdef varargslist_star COMMA DOUBLESTAR NAME'''
    #                    1                2     3          4    5
    p[2].args.insert(0, p[1])
    p[2].kwarg = p[5][0]
    p[0] = p[2]


def p_varargslist_9(p):
    '''varargslist : fpdef varargslist_star'''
    #                    1                2
    p[2].args.insert(0, p[1])
    p[0] = p[2]


def p_varargslist_10(p):
    '''varargslist : fpdef varargslist_star COMMA'''
    #                    1                2     3
    p[2].args.insert(0, p[1])
    p[0] = p[2]


def p_varargslist_11(p):
    '''varargslist : fpdef EQUAL test COMMA STAR NAME'''
    #                    1     2    3     4    5    6
    p[0] = ast.arguments([p[1]], p[6][0], None, [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_12(p):
    '''varargslist : fpdef EQUAL test COMMA STAR NAME COMMA DOUBLESTAR NAME'''
    #                    1     2    3     4    5    6     7          8    9
    p[0] = ast.arguments([p[1]], p[6][0], p[9][0], [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_13(p):
    '''varargslist : fpdef EQUAL test COMMA DOUBLESTAR NAME'''
    #                    1     2    3     4          5    6
    p[0] = ast.arguments([p[1]], None, p[6][0], [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_14(p):
    '''varargslist : fpdef EQUAL test'''
    #                    1     2    3
    p[0] = ast.arguments([p[1]], None, None, [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_15(p):
    '''varargslist : fpdef EQUAL test COMMA'''
    #                    1     2    3     4
    p[0] = ast.arguments([p[1]], None, None, [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_varargslist_16(p):
    '''varargslist : fpdef EQUAL test varargslist_star COMMA STAR NAME'''
    #                    1     2    3                4     5    6    7
    p[4].args.insert(0, p[1])
    p[4].vararg = p[7][0]
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]


def p_varargslist_17(p):
    '''varargslist : fpdef EQUAL test varargslist_star COMMA STAR NAME COMMA DOUBLESTAR NAME'''
    #                    1     2    3                4     5    6    7     8          9   10
    p[4].args.insert(0, p[1])
    p[4].vararg = p[7][0]
    p[4].kwarg = p[10][0]
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]


def p_varargslist_18(p):
    '''varargslist : fpdef EQUAL test varargslist_star COMMA DOUBLESTAR NAME'''
    #                    1     2    3                4     5          6    7
    p[4].args.insert(0, p[1])
    p[4].kwarg = p[7][0]
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]


def p_varargslist_19(p):
    '''varargslist : fpdef EQUAL test varargslist_star'''
    #                    1     2    3                4
    p[4].args.insert(0, p[1])
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]


def p_varargslist_20(p):
    '''varargslist : fpdef EQUAL test varargslist_star COMMA'''
    #                    1     2    3                4     5
    p[4].args.insert(0, p[1])
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]


def p_varargslist_21(p):
    '''varargslist : STAR NAME'''
    #                   1    2
    p[0] = ast.arguments([], p[2][0], None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2][1])


def p_varargslist_22(p):
    '''varargslist : STAR NAME COMMA DOUBLESTAR NAME'''
    #                   1    2     3          4    5
    p[0] = ast.arguments([], p[2][0], p[5][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2][1])


def p_varargslist_23(p):
    '''varargslist : DOUBLESTAR NAME'''
    #                         1    2
    p[0] = ast.arguments([], None, p[2][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2][1])


def p_varargslist_star_1(p):
    '''varargslist_star : COMMA fpdef'''
    #                         1     2
    p[0] = ast.arguments([p[2]], None, None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2])


def p_varargslist_star_2(p):
    '''varargslist_star : COMMA fpdef EQUAL test'''
    #                         1     2     3    4
    p[0] = ast.arguments([p[2]], None, None, [p[4]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2])


def p_varargslist_star_3(p):
    '''varargslist_star : varargslist_star COMMA fpdef'''
    #                                    1     2     3
    p[1].args.append(p[3])
    p[0] = p[1]


def p_varargslist_star_4(p):
    '''varargslist_star : varargslist_star COMMA fpdef EQUAL test'''
    #                                    1     2     3     4    5
    p[1].args.append(p[3])
    p[1].defaults.append(p[5])
    p[0] = p[1]


def p_fpdef_1(p):
    '''fpdef : NAME'''
    #             1
    p[0] = ast.Name(p[1][0], ast.Param(), rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_fpdef_2(p):
    '''fpdef : LPAR fplist RPAR'''
    #             1      2    3
    if isinstance(p[2], ast.Tuple):
        p[2].paren = True
        ctx_to_store(p[2])
    p[0] = p[2]


def p_fplist_1(p):
    '''fplist : fpdef'''
    #               1
    p[0] = p[1]


def p_fplist_2(p):
    '''fplist : fpdef COMMA'''
    #               1     2
    p[0] = ast.Tuple([p[1]], ast.Param(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_fplist_3(p):
    '''fplist : fpdef fplist_star'''
    #               1           2
    p[2].elts.insert(0, p[1])
    p[0] = p[2]
    inherit_lineno(p[0], p[1])


def p_fplist_4(p):
    '''fplist : fpdef fplist_star COMMA'''
    #               1           2     3
    p[2].elts.insert(0, p[1])
    p[0] = p[2]
    inherit_lineno(p[0], p[1])


def p_fplist_star_1(p):
    '''fplist_star : COMMA fpdef'''
    #                    1     2
    p[0] = ast.Tuple([p[2]], ast.Param(), rule=inspect.currentframe().f_code.co_name, paren=False)


def p_fplist_star_2(p):
    '''fplist_star : fplist_star COMMA fpdef'''
    #                          1     2     3
    p[1].elts.append(p[3])
    p[0] = p[1]

# stmt: simple_stmt | compound_stmt


def p_stmt_1(p):
    '''stmt : simple_stmt'''
    #                   1
    p[0] = p[1]


def p_stmt_2(p):
    '''stmt : compound_stmt'''
    #                     1
    p[0] = p[1]


def p_simple_stmt_1(p):
    '''simple_stmt : small_stmt NEWLINE'''
    #                         1       2
    p[0] = [p[1]]


def p_simple_stmt_2(p):
    '''simple_stmt : small_stmt SEMI NEWLINE'''
    #                         1    2       3
    p[0] = [p[1]]


def p_simple_stmt_3(p):
    '''simple_stmt : small_stmt simple_stmt_star NEWLINE'''
    #                         1                2       3
    p[0] = [p[1]] + p[2]


def p_simple_stmt_4(p):
    '''simple_stmt : small_stmt simple_stmt_star SEMI NEWLINE'''
    #                         1                2    3       4
    p[0] = [p[1]] + p[2]


def p_simple_stmt_star_1(p):
    '''simple_stmt_star : SEMI small_stmt'''
    #                        1          2
    p[0] = [p[2]]


def p_simple_stmt_star_2(p):
    '''simple_stmt_star : simple_stmt_star SEMI small_stmt'''
    #                                    1    2          3
    p[0] = p[1] + [p[3]]


def p_small_stmt_1(p):
    '''small_stmt : expr_stmt'''
    #                       1
    p[0] = p[1]


def p_small_stmt_2(p):
    '''small_stmt : print_stmt'''
    #                        1
    p[0] = p[1]


def p_small_stmt_3(p):
    '''small_stmt : del_stmt'''
    #                      1
    p[0] = p[1]


def p_small_stmt_4(p):
    '''small_stmt : pass_stmt'''
    #                       1
    p[0] = p[1]


def p_small_stmt_5(p):
    '''small_stmt : flow_stmt'''
    #                       1
    p[0] = p[1]


def p_small_stmt_6(p):
    '''small_stmt : import_stmt'''
    #                         1
    p[0] = p[1]


def p_small_stmt_7(p):
    '''small_stmt : global_stmt'''
    #                         1
    p[0] = p[1]


def p_small_stmt_8(p):
    '''small_stmt : exec_stmt'''
    #                       1
    p[0] = p[1]


def p_small_stmt_9(p):
    '''small_stmt : assert_stmt'''
    #                         1
    p[0] = p[1]


def p_expr_stmt_1(p):
    '''expr_stmt : testlist augassign yield_expr'''
    #                     1         2          3
    ctx_to_store(p[1])
    p[0] = ast.AugAssign(p[1], p[2], p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_expr_stmt_2(p):
    '''expr_stmt : testlist augassign testlist'''
    #                     1         2        3
    ctx_to_store(p[1])
    p[0] = ast.AugAssign(p[1], p[2], p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_expr_stmt_3(p):
    '''expr_stmt : testlist'''
    #                     1
    p[0] = ast.Expr(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_expr_stmt_4(p):
    '''expr_stmt : testlist expr_stmt_star'''
    #                     1              2
    everything = [p[1]] + p[2]
    targets, value = everything[:-1], everything[-1]
    ctx_to_store(targets)
    p[0] = ast.Assign(targets, value, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], targets[0])


def p_expr_stmt_star_1(p):
    '''expr_stmt_star : EQUAL yield_expr'''
    #                       1          2
    p[0] = [p[2]]


def p_expr_stmt_star_2(p):
    '''expr_stmt_star : EQUAL testlist'''
    #                       1        2
    p[0] = [p[2]]


def p_expr_stmt_star_3(p):
    '''expr_stmt_star : expr_stmt_star EQUAL yield_expr'''
    #                                1     2          3
    p[0] = p[1] + [p[3]]


def p_expr_stmt_star_4(p):
    '''expr_stmt_star : expr_stmt_star EQUAL testlist'''
    #                                1     2        3
    p[0] = p[1] + [p[3]]


def p_augassign_1(p):
    '''augassign : PLUSEQUAL'''
    #                      1
    p[0] = ast.Add(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_2(p):
    '''augassign : MINEQUAL'''
    #                     1
    p[0] = ast.Sub(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_3(p):
    '''augassign : STAREQUAL'''
    #                      1
    p[0] = ast.Mult(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_4(p):
    '''augassign : SLASHEQUAL'''
    #                       1
    p[0] = ast.Div(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_5(p):
    '''augassign : PERCENTEQUAL'''
    #                         1
    p[0] = ast.Mod(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_6(p):
    '''augassign : AMPEREQUAL'''
    #                       1
    p[0] = ast.BitAnd(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_7(p):
    '''augassign : VBAREQUAL'''
    #                      1
    p[0] = ast.BitOr(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_8(p):
    '''augassign : CIRCUMFLEXEQUAL'''
    #                            1
    p[0] = ast.BitXor(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_9(p):
    '''augassign : LEFTSHIFTEQUAL'''
    #                           1
    p[0] = ast.LShift(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_10(p):
    '''augassign : RIGHTSHIFTEQUAL'''
    #                            1
    p[0] = ast.RShift(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_11(p):
    '''augassign : DOUBLESTAREQUAL'''
    #                            1
    p[0] = ast.Pow(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_augassign_12(p):
    '''augassign : DOUBLESLASHEQUAL'''
    #                             1
    p[0] = ast.FloorDiv(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_1(p):
    '''print_stmt : PRINT'''
    #                   1
    p[0] = ast.Print(None, [], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_2(p):
    '''print_stmt : PRINT test'''
    #                   1    2
    p[0] = ast.Print(None, [p[2]], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_3(p):
    '''print_stmt : PRINT test COMMA'''
    #                   1    2     3
    p[0] = ast.Print(None, [p[2]], False, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_4(p):
    '''print_stmt : PRINT test print_stmt_plus'''
    #                   1    2               3
    p[0] = ast.Print(None, [p[2]] + p[3], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_5(p):
    '''print_stmt : PRINT test print_stmt_plus COMMA'''
    #                   1    2               3     4
    p[0] = ast.Print(None, [p[2]] + p[3], False, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_6(p):
    '''print_stmt : PRINT RIGHTSHIFT test'''
    #                   1          2    3
    p[0] = ast.Print(p[3], [], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_7(p):
    '''print_stmt : PRINT RIGHTSHIFT test print_stmt_plus'''
    #                   1          2    3               4
    p[0] = ast.Print(p[3], p[4], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_8(p):
    '''print_stmt : PRINT RIGHTSHIFT test print_stmt_plus COMMA'''
    #                   1          2    3               4     5
    p[0] = ast.Print(p[3], p[4], False, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_print_stmt_plus_1(p):
    '''print_stmt_plus : COMMA test'''
    #                        1    2
    p[0] = [p[2]]


def p_print_stmt_plus_2(p):
    '''print_stmt_plus : print_stmt_plus COMMA test'''
    #                                  1     2    3
    p[0] = p[1] + [p[3]]

# del_stmt: 'del' exprlist


def p_del_stmt(p):
    '''del_stmt : DEL exprlist'''
    #               1        2
    # interesting fact: evaluating Delete nodes with ctx=Store() causes a segmentation fault in Python!
    ctx_to_store(p[2], ast.Del)
    if isinstance(p[2], ast.Tuple) and not p[2].paren:
        p[0] = ast.Delete(p[2].elts, rule=inspect.currentframe().f_code.co_name, **p[1][1])
    else:
        p[0] = ast.Delete([p[2]], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_pass_stmt(p):
    '''pass_stmt : PASS'''
    #                 1
    p[0] = ast.Pass(rule=inspect.currentframe().f_code.co_name, **p[1][1])

# flow_stmt: break_stmt | continue_stmt | return_stmt | raise_stmt | yield_stmt


def p_flow_stmt_1(p):
    '''flow_stmt : break_stmt'''
    #                       1
    p[0] = p[1]


def p_flow_stmt_2(p):
    '''flow_stmt : continue_stmt'''
    #                          1
    p[0] = p[1]


def p_flow_stmt_3(p):
    '''flow_stmt : return_stmt'''
    #                        1
    p[0] = p[1]


def p_flow_stmt_4(p):
    '''flow_stmt : raise_stmt'''
    #                       1
    p[0] = p[1]


def p_flow_stmt_5(p):
    '''flow_stmt : yield_stmt'''
    #                       1
    p[0] = ast.Expr(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])

# break_stmt: 'break'


def p_break_stmt(p):
    '''break_stmt : BREAK'''
    #                   1
    p[0] = ast.Break(rule=inspect.currentframe().f_code.co_name, **p[1][1])

# continue_stmt: 'continue'


def p_continue_stmt(p):
    '''continue_stmt : CONTINUE'''
    #                         1
    p[0] = ast.Continue(rule=inspect.currentframe().f_code.co_name, **p[1][1])

# return_stmt: 'return' [testlist]


def p_return_stmt_1(p):
    '''return_stmt : RETURN'''
    #                     1
    p[0] = ast.Return(None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_return_stmt_2(p):
    '''return_stmt : RETURN testlist'''
    #                     1        2
    p[0] = ast.Return(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# yield_stmt: yield_expr


def p_yield_stmt(p):
    '''yield_stmt : yield_expr'''
    #                        1
    p[0] = p[1]

# raise_stmt: 'raise' [test [',' test [',' test]]]


def p_raise_stmt_1(p):
    '''raise_stmt : RAISE'''
    #                   1
    p[0] = ast.Raise(None, None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_raise_stmt_2(p):
    '''raise_stmt : RAISE test'''
    #                   1    2
    p[0] = ast.Raise(p[2], None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_raise_stmt_3(p):
    '''raise_stmt : RAISE test COMMA test'''
    #                   1    2     3    4
    p[0] = ast.Raise(p[2], p[4], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_raise_stmt_4(p):
    '''raise_stmt : RAISE test COMMA test COMMA test'''
    #                   1    2     3    4     5    6
    p[0] = ast.Raise(p[2], p[4], p[6], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# import_stmt: import_name | import_from


def p_import_stmt_1(p):
    '''import_stmt : import_name'''
    #                          1
    p[0] = p[1]


def p_import_stmt_2(p):
    '''import_stmt : import_from'''
    #                          1
    p[0] = p[1]

# import_name: 'import' dotted_as_names


def p_import_name(p):
    '''import_name : IMPORT dotted_as_names'''
    #                     1               2
    p[0] = ast.Import(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# import_from: ('from' ('.'* dotted_name | '.'+)
#               'import' ('*' | '(' import_as_names ')' | import_as_names))


def p_import_from_1(p):
    '''import_from : FROM dotted_name IMPORT STAR'''
    #                   1           2      3    4
    dotted = []
    last = p[2]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted),
                          [ast.alias("*",
                                     None,
                                     rule=inspect.currentframe().f_code.co_name,
                                     **p[3][1])],
                          0,
                          rule=inspect.currentframe().f_code.co_name,
                          **p[1][1])


def p_import_from_2(p):
    '''import_from : FROM dotted_name IMPORT LPAR import_as_names RPAR'''
    #                   1           2      3    4               5    6
    dotted = []
    last = p[2]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[5], 0, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_3(p):
    '''import_from : FROM dotted_name IMPORT import_as_names'''
    #                   1           2      3               4
    dotted = []
    last = p[2]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[4], 0, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_4(p):
    '''import_from : FROM import_from_plus dotted_name IMPORT STAR'''
    #                   1                2           3      4    5
    dotted = []
    last = p[3]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(
        ".".join(dotted), [
            ast.alias(
                "*", None, rule=inspect.currentframe().f_code.co_name, **p[4][1])], p[2], **p[1][1])


def p_import_from_5(p):
    '''import_from : FROM import_from_plus dotted_name IMPORT LPAR import_as_names RPAR'''
    #                   1                2           3      4    5               6    7
    dotted = []
    last = p[3]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[6], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_6(p):
    '''import_from : FROM import_from_plus dotted_name IMPORT import_as_names'''
    #                   1                2           3      4               5
    dotted = []
    last = p[3]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[5], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_7(p):
    '''import_from : FROM import_from_plus IMPORT STAR'''
    #                   1                2      3    4
    p[0] = ast.ImportFrom(None, [ast.alias("*", None, rule=inspect.currentframe().f_code.co_name,
                                           **p[3][1])], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_8(p):
    '''import_from : FROM import_from_plus IMPORT LPAR import_as_names RPAR'''
    #                   1                2      3    4               5    6
    p[0] = ast.ImportFrom(None, p[5], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_9(p):
    '''import_from : FROM import_from_plus IMPORT import_as_names'''
    #                   1                2      3               4
    p[0] = ast.ImportFrom(None, p[4], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_from_plus_1(p):
    '''import_from_plus : DOT'''
    #                       1
    p[0] = 1


def p_import_from_plus_2(p):
    '''import_from_plus : import_from_plus DOT'''
    #                                    1   2
    p[0] = p[1] + 1

# import_as_name: NAME ['as' NAME]


def p_import_as_name_1(p):
    '''import_as_name : NAME'''
    #                      1
    p[0] = ast.alias(p[1][0], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_import_as_name_2(p):
    '''import_as_name : NAME AS NAME'''
    #                      1  2    3
    p[0] = ast.alias(p[1][0], p[3][0], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# dotted_as_name: dotted_name ['as' NAME]


def p_dotted_as_name_1(p):
    '''dotted_as_name : dotted_name'''
    #                             1
    dotted = []
    last = p[1]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.alias(".".join(dotted), None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_dotted_as_name_2(p):
    '''dotted_as_name : dotted_name AS NAME'''
    #                             1  2    3
    dotted = []
    last = p[1]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.alias(".".join(dotted), p[3][0], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])

# import_as_names: import_as_name (',' import_as_name)* [',']


def p_import_as_names_1(p):
    '''import_as_names : import_as_name'''
    #                                 1
    p[0] = [p[1]]


def p_import_as_names_2(p):
    '''import_as_names : import_as_name COMMA'''
    #                                 1     2
    p[0] = [p[1]]


def p_import_as_names_3(p):
    '''import_as_names : import_as_name import_as_names_star'''
    #                                 1                    2
    p[0] = [p[1]] + p[2]


def p_import_as_names_4(p):
    '''import_as_names : import_as_name import_as_names_star COMMA'''
    #                                 1                    2     3
    p[0] = [p[1]] + p[2]


def p_import_as_names_star_1(p):
    '''import_as_names_star : COMMA import_as_name'''
    #                             1              2
    p[0] = [p[2]]


def p_import_as_names_star_2(p):
    '''import_as_names_star : import_as_names_star COMMA import_as_name'''
    #                                            1     2              3
    p[0] = p[1] + [p[3]]

# dotted_as_names: dotted_as_name (',' dotted_as_name)*


def p_dotted_as_names_1(p):
    '''dotted_as_names : dotted_as_name'''
    #                                 1
    p[0] = [p[1]]


def p_dotted_as_names_2(p):
    '''dotted_as_names : dotted_as_name dotted_as_names_star'''
    #                                 1                    2
    p[0] = [p[1]] + p[2]


def p_dotted_as_names_star_1(p):
    '''dotted_as_names_star : COMMA dotted_as_name'''
    #                             1              2
    p[0] = [p[2]]


def p_dotted_as_names_star_2(p):
    '''dotted_as_names_star : dotted_as_names_star COMMA dotted_as_name'''
    #                                            1     2              3
    p[0] = p[1] + [p[3]]

# dotted_name: NAME ('.' NAME)*


def p_dotted_name_1(p):
    '''dotted_name : NAME'''
    #                   1
    p[0] = ast.Name(p[1][0], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_dotted_name_2(p):
    '''dotted_name : NAME dotted_name_star'''
    #                   1                2
    last = p[2]
    if isinstance(last, ast.Attribute):
        inherit_lineno(last, p[1][1])
        while isinstance(last.value, ast.Attribute):
            last = last.value
            inherit_lineno(last, p[1][1])
        last.value = ast.Attribute(
            ast.Name(
                p[1][0],
                ast.Load(),
                rule=inspect.currentframe().f_code.co_name,
                **p[1][1]),
            last.value,
            ast.Load(),
            rule=inspect.currentframe().f_code.co_name,
            **p[1][1])
        p[0] = p[2]
    else:
        p[0] = ast.Attribute(
            ast.Name(
                p[1][0],
                ast.Load(),
                rule=inspect.currentframe().f_code.co_name,
                **p[1][1]),
            p[2],
            ast.Load(),
            rule=inspect.currentframe().f_code.co_name,
            **p[1][1])


def p_dotted_name_star_1(p):
    '''dotted_name_star : DOT NAME'''
    #                       1    2
    p[0] = p[2][0]


def p_dotted_name_star_2(p):
    '''dotted_name_star : dotted_name_star DOT NAME'''
    #                                    1   2    3
    p[0] = ast.Attribute(p[1], p[3][0], ast.Load(), rule=inspect.currentframe().f_code.co_name)

# global_stmt: 'global' NAME (',' NAME)*


def p_global_stmt_1(p):
    '''global_stmt : GLOBAL NAME'''
    #                     1    2
    p[0] = ast.Global([p[2][0]], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_global_stmt_2(p):
    '''global_stmt : GLOBAL NAME global_stmt_star'''
    #                     1    2                3
    p[0] = ast.Global([p[2][0]] + p[3], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_global_stmt_star_1(p):
    '''global_stmt_star : COMMA NAME'''
    #                         1    2
    p[0] = [p[2][0]]


def p_global_stmt_star_2(p):
    '''global_stmt_star : global_stmt_star COMMA NAME'''
    #                                    1     2    3
    p[0] = p[1] + [p[3][0]]

# exec_stmt: 'exec' expr ['in' test [',' test]]


def p_exec_stmt_1(p):
    '''exec_stmt : EXEC expr'''
    #                 1    2
    p[0] = ast.Exec(p[2], None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_exec_stmt_2(p):
    '''exec_stmt : EXEC expr IN test'''
    #                 1    2  3    4
    p[0] = ast.Exec(p[2], p[4], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_exec_stmt_3(p):
    '''exec_stmt : EXEC expr IN test COMMA test'''
    #                 1    2  3    4     5    6
    p[0] = ast.Exec(p[2], p[4], p[6], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# assert_stmt: 'assert' test [',' test]


def p_assert_stmt_1(p):
    '''assert_stmt : ASSERT test'''
    #                     1    2
    p[0] = ast.Assert(p[2], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_assert_stmt_2(p):
    '''assert_stmt : ASSERT test COMMA test'''
    #                     1    2     3    4
    p[0] = ast.Assert(p[2], p[4], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# compound_stmt: if_stmt | while_stmt | for_stmt | try_stmt | with_stmt | funcdef | classdef | decorated


def p_compound_stmt_1(p):
    '''compound_stmt : if_stmt'''
    #                        1
    p[0] = [p[1]]


def p_compound_stmt_2(p):
    '''compound_stmt : while_stmt'''
    #                           1
    p[0] = [p[1]]


def p_compound_stmt_3(p):
    '''compound_stmt : for_stmt'''
    #                         1
    p[0] = [p[1]]


def p_compound_stmt_4(p):
    '''compound_stmt : try_stmt'''
    #                         1
    p[0] = [p[1]]


def p_compound_stmt_5(p):
    '''compound_stmt : with_stmt'''
    #                          1
    p[0] = [p[1]]


def p_compound_stmt_6(p):
    '''compound_stmt : funcdef'''
    #                        1
    p[0] = [p[1]]


def p_compound_stmt_7(p):
    '''compound_stmt : classdef'''
    #                         1
    p[0] = [p[1]]


def p_compound_stmt_8(p):
    '''compound_stmt : decorated'''
    #                          1
    p[0] = [p[1]]

# if_stmt: 'if' test ':' suite ('elif' test ':' suite)* ['else' ':' suite]


def p_if_stmt_1(p):
    '''if_stmt : IF test COLON suite'''
    #             1    2     3     4
    p[0] = ast.If(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_if_stmt_2(p):
    '''if_stmt : IF test COLON suite ELSE COLON suite'''
    #             1    2     3     4    5     6     7
    p[0] = ast.If(p[2], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_if_stmt_3(p):
    '''if_stmt : IF test COLON suite if_stmt_star'''
    #             1    2     3     4            5
    p[0] = ast.If(p[2], p[4], [p[5]], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_if_stmt_4(p):
    '''if_stmt : IF test COLON suite if_stmt_star ELSE COLON suite'''
    #             1    2     3     4            5    6     7     8
    last = p[5]
    while len(last.orelse) > 0:
        last = last.orelse[0]
    last.orelse.extend(p[8])
    p[0] = ast.If(p[2], p[4], [p[5]], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_if_stmt_star_1(p):
    '''if_stmt_star : ELIF test COLON suite'''
    #                    1    2     3     4
    p[0] = ast.If(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2])


def p_if_stmt_star_2(p):
    '''if_stmt_star : if_stmt_star ELIF test COLON suite'''
    #                            1    2    3     4     5
    last = p[1]
    while len(last.orelse) > 0:
        last = last.orelse[0]
    last.orelse.append(ast.If(p[3], p[5], [], rule=inspect.currentframe().f_code.co_name))
    inherit_lineno(last.orelse[-1], p[3])
    p[0] = p[1]

# while_stmt: 'while' test ':' suite ['else' ':' suite]


def p_while_stmt_1(p):
    '''while_stmt : WHILE test COLON suite'''
    #                   1    2     3     4
    p[0] = ast.While(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_while_stmt_2(p):
    '''while_stmt : WHILE test COLON suite ELSE COLON suite'''
    #                   1    2     3     4    5     6     7
    p[0] = ast.While(p[2], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# for_stmt: 'for' exprlist 'in' testlist ':' suite ['else' ':' suite]


def p_for_stmt_1(p):
    '''for_stmt : FOR exprlist IN testlist COLON suite'''
    #               1        2  3        4     5     6
    ctx_to_store(p[2])
    p[0] = ast.For(p[2], p[4], p[6], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_for_stmt_2(p):
    '''for_stmt : FOR exprlist IN testlist COLON suite ELSE COLON suite'''
    #               1        2  3        4     5     6    7     8     9
    ctx_to_store(p[2])
    p[0] = ast.For(p[2], p[4], p[6], p[9], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# try_stmt: ('try' ':' suite
#            ((except_clause ':' suite)+
#             ['else' ':' suite]
#             ['finally' ':' suite] |
#            'finally' ':' suite))


def p_try_stmt_1(p):
    '''try_stmt : TRY COLON suite try_stmt_plus'''
    #               1     2     3             4
    p[0] = ast.TryExcept(p[3], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_try_stmt_2(p):
    '''try_stmt : TRY COLON suite try_stmt_plus FINALLY COLON suite'''
    #               1     2     3             4       5     6     7
    p[0] = ast.TryFinally([ast.TryExcept(p[3], p[4], [], rule=inspect.currentframe().f_code.co_name,
                                         **p[1][1])], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_try_stmt_3(p):
    '''try_stmt : TRY COLON suite try_stmt_plus ELSE COLON suite'''
    #               1     2     3             4    5     6     7
    p[0] = ast.TryExcept(p[3], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_try_stmt_4(p):
    '''try_stmt : TRY COLON suite try_stmt_plus ELSE COLON suite FINALLY COLON suite'''
    #               1     2     3             4    5     6     7       8     9    10
    p[0] = ast.TryFinally([ast.TryExcept(p[3], p[4], p[7], rule=inspect.currentframe().f_code.co_name,
                                         **p[1][1])], p[10], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_try_stmt_5(p):
    '''try_stmt : TRY COLON suite FINALLY COLON suite'''
    #               1     2     3       4     5     6
    p[0] = ast.TryFinally(p[3], p[6], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_try_stmt_plus_1(p):
    '''try_stmt_plus : except_clause COLON suite'''
    #                              1     2     3
    p[1].body = p[3]
    p[0] = [p[1]]


def p_try_stmt_plus_2(p):
    '''try_stmt_plus : try_stmt_plus except_clause COLON suite'''
    #                              1             2     3     4
    p[2].body = p[4]
    p[0] = p[1] + [p[2]]

# with_stmt: 'with' with_item (',' with_item)*  ':' suite


def p_with_stmt_1(p):
    '''with_stmt : WITH with_item COLON suite'''
    #                 1         2     3     4
    p[2].body = p[4]
    p[0] = p[2]


def p_with_stmt_2(p):
    '''with_stmt : WITH with_item with_stmt_star COLON suite'''
    #                 1         2              3     4     5
    p[2].body.append(p[3])
    last = p[2]
    while len(last.body) > 0:
        last = last.body[0]
    last.body = p[5]
    p[0] = p[2]


def p_with_stmt_star_1(p):
    '''with_stmt_star : COMMA with_item'''
    #                       1         2
    p[0] = p[2]


def p_with_stmt_star_2(p):
    '''with_stmt_star : with_stmt_star COMMA with_item'''
    #                                1     2         3
    last = p[1]
    while len(last.body) > 0:
        last = last.body[0]
    last.body.append(p[3])
    p[0] = p[1]

# with_item: test ['as' expr]


def p_with_item_1(p):
    '''with_item : test'''
    #                 1
    p[0] = ast.With(p[1], None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_with_item_2(p):
    '''with_item : test AS expr'''
    #                 1  2    3
    ctx_to_store(p[3])
    p[0] = ast.With(p[1], p[3], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])

# except_clause: 'except' [test [('as' | ',') test]]


def p_except_clause_1(p):
    '''except_clause : EXCEPT'''
    #                       1
    p[0] = ast.ExceptHandler(None, None, [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_except_clause_2(p):
    '''except_clause : EXCEPT test'''
    #                       1    2
    p[0] = ast.ExceptHandler(p[2], None, [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_except_clause_3(p):
    '''except_clause : EXCEPT test AS test'''
    #                       1    2  3    4
    ctx_to_store(p[4])
    p[0] = ast.ExceptHandler(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_except_clause_4(p):
    '''except_clause : EXCEPT test COMMA test'''
    #                       1    2     3    4
    ctx_to_store(p[4])
    p[0] = ast.ExceptHandler(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# suite: simple_stmt | NEWLINE INDENT stmt+ DEDENT


def p_suite_1(p):
    '''suite : simple_stmt'''
    #                    1
    p[0] = p[1]


def p_suite_2(p):
    '''suite : NEWLINE INDENT suite_plus DEDENT'''
    #                1      2          3      4
    p[0] = p[3]


def p_suite_plus_1(p):
    '''suite_plus : stmt'''
    #                  1
    p[0] = p[1]


def p_suite_plus_2(p):
    '''suite_plus : suite_plus stmt'''
    #                        1    2
    p[0] = p[1] + p[2]

# testlist_safe: old_test [(',' old_test)+ [',']]


def p_testlist_safe_1(p):
    '''testlist_safe : old_test'''
    #                         1
    p[0] = p[1]


def p_testlist_safe_2(p):
    '''testlist_safe : old_test testlist_safe_plus'''
    #                         1                  2
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_safe_3(p):
    '''testlist_safe : old_test testlist_safe_plus COMMA'''
    #                         1                  2     3
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_safe_plus_1(p):
    '''testlist_safe_plus : COMMA old_test'''
    #                           1        2
    p[0] = [p[2]]


def p_testlist_safe_plus_2(p):
    '''testlist_safe_plus : testlist_safe_plus COMMA old_test'''
    #                                        1     2        3
    p[0] = p[1] + [p[3]]

# old_test: or_test | old_lambdef


def p_old_test_1(p):
    '''old_test : or_test'''
    #                   1
    p[0] = p[1]


def p_old_test_2(p):
    '''old_test : old_lambdef'''
    #                       1
    p[0] = p[1]

# old_lambdef: 'lambda' [varargslist] ':' old_test


def p_old_lambdef_1(p):
    '''old_lambdef : LAMBDA COLON old_test'''
    #                     1     2        3
    p[0] = ast.Lambda(
        ast.arguments(
            [],
            None,
            None,
            [],
            rule=inspect.currentframe().f_code.co_name,
            **p[2][1]),
        p[3],
        rule=inspect.currentframe().f_code.co_name,
        **p[1][1])


def p_old_lambdef_2(p):
    '''old_lambdef : LAMBDA varargslist COLON old_test'''
    #                     1           2     3        4
    p[0] = ast.Lambda(p[2], p[4], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# test: or_test ['if' or_test 'else' test] | lambdef


def p_test_1(p):
    '''test : or_test'''
    #               1
    p[0] = p[1]


def p_test_2(p):
    '''test : or_test IF or_test ELSE test'''
    #               1  2       3    4    5
    p[0] = ast.IfExp(p[3], p[1], p[5], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_test_3(p):
    '''test : lambdef'''
    #               1
    p[0] = p[1]

# or_test: and_test ('or' and_test)*


def p_or_test_1(p):
    '''or_test : and_test'''
    #                   1
    p[0] = p[1]


def p_or_test_2(p):
    '''or_test : and_test or_test_star'''
    #                   1            2
    theor = ast.Or(rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(theor, p[2][0])
    p[0] = ast.BoolOp(theor, [p[1]] + p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_or_test_star_1(p):
    '''or_test_star : OR and_test'''
    #                  1        2
    p[0] = [p[2]]


def p_or_test_star_2(p):
    '''or_test_star : or_test_star OR and_test'''
    #                            1  2        3
    p[0] = p[1] + [p[3]]

# and_test: not_test ('and' not_test)*


def p_and_test_1(p):
    '''and_test : not_test'''
    #                    1
    p[0] = p[1]


def p_and_test_2(p):
    '''and_test : not_test and_test_star'''
    #                    1             2
    theand = ast.And(rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(theand, p[2][0])
    p[0] = ast.BoolOp(theand, [p[1]] + p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_and_test_star_1(p):
    '''and_test_star : AND not_test'''
    #                    1        2
    p[0] = [p[2]]


def p_and_test_star_2(p):
    '''and_test_star : and_test_star AND not_test'''
    #                              1   2        3
    p[0] = p[1] + [p[3]]

# not_test: 'not' not_test | comparison


def p_not_test_1(p):
    '''not_test : NOT not_test'''
    #               1        2
    thenot = ast.Not(rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(thenot, p[2])
    p[0] = ast.UnaryOp(thenot, p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_not_test_2(p):
    '''not_test : comparison'''
    #                      1
    p[0] = p[1]

# comparison: expr (comp_op expr)*


def p_comparison_1(p):
    '''comparison : expr'''
    #                  1
    p[0] = p[1]


def p_comparison_2(p):
    '''comparison : expr comparison_star'''
    #                  1               2
    ops, exprs = p[2]
    p[0] = ast.Compare(p[1], ops, exprs, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_comparison_star_1(p):
    '''comparison_star : comp_op expr'''
    #                          1    2
    inherit_lineno(p[1], p[2])
    p[0] = ([p[1]], [p[2]])


def p_comparison_star_2(p):
    '''comparison_star : comparison_star comp_op expr'''
    #                                  1       2    3
    ops, exprs = p[1]
    inherit_lineno(p[2], p[3])
    p[0] = (ops + [p[2]], exprs + [p[3]])

# comp_op: '<'|'>'|'=='|'>='|'<='|'!='|'in'|'not' 'in'|'is'|'is' 'not'


def p_comp_op_1(p):
    '''comp_op : LESS'''
    #               1
    p[0] = ast.Lt(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_2(p):
    '''comp_op : GREATER'''
    #                  1
    p[0] = ast.Gt(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_3(p):
    '''comp_op : EQEQUAL'''
    #                  1
    p[0] = ast.Eq(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_4(p):
    '''comp_op : GREATEREQUAL'''
    #                       1
    p[0] = ast.GtE(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_5(p):
    '''comp_op : LESSEQUAL'''
    #                    1
    p[0] = ast.LtE(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_6(p):
    '''comp_op : NOTEQUAL'''
    #                   1
    p[0] = ast.NotEq(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_7(p):
    '''comp_op : IN'''
    #             1
    p[0] = ast.In(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_8(p):
    '''comp_op : NOT IN'''
    #              1  2
    p[0] = ast.NotIn(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_9(p):
    '''comp_op : IS'''
    #             1
    p[0] = ast.Is(rule=inspect.currentframe().f_code.co_name)


def p_comp_op_10(p):
    '''comp_op : IS NOT'''
    #             1   2
    p[0] = ast.IsNot(rule=inspect.currentframe().f_code.co_name)

# expr: xor_expr ('|' xor_expr)*


def p_expr_1(p):
    '''expr : xor_expr'''
    #                1
    p[0] = p[1]


def p_expr_2(p):
    '''expr : xor_expr expr_star'''
    #                1         2
    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)


def p_expr_star_1(p):
    '''expr_star : VBAR xor_expr'''
    #                 1        2
    p[0] = [ast.BitOr(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_expr_star_2(p):
    '''expr_star : expr_star VBAR xor_expr'''
    #                      1    2        3
    p[0] = p[1] + [ast.BitOr(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]

# xor_expr: and_expr ('^' and_expr)*


def p_xor_expr_1(p):
    '''xor_expr : and_expr'''
    #                    1
    p[0] = p[1]


def p_xor_expr_2(p):
    '''xor_expr : and_expr xor_expr_star'''
    #                    1             2
    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)


def p_xor_expr_star_1(p):
    '''xor_expr_star : CIRCUMFLEX and_expr'''
    #                           1        2
    p[0] = [ast.BitXor(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_xor_expr_star_2(p):
    '''xor_expr_star : xor_expr_star CIRCUMFLEX and_expr'''
    #                              1          2        3
    p[0] = p[1] + [ast.BitXor(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]

# and_expr: shift_expr ('&' shift_expr)*


def p_and_expr_1(p):
    '''and_expr : shift_expr'''
    #                      1
    p[0] = p[1]


def p_and_expr_2(p):
    '''and_expr : shift_expr and_expr_star'''
    #                      1             2
    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 0)


def p_and_expr_star_1(p):
    '''and_expr_star : AMPER shift_expr'''
    #                      1          2
    p[0] = [ast.BitAnd(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_and_expr_star_2(p):
    '''and_expr_star : and_expr_star AMPER shift_expr'''
    #                              1     2          3
    p[0] = p[1] + [ast.BitAnd(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]

# shift_expr: arith_expr (('<<'|'>>') arith_expr)*


def p_shift_expr_1(p):
    '''shift_expr : arith_expr'''
    #                        1
    p[0] = p[1]


def p_shift_expr_2(p):
    '''shift_expr : arith_expr shift_expr_star'''
    #                        1               2
    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)


def p_shift_expr_star_1(p):
    '''shift_expr_star : LEFTSHIFT arith_expr'''
    #                            1          2
    p[0] = [ast.LShift(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_shift_expr_star_2(p):
    '''shift_expr_star : RIGHTSHIFT arith_expr'''
    #                             1          2
    p[0] = [ast.RShift(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_shift_expr_star_3(p):
    '''shift_expr_star : shift_expr_star LEFTSHIFT arith_expr'''
    #                                  1         2          3
    p[0] = p[1] + [ast.LShift(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]


def p_shift_expr_star_4(p):
    '''shift_expr_star : shift_expr_star RIGHTSHIFT arith_expr'''
    #                                  1          2          3
    p[0] = p[1] + [ast.RShift(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]

# arith_expr: term (('+'|'-') term)*


def p_arith_expr_1(p):
    '''arith_expr : term'''
    #                  1
    p[0] = p[1]


def p_arith_expr_2(p):
    '''arith_expr : term arith_expr_star'''
    #                  1               2
    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)


def p_arith_expr_star_1(p):
    '''arith_expr_star : PLUS term'''
    #                       1    2
    p[0] = [ast.Add(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_arith_expr_star_2(p):
    '''arith_expr_star : MINUS term'''
    #                        1    2
    p[0] = [ast.Sub(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_arith_expr_star_3(p):
    '''arith_expr_star : arith_expr_star PLUS term'''
    #                                  1    2    3
    p[0] = p[1] + [ast.Add(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]


def p_arith_expr_star_4(p):
    '''arith_expr_star : arith_expr_star MINUS term'''
    #                                  1     2    3
    p[0] = p[1] + [ast.Sub(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]

# term: factor (('*'|'/'|'%'|'//') factor)*


def p_term_1(p):
    '''term : factor'''
    #              1
    p[0] = p[1]


def p_term_2(p):
    '''term : factor term_star'''
    #              1         2
    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)


def p_term_star_1(p):
    '''term_star : STAR factor'''
    #                 1      2
    p[0] = [ast.Mult(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_term_star_2(p):
    '''term_star : SLASH factor'''
    #                  1      2
    p[0] = [ast.Div(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_term_star_3(p):
    '''term_star : PERCENT factor'''
    #                    1      2
    p[0] = [ast.Mod(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_term_star_4(p):
    '''term_star : DOUBLESLASH factor'''
    #                        1      2
    p[0] = [ast.FloorDiv(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]


def p_term_star_5(p):
    '''term_star : term_star STAR factor'''
    #                      1    2      3
    p[0] = p[1] + [ast.Mult(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]


def p_term_star_6(p):
    '''term_star : term_star SLASH factor'''
    #                      1     2      3
    p[0] = p[1] + [ast.Div(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]


def p_term_star_7(p):
    '''term_star : term_star PERCENT factor'''
    #                      1       2      3
    p[0] = p[1] + [ast.Mod(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]


def p_term_star_8(p):
    '''term_star : term_star DOUBLESLASH factor'''
    #                      1           2      3
    p[0] = p[1] + [ast.FloorDiv(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]

# factor: ('+'|'-'|'~') factor | power


def p_factor_1(p):
    '''factor : PLUS factor'''
    #              1      2
    op = ast.UAdd(rule=inspect.currentframe().f_code.co_name, **p[1][1])
    p[0] = ast.UnaryOp(op, p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], op)


def p_factor_2(p):
    '''factor : MINUS factor'''
    #               1      2
    if isinstance(p[2], ast.Num) and not hasattr(p[2], "unary"):
        p[2].n *= -1
        p[0] = p[2]
        p[0].unary = True
        inherit_lineno(p[0], p[1][1])
    else:
        op = ast.USub(rule=inspect.currentframe().f_code.co_name, **p[1][1])
        p[0] = ast.UnaryOp(op, p[2], rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], op)


def p_factor_3(p):
    '''factor : TILDE factor'''
    #               1      2
    op = ast.Invert(rule=inspect.currentframe().f_code.co_name, **p[1][1])
    p[0] = ast.UnaryOp(op, p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], op)


def p_factor_4(p):
    '''factor : power'''
    #               1
    p[0] = p[1]

# power: atom trailer* ['**' factor]


def p_power_1(p):
    '''power : atom'''
    #             1
    p[0] = p[1]


def p_power_2(p):
    '''power : atom DOUBLESTAR factor'''
    #             1          2      3
    p[0] = ast.BinOp(
        p[1],
        ast.Pow(
            rule=inspect.currentframe().f_code.co_name,
            **p[2][1]),
        p[3],
        rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_power_3(p):
    '''power : atom power_star'''
    #             1          2
    p[0] = unpack_trailer(p[1], p[2])


def p_power_4(p):
    '''power : atom power_star DOUBLESTAR factor'''
    #             1          2          3      4
    p[0] = ast.BinOp(
        unpack_trailer(
            p[1],
            p[2]),
        ast.Pow(
            rule=inspect.currentframe().f_code.co_name,
            **p[3][1]),
        p[4],
        rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_power_star_1(p):
    '''power_star : trailer'''
    #                     1
    p[0] = [p[1]]


def p_power_star_2(p):
    '''power_star : power_star trailer'''
    #                        1       2
    p[0] = p[1] + [p[2]]

# atom: ('(' [yield_expr|testlist_comp] ')' |
#        '[' [listmaker] ']' |
#        '{' [dictorsetmaker] '}' |
#        '`' testlist1 '`' |
#        NAME | NUMBER | STRING+ | DOLLARNUMBER)


def p_atom_1(p):
    '''atom : LPAR RPAR'''
    #            1    2
    p[0] = ast.Tuple([], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=True, **p[1][1])


def p_atom_2(p):
    '''atom : LPAR yield_expr RPAR'''
    #            1          2    3
    p[0] = p[2]
    if isinstance(p[0], ast.Tuple):
        p[0].paren = True
    p[0].alt = p[1][1]


def p_atom_3(p):
    '''atom : LPAR testlist_comp RPAR'''
    #            1             2    3
    p[0] = p[2]
    if isinstance(p[0], ast.Tuple):
        p[0].paren = True
    p[0].alt = p[1][1]


def p_atom_4(p):
    '''atom : LSQB RSQB'''
    #            1    2
    p[0] = ast.List([], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_5(p):
    '''atom : LSQB listmaker RSQB'''
    #            1         2    3
    if isinstance(p[2], ast.ListComp):
        p[0] = p[2]
        p[0].alt = p[1][1]
    else:
        p[0] = ast.List(p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_6(p):
    '''atom : LBRACE RBRACE'''
    #              1      2
    p[0] = ast.Dict([], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_7(p):
    '''atom : LBRACE dictorsetmaker RBRACE'''
    #              1              2      3
    if isinstance(p[2], (ast.SetComp, ast.DictComp)):
        p[0] = p[2]
        p[0].alt = p[1][1]
    else:
        keys, values = p[2]
        if keys is None:
            p[0] = ast.Set(values, rule=inspect.currentframe().f_code.co_name, **p[1][1])
        else:
            p[0] = ast.Dict(keys, values, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_8(p):
    '''atom : BACKQUOTE testlist1 BACKQUOTE'''
    #                 1         2         3
    p[0] = ast.Repr(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_9(p):
    '''atom : NAME'''
    #            1
    p[0] = ast.Name(p[1][0], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_10(p):
    '''atom : NUMBER'''
    #              1
    p[0] = ast.Num(p[1][0], rule=inspect.currentframe().f_code.co_name, **p[1][2])


def p_atom_11(p):
    '''atom : atom_plus'''
    #                 1
    p[0] = p[1]


def p_atom_12(p):
    '''atom : DOLLARNUMBER'''
    #                    1
    p[0] = DollarNumber(int(p[1][0][1:]), **p[1][1])


def p_atom_plus_1(p):
    '''atom_plus : STRING'''
    #                   1
    p[0] = ast.Str(p[1][0], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_atom_plus_2(p):
    '''atom_plus : atom_plus STRING'''
    #                      1      2
    p[1].s = p[1].s + p[2][0]
    p[0] = p[1]

# listmaker: test ( list_for | (',' test)* [','] )


def p_listmaker_1(p):
    '''listmaker : test list_for'''
    #                 1        2
    p[0] = ast.ListComp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_listmaker_2(p):
    '''listmaker : test'''
    #                 1
    p[0] = [p[1]]


def p_listmaker_3(p):
    '''listmaker : test COMMA'''
    #                 1     2
    p[0] = [p[1]]


def p_listmaker_4(p):
    '''listmaker : test listmaker_star'''
    #                 1              2
    p[0] = [p[1]] + p[2]


def p_listmaker_5(p):
    '''listmaker : test listmaker_star COMMA'''
    #                 1              2     3
    p[0] = [p[1]] + p[2]


def p_listmaker_star_1(p):
    '''listmaker_star : COMMA test'''
    #                       1    2
    p[0] = [p[2]]


def p_listmaker_star_2(p):
    '''listmaker_star : listmaker_star COMMA test'''
    #                                1     2    3
    p[0] = p[1] + [p[3]]

# testlist_comp: test ( comp_for | (',' test)* [','] )


def p_testlist_comp_1(p):
    '''testlist_comp : test comp_for'''
    #                     1        2
    p[0] = ast.GeneratorExp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_testlist_comp_2(p):
    '''testlist_comp : test'''
    #                     1
    p[0] = p[1]


def p_testlist_comp_3(p):
    '''testlist_comp : test COMMA'''
    #                     1     2
    p[0] = ast.Tuple([p[1]], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_comp_4(p):
    '''testlist_comp : test testlist_comp_star'''
    #                     1                  2
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_comp_5(p):
    '''testlist_comp : test testlist_comp_star COMMA'''
    #                     1                  2     3
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_comp_star_1(p):
    '''testlist_comp_star : COMMA test'''
    #                           1    2
    p[0] = [p[2]]


def p_testlist_comp_star_2(p):
    '''testlist_comp_star : testlist_comp_star COMMA test'''
    #                                        1     2    3
    p[0] = p[1] + [p[3]]

# lambdef: 'lambda' [varargslist] ':' test


def p_lambdef_1(p):
    '''lambdef : LAMBDA COLON test'''
    #                 1     2    3
    p[0] = ast.Lambda(
        ast.arguments(
            [],
            None,
            None,
            [],
            rule=inspect.currentframe().f_code.co_name,
            **p[2][1]),
        p[3],
        rule=inspect.currentframe().f_code.co_name,
        **p[1][1])


def p_lambdef_2(p):
    '''lambdef : LAMBDA varargslist COLON test'''
    #                 1           2     3    4
    p[0] = ast.Lambda(p[2], p[4], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# trailer: '(' [arglist] ')' | '[' subscriptlist ']' | '.' NAME


def p_trailer_1(p):
    '''trailer : LPAR RPAR'''
    #               1    2
    p[0] = ast.Call(None, [], [], None, None, rule=inspect.currentframe().f_code.co_name)


def p_trailer_2(p):
    '''trailer : LPAR arglist RPAR'''
    #               1       2    3
    p[0] = p[2]


def p_trailer_3(p):
    '''trailer : LSQB subscriptlist RSQB'''
    #               1             2    3
    p[0] = ast.Subscript(None, p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name)


def p_trailer_4(p):
    '''trailer : DOT NAME'''
    #              1    2
    p[0] = ast.Attribute(None, p[2][0], ast.Load(), rule=inspect.currentframe().f_code.co_name)

# subscriptlist: subscript (',' subscript)* [',']


def p_subscriptlist_1(p):
    '''subscriptlist : subscript'''
    #                          1
    p[0] = p[1]


def p_subscriptlist_2(p):
    '''subscriptlist : subscript COMMA'''
    #                          1     2
    if isinstance(p[1], ast.Index):
        tup = ast.Tuple([p[1].value], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
        inherit_lineno(tup, p[1].value)
        p[0] = ast.Index(tup, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], tup)
    else:
        p[0] = ast.ExtSlice([p[1]], rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], p[1])


def p_subscriptlist_3(p):
    '''subscriptlist : subscript subscriptlist_star'''
    #                          1                  2
    args = [p[1]] + p[2]
    if all(isinstance(x, ast.Index) for x in args):
        tup = ast.Tuple([x.value for x in args], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
        inherit_lineno(tup, args[0].value)
        p[0] = ast.Index(tup, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], tup)
    else:
        p[0] = ast.ExtSlice(args, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], p[1])


def p_subscriptlist_4(p):
    '''subscriptlist : subscript subscriptlist_star COMMA'''
    #                          1                  2     3
    args = [p[1]] + p[2]
    if all(isinstance(x, ast.Index) for x in args):
        tup = ast.Tuple([x.value for x in args], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
        inherit_lineno(tup, args[0].value)
        p[0] = ast.Index(tup, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], tup)
    else:
        p[0] = ast.ExtSlice(args, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], p[1])


def p_subscriptlist_star_1(p):
    '''subscriptlist_star : COMMA subscript'''
    #                           1         2
    p[0] = [p[2]]


def p_subscriptlist_star_2(p):
    '''subscriptlist_star : subscriptlist_star COMMA subscript'''
    #                                        1     2         3
    p[0] = p[1] + [p[3]]

# subscript: '.' '.' '.' | test | [test] ':' [test] [sliceop]


def p_subscript_1(p):
    '''subscript : DOT DOT DOT'''
    #                1   2   3
    p[0] = ast.Ellipsis(rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_subscript_2(p):
    '''subscript : test'''
    #                 1
    p[0] = ast.Index(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_subscript_3(p):
    '''subscript : COLON'''
    #                  1
    p[0] = ast.Slice(None, None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_subscript_4(p):
    '''subscript : COLON sliceop'''
    #                  1       2
    p[0] = ast.Slice(None, None, p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_subscript_5(p):
    '''subscript : COLON test'''
    #                  1    2
    p[0] = ast.Slice(None, p[2], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_subscript_6(p):
    '''subscript : COLON test sliceop'''
    #                  1    2       3
    p[0] = ast.Slice(None, p[2], p[3], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_subscript_7(p):
    '''subscript : test COLON'''
    #                 1     2
    p[0] = ast.Slice(p[1], None, None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_subscript_8(p):
    '''subscript : test COLON sliceop'''
    #                 1     2       3
    p[0] = ast.Slice(p[1], None, p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_subscript_9(p):
    '''subscript : test COLON test'''
    #                 1     2    3
    p[0] = ast.Slice(p[1], p[3], None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_subscript_10(p):
    '''subscript : test COLON test sliceop'''
    #                 1     2    3       4
    p[0] = ast.Slice(p[1], p[3], p[4], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])

# sliceop: ':' [test]


def p_sliceop_1(p):
    '''sliceop : COLON'''
    #                1
    p[0] = ast.Name("None", ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_sliceop_2(p):
    '''sliceop : COLON test'''
    #                1    2
    p[0] = p[2]

# exprlist: expr (',' expr)* [',']


def p_exprlist_1(p):
    '''exprlist : expr'''
    #                1
    p[0] = p[1]


def p_exprlist_2(p):
    '''exprlist : expr COMMA'''
    #                1     2
    p[0] = ast.Tuple([p[1]], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_exprlist_3(p):
    '''exprlist : expr exprlist_star'''
    #                1             2
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_exprlist_4(p):
    '''exprlist : expr exprlist_star COMMA'''
    #                1             2     3
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_exprlist_star_1(p):
    '''exprlist_star : COMMA expr'''
    #                      1    2
    p[0] = [p[2]]


def p_exprlist_star_2(p):
    '''exprlist_star : exprlist_star COMMA expr'''
    #                              1     2    3
    p[0] = p[1] + [p[3]]

# testlist: test (',' test)* [',']


def p_testlist_1(p):
    '''testlist : test'''
    #                1
    p[0] = p[1]


def p_testlist_2(p):
    '''testlist : test COMMA'''
    #                1     2
    p[0] = ast.Tuple([p[1]], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_3(p):
    '''testlist : test testlist_star'''
    #                1             2
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_4(p):
    '''testlist : test testlist_star COMMA'''
    #                1             2     3
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist_star_1(p):
    '''testlist_star : COMMA test'''
    #                      1    2
    p[0] = [p[2]]


def p_testlist_star_2(p):
    '''testlist_star : testlist_star COMMA test'''
    #                              1     2    3
    p[0] = p[1] + [p[3]]

# dictorsetmaker: ( (test ':' test (comp_for | (',' test ':' test)* [','])) |
#                   (test (comp_for | (',' test)* [','])) )


def p_dictorsetmaker_1(p):
    '''dictorsetmaker : test COLON test comp_for'''
    #                      1     2    3        4
    p[0] = ast.DictComp(p[1], p[3], p[4], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_dictorsetmaker_2(p):
    '''dictorsetmaker : test COLON test'''
    #                      1     2    3
    p[0] = ([p[1]], [p[3]])


def p_dictorsetmaker_3(p):
    '''dictorsetmaker : test COLON test COMMA'''
    #                      1     2    3     4
    p[0] = ([p[1]], [p[3]])


def p_dictorsetmaker_4(p):
    '''dictorsetmaker : test COLON test dictorsetmaker_star'''
    #                      1     2    3                   4
    keys, values = p[4]
    p[0] = ([p[1]] + keys, [p[3]] + values)


def p_dictorsetmaker_5(p):
    '''dictorsetmaker : test COLON test dictorsetmaker_star COMMA'''
    #                      1     2    3                   4     5
    keys, values = p[4]
    p[0] = ([p[1]] + keys, [p[3]] + values)


def p_dictorsetmaker_6(p):
    '''dictorsetmaker : test comp_for'''
    #                      1        2
    p[0] = ast.SetComp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_dictorsetmaker_7(p):
    '''dictorsetmaker : test'''
    #                      1
    p[0] = (None, [p[1]])


def p_dictorsetmaker_8(p):
    '''dictorsetmaker : test COMMA'''
    #                      1     2
    p[0] = (None, [p[1]])


def p_dictorsetmaker_9(p):
    '''dictorsetmaker : test dictorsetmaker_star2'''
    #                      1                    2
    keys, values = p[2]
    p[0] = (keys, [p[1]] + values)


def p_dictorsetmaker_10(p):
    '''dictorsetmaker : test dictorsetmaker_star2 COMMA'''
    #                      1                    2     3
    keys, values = p[2]
    p[0] = (keys, [p[1]] + values)


def p_dictorsetmaker_star_1(p):
    '''dictorsetmaker_star : COMMA test COLON test'''
    #                            1    2     3    4
    p[0] = ([p[2]], [p[4]])


def p_dictorsetmaker_star_2(p):
    '''dictorsetmaker_star : dictorsetmaker_star COMMA test COLON test'''
    #                                          1     2    3     4    5
    keys, values = p[1]
    p[0] = (keys + [p[3]], values + [p[5]])


def p_dictorsetmaker_star2_1(p):
    '''dictorsetmaker_star2 : COMMA test'''
    #                             1    2
    p[0] = (None, [p[2]])


def p_dictorsetmaker_star2_2(p):
    '''dictorsetmaker_star2 : dictorsetmaker_star2 COMMA test'''
    #                                            1     2    3
    keys, values = p[1]
    p[0] = (keys, values + [p[3]])

# classdef: 'class' NAME ['(' [testlist] ')'] ':' suite


def p_classdef_1(p):
    '''classdef : CLASS NAME COLON suite'''
    #                 1    2     3     4
    p[0] = ast.ClassDef(p[2][0], [], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_classdef_2(p):
    '''classdef : CLASS NAME LPAR RPAR COLON suite'''
    #                 1    2    3    4     5     6
    p[0] = ast.ClassDef(p[2][0], [], p[6], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_classdef_3(p):
    '''classdef : CLASS NAME LPAR testlist RPAR COLON suite'''
    #                 1    2    3        4    5     6     7
    if isinstance(p[4], ast.Tuple):
        p[0] = ast.ClassDef(p[2][0], p[4].elts, p[7], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])
    else:
        p[0] = ast.ClassDef(p[2][0], [p[4]], p[7], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])

# arglist: (argument ',')* (argument [',']
#                          |'*' test (',' argument)* [',' '**' test]
#                          |'**' test)


def p_arglist_1(p):
    '''arglist : argument'''
    #                   1
    if notkeyword(p[1]):
        p[0] = ast.Call(None, [p[1]], [], None, None, rule=inspect.currentframe().f_code.co_name)
    else:
        p[0] = ast.Call(None, [], [p[1]], None, None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_2(p):
    '''arglist : argument COMMA'''
    #                   1     2
    if notkeyword(p[1]):
        p[0] = ast.Call(None, [p[1]], [], None, None, rule=inspect.currentframe().f_code.co_name)
    else:
        p[0] = ast.Call(None, [], [p[1]], None, None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_3(p):
    '''arglist : STAR test'''
    #               1    2
    p[0] = ast.Call(None, [], [], p[2], None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_4(p):
    '''arglist : STAR test COMMA DOUBLESTAR test'''
    #               1    2     3          4    5
    p[0] = ast.Call(None, [], [], p[2], p[5], rule=inspect.currentframe().f_code.co_name)


def p_arglist_5(p):
    '''arglist : STAR test arglist_star'''
    #               1    2            3
    p[0] = ast.Call(None, filter(notkeyword, p[3]), filter(iskeyword, p[3]),
                    p[2], None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_6(p):
    '''arglist : STAR test arglist_star COMMA DOUBLESTAR test'''
    #               1    2            3     4          5    6
    p[0] = ast.Call(None, filter(notkeyword, p[3]), filter(iskeyword, p[3]),
                    p[2], p[6], rule=inspect.currentframe().f_code.co_name)


def p_arglist_7(p):
    '''arglist : DOUBLESTAR test'''
    #                     1    2
    p[0] = ast.Call(None, [], [], None, p[2], rule=inspect.currentframe().f_code.co_name)


def p_arglist_8(p):
    '''arglist : arglist_star2 argument'''
    #                        1        2
    args = p[1] + [p[2]]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args),
                    None, None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_9(p):
    '''arglist : arglist_star2 argument COMMA'''
    #                        1        2     3
    args = p[1] + [p[2]]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args),
                    None, None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_10(p):
    '''arglist : arglist_star2 STAR test'''
    #                        1    2    3
    p[0] = ast.Call(None, filter(notkeyword, p[1]), filter(iskeyword, p[1]),
                    p[3], None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_11(p):
    '''arglist : arglist_star2 STAR test COMMA DOUBLESTAR test'''
    #                        1    2    3     4          5    6
    p[0] = ast.Call(None, filter(notkeyword, p[1]), filter(iskeyword, p[1]),
                    p[3], p[6], rule=inspect.currentframe().f_code.co_name)


def p_arglist_12(p):
    '''arglist : arglist_star2 STAR test arglist_star3'''
    #                        1    2    3             4
    args = p[1] + p[4]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args),
                    p[3], None, rule=inspect.currentframe().f_code.co_name)


def p_arglist_13(p):
    '''arglist : arglist_star2 STAR test arglist_star3 COMMA DOUBLESTAR test'''
    #                        1    2    3             4     5          6    7
    args = p[1] + p[4]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args),
                    p[3], p[7], rule=inspect.currentframe().f_code.co_name)


def p_arglist_14(p):
    '''arglist : arglist_star2 DOUBLESTAR test'''
    #                        1          2    3
    p[0] = ast.Call(None, filter(notkeyword, p[1]), filter(iskeyword, p[1]),
                    None, p[3], rule=inspect.currentframe().f_code.co_name)


def p_arglist_star_1(p):
    '''arglist_star : COMMA argument'''
    #                     1        2
    p[0] = [p[2]]


def p_arglist_star_2(p):
    '''arglist_star : arglist_star COMMA argument'''
    #                            1     2        3
    p[0] = p[1] + [p[3]]


def p_arglist_star3_1(p):
    '''arglist_star3 : COMMA argument'''
    #                      1        2
    p[0] = [p[2]]


def p_arglist_star3_2(p):
    '''arglist_star3 : arglist_star3 COMMA argument'''
    #                              1     2        3
    p[0] = p[1] + [p[3]]


def p_arglist_star2_1(p):
    '''arglist_star2 : argument COMMA'''
    #                         1     2
    p[0] = [p[1]]


def p_arglist_star2_2(p):
    '''arglist_star2 : arglist_star2 argument COMMA'''
    #                              1        2     3
    p[0] = p[1] + [p[2]]

# argument: test [comp_for] | test '=' test


def p_argument_1(p):
    '''argument : test'''
    #                1
    p[0] = p[1]


def p_argument_2(p):
    '''argument : test comp_for'''
    #                1        2
    p[0] = ast.GeneratorExp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])


def p_argument_3(p):
    '''argument : test EQUAL test'''
    #                1     2    3
    p[0] = ast.keyword(p[1].id, p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])

# list_iter: list_for | list_if


def p_list_iter_1(p):
    '''list_iter : list_for'''
    #                     1
    p[0] = ([], p[1])


def p_list_iter_2(p):
    '''list_iter : list_if'''
    #                    1
    p[0] = p[1]

# list_for: 'for' exprlist 'in' testlist_safe [list_iter]


def p_list_for_1(p):
    '''list_for : FOR exprlist IN testlist_safe'''
    #               1        2  3             4
    ctx_to_store(p[2])
    p[0] = [ast.comprehension(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])]


def p_list_for_2(p):
    '''list_for : FOR exprlist IN testlist_safe list_iter'''
    #               1        2  3             4         5
    ctx_to_store(p[2])
    ifs, iters = p[5]
    p[0] = [ast.comprehension(p[2], p[4], ifs, rule=inspect.currentframe().f_code.co_name, **p[1][1])] + iters

# list_if: 'if' old_test [list_iter]


def p_list_if_1(p):
    '''list_if : IF old_test'''
    #             1        2
    p[0] = ([p[2]], [])


def p_list_if_2(p):
    '''list_if : IF old_test list_iter'''
    #             1        2         3
    ifs, iters = p[3]
    p[0] = ([p[2]] + ifs, iters)

# comp_iter: comp_for | comp_if


def p_comp_iter_1(p):
    '''comp_iter : comp_for'''
    #                     1
    p[0] = ([], p[1])


def p_comp_iter_2(p):
    '''comp_iter : comp_if'''
    #                    1
    p[0] = p[1]

# comp_for: 'for' exprlist 'in' or_test [comp_iter]


def p_comp_for_1(p):
    '''comp_for : FOR exprlist IN or_test'''
    #               1        2  3       4
    ctx_to_store(p[2])
    p[0] = [ast.comprehension(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])]


def p_comp_for_2(p):
    '''comp_for : FOR exprlist IN or_test comp_iter'''
    #               1        2  3       4         5
    ctx_to_store(p[2])
    ifs, iters = p[5]
    p[0] = [ast.comprehension(p[2], p[4], ifs, rule=inspect.currentframe().f_code.co_name, **p[1][1])] + iters

# comp_if: 'if' old_test [comp_iter]


def p_comp_if_1(p):
    '''comp_if : IF old_test'''
    #             1        2
    p[0] = ([p[2]], [])


def p_comp_if_2(p):
    '''comp_if : IF old_test comp_iter'''
    #             1        2         3
    ifs, iters = p[3]
    p[0] = ([p[2]] + ifs, iters)

# testlist1: test (',' test)*


def p_testlist1_1(p):
    '''testlist1 : test'''
    #                 1
    p[0] = p[1]


def p_testlist1_2(p):
    '''testlist1 : test testlist1_star'''
    #                 1              2
    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])


def p_testlist1_star_1(p):
    '''testlist1_star : COMMA test'''
    #                       1    2
    p[0] = [p[2]]


def p_testlist1_star_2(p):
    '''testlist1_star : testlist1_star COMMA test'''
    #                                1     2    3
    p[0] = p[1] + [p[3]]

# encoding_decl: NAME


def p_encoding_decl(p):
    '''encoding_decl : NAME'''
    #                     1
    p[0] = p[1]

# yield_expr: 'yield' [testlist]


def p_yield_expr_1(p):
    '''yield_expr : YIELD'''
    #                   1
    p[0] = ast.Yield(None, rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_yield_expr_2(p):
    '''yield_expr : YIELD testlist'''
    #                   1        2
    p[0] = ast.Yield(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])


def p_error(p):
    lexer = p.lexer
    if isinstance(lexer, PythonLexer):
        lexer = lexer.lexer
    pos = lexer.kwds(lexer.lexpos)
    line = re.split("\r?\n", lexer.source)[max(pos["lineno"] - 3, 0):pos["lineno"]]
    indicator = "-" * pos["col_offset"] + "^"
    raise SyntaxError("invalid syntax\n  File " +
                      lexer.fileName +
                      ", line " +
                      str(pos["lineno"]) +
                      " col " +
                      str(pos["col_offset"]) +
                      "\n    " +
                      "\n    ".join(line) +
                      "\n----" +
                      indicator)


def parse(source, fileName="<unknown>"):
    lexer = PythonLexer(fileName=fileName)
    parser = yacc.yacc(debug=False, write_tables=True, tabmodule="hgawk_grammar_table", errorlog=yacc.NullLogger())
    return parser.parse(source + "\n", lexer=lexer)
