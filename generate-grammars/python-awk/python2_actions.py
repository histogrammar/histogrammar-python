#!/usr/bin/env python

actions = {}
asts = []

# hgawk
asts.append('''class DollarNumber(ast.expr):
    _fields = ("n",)
    def __init__(self, n, **kwds):
        self.n = n
        self.__dict__.update(kwds)
''')
actions['''atom : DOLLARNUMBER'''] = '''    p[0] = DollarNumber(int(p[1][0][1:]), **p[1][1])'''

# Python
actions['''file_input : ENDMARKER'''] = '''    p[0] = ast.Module([], rule=inspect.currentframe().f_code.co_name, lineno=0, col_offset=0)'''
actions['''file_input : file_input_star ENDMARKER'''] = '''    p[0] = ast.Module(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1][0])'''
actions['''file_input_star : NEWLINE'''] = '''    p[0] = ast.Module([], rule=inspect.currentframe().f_code.co_name, lineno=0, col_offset=0)'''
actions['''file_input_star : stmt'''] = '''    p[0] = p[1]'''
actions['''file_input_star : file_input_star NEWLINE'''] = '''    p[0] = ast.Module(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1][0])'''
actions['''file_input_star : file_input_star stmt'''] = '''    p[0] = p[1] + p[2]'''
actions['''decorator : AT dotted_name NEWLINE'''] = '''    p[0] = p[2]
    p[0].alt = p[1][1]'''
actions['''decorator : AT dotted_name LPAR RPAR NEWLINE'''] = '''    p[0] = ast.Call(p[2], [], [], None, None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1][1])'''
actions['''decorator : AT dotted_name LPAR arglist RPAR NEWLINE'''] = '''    p[4].func = p[2]
    p[0] = p[4]
    inherit_lineno(p[0], p[2])
    p[0].alt = p[1][1]'''
actions['''decorators : decorators_plus'''] = '''    p[0] = p[1]'''
actions['''decorators_plus : decorator'''] = '''    p[0] = [p[1]]'''
actions['''decorators_plus : decorators_plus decorator'''] = '''    p[0] = p[1] + [p[2]]'''
actions['''decorated : decorators classdef'''] = '''    p[2].decorator_list = p[1]
    p[0] = p[2]
    inherit_lineno(p[0], p[1][0])'''
actions['''decorated : decorators funcdef'''] = '''    p[2].decorator_list = p[1]
    p[0] = p[2]
    inherit_lineno(p[0], p[1][0])'''
actions['''funcdef : DEF NAME parameters COLON suite'''] = '''    p[0] = ast.FunctionDef(p[2][0], p[3], p[5], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''parameters : LPAR RPAR'''] = '''    p[0] = ast.arguments([], None, None, [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''parameters : LPAR varargslist RPAR'''] = '''    p[0] = p[2]'''
actions['''varargslist : fpdef COMMA STAR NAME'''] = '''    p[0] = ast.arguments([p[1]], p[4][0], None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef COMMA STAR NAME COMMA DOUBLESTAR NAME'''] = '''    p[0] = ast.arguments([p[1]], p[4][0], p[7][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef COMMA DOUBLESTAR NAME'''] = '''    p[0] = ast.arguments([p[1]], None, p[4][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef'''] = '''    p[0] = ast.arguments([p[1]], None, None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef COMMA'''] = '''    p[0] = ast.arguments([p[1]], None, None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef varargslist_star COMMA STAR NAME'''] = '''    p[2].args.insert(0, p[1])
    p[2].vararg = p[5][0]
    p[0] = p[2]'''
actions['''varargslist : fpdef varargslist_star COMMA STAR NAME COMMA DOUBLESTAR NAME'''] = '''    p[2].args.insert(0, p[1])
    p[2].vararg = p[5][0]
    p[2].kwarg = p[8][0]
    p[0] = p[2]'''
actions['''varargslist : fpdef varargslist_star COMMA DOUBLESTAR NAME'''] = '''    p[2].args.insert(0, p[1])
    p[2].kwarg = p[5][0]
    p[0] = p[2]'''
actions['''varargslist : fpdef varargslist_star'''] = '''    p[2].args.insert(0, p[1])
    p[0] = p[2]'''
actions['''varargslist : fpdef varargslist_star COMMA'''] = '''    p[2].args.insert(0, p[1])
    p[0] = p[2]'''
actions['''varargslist : fpdef EQUAL test COMMA STAR NAME'''] = '''    p[0] = ast.arguments([p[1]], p[6][0], None, [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef EQUAL test COMMA STAR NAME COMMA DOUBLESTAR NAME'''] = '''    p[0] = ast.arguments([p[1]], p[6][0], p[9][0], [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef EQUAL test COMMA DOUBLESTAR NAME'''] = '''    p[0] = ast.arguments([p[1]], None, p[6][0], [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef EQUAL test'''] = '''    p[0] = ast.arguments([p[1]], None, None, [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef EQUAL test COMMA'''] = '''    p[0] = ast.arguments([p[1]], None, None, [p[3]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''varargslist : fpdef EQUAL test varargslist_star COMMA STAR NAME'''] = '''    p[4].args.insert(0, p[1])
    p[4].vararg = p[7][0]
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]'''
actions['''varargslist : fpdef EQUAL test varargslist_star COMMA STAR NAME COMMA DOUBLESTAR NAME'''] = '''    p[4].args.insert(0, p[1])
    p[4].vararg = p[7][0]
    p[4].kwarg = p[10][0]
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]'''
actions['''varargslist : fpdef EQUAL test varargslist_star COMMA DOUBLESTAR NAME'''] = '''    p[4].args.insert(0, p[1])
    p[4].kwarg = p[7][0]
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]'''
actions['''varargslist : fpdef EQUAL test varargslist_star'''] = '''    p[4].args.insert(0, p[1])
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]'''
actions['''varargslist : fpdef EQUAL test varargslist_star COMMA'''] = '''    p[4].args.insert(0, p[1])
    p[4].defaults.insert(0, p[3])
    p[0] = p[4]'''
actions['''varargslist : STAR NAME'''] = '''    p[0] = ast.arguments([], p[2][0], None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2][1])'''
actions['''varargslist : STAR NAME COMMA DOUBLESTAR NAME'''] = '''    p[0] = ast.arguments([], p[2][0], p[5][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2][1])'''
actions['''varargslist : DOUBLESTAR NAME'''] = '''    p[0] = ast.arguments([], None, p[2][0], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2][1])'''
actions['''varargslist_star : COMMA fpdef'''] = '''    p[0] = ast.arguments([p[2]], None, None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2])'''
actions['''varargslist_star : COMMA fpdef EQUAL test'''] = '''    p[0] = ast.arguments([p[2]], None, None, [p[4]], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2])'''
actions['''varargslist_star : varargslist_star COMMA fpdef'''] = '''    p[1].args.append(p[3])
    p[0] = p[1]'''
actions['''varargslist_star : varargslist_star COMMA fpdef EQUAL test'''] = '''    p[1].args.append(p[3])
    p[1].defaults.append(p[5])
    p[0] = p[1]'''
actions['''fpdef : NAME'''] = '''    p[0] = ast.Name(p[1][0], ast.Param(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''fpdef : LPAR fplist RPAR'''] = '''    if isinstance(p[2], ast.Tuple):
        p[2].paren = True
        ctx_to_store(p[2])
    p[0] = p[2]'''
actions['''fplist : fpdef'''] = '''    p[0] = p[1]'''
actions['''fplist : fpdef COMMA'''] = '''    p[0] = ast.Tuple([p[1]], ast.Param(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''fplist : fpdef fplist_star'''] = '''    p[2].elts.insert(0, p[1])
    p[0] = p[2]
    inherit_lineno(p[0], p[1])'''
actions['''fplist : fpdef fplist_star COMMA'''] = '''    p[2].elts.insert(0, p[1])
    p[0] = p[2]
    inherit_lineno(p[0], p[1])'''
actions['''fplist_star : COMMA fpdef'''] = '''    p[0] = ast.Tuple([p[2]], ast.Param(), rule=inspect.currentframe().f_code.co_name, paren=False)'''
actions['''fplist_star : fplist_star COMMA fpdef'''] = '''    p[1].elts.append(p[3])
    p[0] = p[1]'''
actions['''stmt : simple_stmt'''] = '''    p[0] = p[1]'''
actions['''stmt : compound_stmt'''] = '''    p[0] = p[1]'''
actions['''simple_stmt : small_stmt NEWLINE'''] = '''    p[0] = [p[1]]'''
actions['''simple_stmt : small_stmt SEMI NEWLINE'''] = '''    p[0] = [p[1]]'''
actions['''simple_stmt : small_stmt simple_stmt_star NEWLINE'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''simple_stmt : small_stmt simple_stmt_star SEMI NEWLINE'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''simple_stmt_star : SEMI small_stmt'''] = '''    p[0] = [p[2]]'''
actions['''simple_stmt_star : simple_stmt_star SEMI small_stmt'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''small_stmt : expr_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : print_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : del_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : pass_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : flow_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : import_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : global_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : exec_stmt'''] = '''    p[0] = p[1]'''
actions['''small_stmt : assert_stmt'''] = '''    p[0] = p[1]'''
actions['''expr_stmt : testlist augassign yield_expr'''] = '''    ctx_to_store(p[1])
    p[0] = ast.AugAssign(p[1], p[2], p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''expr_stmt : testlist augassign testlist'''] = '''    ctx_to_store(p[1])
    p[0] = ast.AugAssign(p[1], p[2], p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''expr_stmt : testlist'''] = '''    p[0] = ast.Expr(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''expr_stmt : testlist expr_stmt_star'''] = '''    everything = [p[1]] + p[2]
    targets, value = everything[:-1], everything[-1]
    ctx_to_store(targets)
    p[0] = ast.Assign(targets, value, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], targets[0])'''
actions['''expr_stmt_star : EQUAL yield_expr'''] = '''    p[0] = [p[2]]'''
actions['''expr_stmt_star : EQUAL testlist'''] = '''    p[0] = [p[2]]'''
actions['''expr_stmt_star : expr_stmt_star EQUAL yield_expr'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''expr_stmt_star : expr_stmt_star EQUAL testlist'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''augassign : PLUSEQUAL'''] = '''    p[0] = ast.Add(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : MINEQUAL'''] = '''    p[0] = ast.Sub(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : STAREQUAL'''] = '''    p[0] = ast.Mult(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : SLASHEQUAL'''] = '''    p[0] = ast.Div(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : PERCENTEQUAL'''] = '''    p[0] = ast.Mod(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : AMPEREQUAL'''] = '''    p[0] = ast.BitAnd(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : VBAREQUAL'''] = '''    p[0] = ast.BitOr(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : CIRCUMFLEXEQUAL'''] = '''    p[0] = ast.BitXor(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : LEFTSHIFTEQUAL'''] = '''    p[0] = ast.LShift(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : RIGHTSHIFTEQUAL'''] = '''    p[0] = ast.RShift(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : DOUBLESTAREQUAL'''] = '''    p[0] = ast.Pow(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''augassign : DOUBLESLASHEQUAL'''] = '''    p[0] = ast.FloorDiv(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT'''] = '''    p[0] = ast.Print(None, [], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT test'''] = '''    p[0] = ast.Print(None, [p[2]], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT test COMMA'''] = '''    p[0] = ast.Print(None, [p[2]], False, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT test print_stmt_plus'''] = '''    p[0] = ast.Print(None, [p[2]] + p[3], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT test print_stmt_plus COMMA'''] = '''    p[0] = ast.Print(None, [p[2]] + p[3], False, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT RIGHTSHIFT test'''] = '''    p[0] = ast.Print(p[3], [], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT RIGHTSHIFT test print_stmt_plus'''] = '''    p[0] = ast.Print(p[3], p[4], True, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt : PRINT RIGHTSHIFT test print_stmt_plus COMMA'''] = '''    p[0] = ast.Print(p[3], p[4], False, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''print_stmt_plus : COMMA test'''] = '''    p[0] = [p[2]]'''
actions['''print_stmt_plus : print_stmt_plus COMMA test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''del_stmt : DEL exprlist'''] = '''    ctx_to_store(p[2], ast.Del)          # interesting fact: evaluating Delete nodes with ctx=Store() causes a segmentation fault in Python!
    if isinstance(p[2], ast.Tuple) and not p[2].paren:
        p[0] = ast.Delete(p[2].elts, rule=inspect.currentframe().f_code.co_name, **p[1][1])
    else:
        p[0] = ast.Delete([p[2]], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''pass_stmt : PASS'''] = '''    p[0] = ast.Pass(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''flow_stmt : break_stmt'''] = '''    p[0] = p[1]'''
actions['''flow_stmt : continue_stmt'''] = '''    p[0] = p[1]'''
actions['''flow_stmt : return_stmt'''] = '''    p[0] = p[1]'''
actions['''flow_stmt : raise_stmt'''] = '''    p[0] = p[1]'''
actions['''flow_stmt : yield_stmt'''] = '''    p[0] = ast.Expr(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''break_stmt : BREAK'''] = '''    p[0] = ast.Break(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''continue_stmt : CONTINUE'''] = '''    p[0] = ast.Continue(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''return_stmt : RETURN'''] = '''    p[0] = ast.Return(None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''return_stmt : RETURN testlist'''] = '''    p[0] = ast.Return(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''yield_stmt : yield_expr'''] = '''    p[0] = p[1]'''
actions['''raise_stmt : RAISE'''] = '''    p[0] = ast.Raise(None, None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''raise_stmt : RAISE test'''] = '''    p[0] = ast.Raise(p[2], None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''raise_stmt : RAISE test COMMA test'''] = '''    p[0] = ast.Raise(p[2], p[4], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''raise_stmt : RAISE test COMMA test COMMA test'''] = '''    p[0] = ast.Raise(p[2], p[4], p[6], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_stmt : import_name'''] = '''    p[0] = p[1]'''
actions['''import_stmt : import_from'''] = '''    p[0] = p[1]'''
actions['''import_name : IMPORT dotted_as_names'''] = '''    p[0] = ast.Import(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM dotted_name IMPORT STAR'''] = '''    dotted = []
    last = p[2]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), [ast.alias("*", None, rule=inspect.currentframe().f_code.co_name, **p[3][1])], 0, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM dotted_name IMPORT LPAR import_as_names RPAR'''] = '''    dotted = []
    last = p[2]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[5], 0, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM dotted_name IMPORT import_as_names'''] = '''    dotted = []
    last = p[2]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[4], 0, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM import_from_plus dotted_name IMPORT STAR'''] = '''    dotted = []
    last = p[3]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), [ast.alias("*", None, rule=inspect.currentframe().f_code.co_name, **p[4][1])], p[2], **p[1][1])'''
actions['''import_from : FROM import_from_plus dotted_name IMPORT LPAR import_as_names RPAR'''] = '''    dotted = []
    last = p[3]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[6], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM import_from_plus dotted_name IMPORT import_as_names'''] = '''    dotted = []
    last = p[3]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.ImportFrom(".".join(dotted), p[5], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM import_from_plus IMPORT STAR'''] = '''    p[0] = ast.ImportFrom(None, [ast.alias("*", None, rule=inspect.currentframe().f_code.co_name, **p[3][1])], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM import_from_plus IMPORT LPAR import_as_names RPAR'''] = '''    p[0] = ast.ImportFrom(None, p[5], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from : FROM import_from_plus IMPORT import_as_names'''] = '''    p[0] = ast.ImportFrom(None, p[4], p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_from_plus : DOT'''] = '''    p[0] = 1'''
actions['''import_from_plus : import_from_plus DOT'''] = '''    p[0] = p[1] + 1'''
actions['''import_as_name : NAME'''] = '''    p[0] = ast.alias(p[1][0], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''import_as_name : NAME AS NAME'''] = '''    p[0] = ast.alias(p[1][0], p[3][0], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''dotted_as_name : dotted_name'''] = '''    dotted = []
    last = p[1]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.alias(".".join(dotted), None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''dotted_as_name : dotted_name AS NAME'''] = '''    dotted = []
    last = p[1]
    while isinstance(last, ast.Attribute):
        dotted.insert(0, last.attr)
        last = last.value
    dotted.insert(0, last.id)
    p[0] = ast.alias(".".join(dotted), p[3][0], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''import_as_names : import_as_name'''] = '''    p[0] = [p[1]]'''
actions['''import_as_names : import_as_name COMMA'''] = '''    p[0] = [p[1]]'''
actions['''import_as_names : import_as_name import_as_names_star'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''import_as_names : import_as_name import_as_names_star COMMA'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''import_as_names_star : COMMA import_as_name'''] = '''    p[0] = [p[2]]'''
actions['''import_as_names_star : import_as_names_star COMMA import_as_name'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''dotted_as_names : dotted_as_name'''] = '''    p[0] = [p[1]]'''
actions['''dotted_as_names : dotted_as_name dotted_as_names_star'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''dotted_as_names_star : COMMA dotted_as_name'''] = '''    p[0] = [p[2]]'''
actions['''dotted_as_names_star : dotted_as_names_star COMMA dotted_as_name'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''dotted_name : NAME'''] = '''    p[0] = ast.Name(p[1][0], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''dotted_name : NAME dotted_name_star'''] = '''    last = p[2]
    if isinstance(last, ast.Attribute):
        inherit_lineno(last, p[1][1])
        while isinstance(last.value, ast.Attribute):
            last = last.value
            inherit_lineno(last, p[1][1])
        last.value = ast.Attribute(ast.Name(p[1][0], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1]), last.value, ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])
        p[0] = p[2]
    else:
        p[0] = ast.Attribute(ast.Name(p[1][0], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''dotted_name_star : DOT NAME'''] = '''    p[0] = p[2][0]'''
actions['''dotted_name_star : dotted_name_star DOT NAME'''] = '''    p[0] = ast.Attribute(p[1], p[3][0], ast.Load(), rule=inspect.currentframe().f_code.co_name)'''
actions['''global_stmt : GLOBAL NAME'''] = '''    p[0] = ast.Global([p[2][0]], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''global_stmt : GLOBAL NAME global_stmt_star'''] = '''    p[0] = ast.Global([p[2][0]] + p[3], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''global_stmt_star : COMMA NAME'''] = '''    p[0] = [p[2][0]]'''
actions['''global_stmt_star : global_stmt_star COMMA NAME'''] = '''    p[0] = p[1] + [p[3][0]]'''
actions['''exec_stmt : EXEC expr'''] = '''    p[0] = ast.Exec(p[2], None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''exec_stmt : EXEC expr IN test'''] = '''    p[0] = ast.Exec(p[2], p[4], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''exec_stmt : EXEC expr IN test COMMA test'''] = '''    p[0] = ast.Exec(p[2], p[4], p[6], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''assert_stmt : ASSERT test'''] = '''    p[0] = ast.Assert(p[2], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''assert_stmt : ASSERT test COMMA test'''] = '''    p[0] = ast.Assert(p[2], p[4], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''compound_stmt : if_stmt'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : while_stmt'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : for_stmt'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : try_stmt'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : with_stmt'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : funcdef'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : classdef'''] = '''    p[0] = [p[1]]'''
actions['''compound_stmt : decorated'''] = '''    p[0] = [p[1]]'''
actions['''if_stmt : IF test COLON suite'''] = '''    p[0] = ast.If(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''if_stmt : IF test COLON suite ELSE COLON suite'''] = '''    p[0] = ast.If(p[2], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''if_stmt : IF test COLON suite if_stmt_star'''] = '''    p[0] = ast.If(p[2], p[4], [p[5]], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''if_stmt : IF test COLON suite if_stmt_star ELSE COLON suite'''] = '''    last = p[5]
    while len(last.orelse) > 0:
        last = last.orelse[0]
    last.orelse.extend(p[8])
    p[0] = ast.If(p[2], p[4], [p[5]], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''if_stmt_star : ELIF test COLON suite'''] = '''    p[0] = ast.If(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[2])'''
actions['''if_stmt_star : if_stmt_star ELIF test COLON suite'''] = '''    last = p[1]
    while len(last.orelse) > 0:
        last = last.orelse[0]
    last.orelse.append(ast.If(p[3], p[5], [], rule=inspect.currentframe().f_code.co_name))
    inherit_lineno(last.orelse[-1], p[3])
    p[0] = p[1]'''
actions['''while_stmt : WHILE test COLON suite'''] = '''    p[0] = ast.While(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''while_stmt : WHILE test COLON suite ELSE COLON suite'''] = '''    p[0] = ast.While(p[2], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''for_stmt : FOR exprlist IN testlist COLON suite'''] = '''    ctx_to_store(p[2])
    p[0] = ast.For(p[2], p[4], p[6], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''for_stmt : FOR exprlist IN testlist COLON suite ELSE COLON suite'''] = '''    ctx_to_store(p[2])
    p[0] = ast.For(p[2], p[4], p[6], p[9], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''try_stmt : TRY COLON suite try_stmt_plus'''] = '''    p[0] = ast.TryExcept(p[3], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''try_stmt : TRY COLON suite try_stmt_plus FINALLY COLON suite'''] = '''    p[0] = ast.TryFinally([ast.TryExcept(p[3], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''try_stmt : TRY COLON suite try_stmt_plus ELSE COLON suite'''] = '''    p[0] = ast.TryExcept(p[3], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''try_stmt : TRY COLON suite try_stmt_plus ELSE COLON suite FINALLY COLON suite'''] = '''    p[0] = ast.TryFinally([ast.TryExcept(p[3], p[4], p[7], rule=inspect.currentframe().f_code.co_name, **p[1][1])], p[10], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''try_stmt : TRY COLON suite FINALLY COLON suite'''] = '''    p[0] = ast.TryFinally(p[3], p[6], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''try_stmt_plus : except_clause COLON suite'''] = '''    p[1].body = p[3]
    p[0] = [p[1]]'''
actions['''try_stmt_plus : try_stmt_plus except_clause COLON suite'''] = '''    p[2].body = p[4]
    p[0] = p[1] + [p[2]]'''
actions['''with_stmt : WITH with_item COLON suite'''] = '''    p[2].body = p[4]
    p[0] = p[2]'''
actions['''with_stmt : WITH with_item with_stmt_star COLON suite'''] = '''    p[2].body.append(p[3])
    last = p[2]
    while len(last.body) > 0:
        last = last.body[0]
    last.body = p[5]
    p[0] = p[2]'''
actions['''with_stmt_star : COMMA with_item'''] = '''    p[0] = p[2]'''
actions['''with_stmt_star : with_stmt_star COMMA with_item'''] = '''    last = p[1]
    while len(last.body) > 0:
        last = last.body[0]
    last.body.append(p[3])
    p[0] = p[1]'''
actions['''with_item : test'''] = '''    p[0] = ast.With(p[1], None, [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''with_item : test AS expr'''] = '''    ctx_to_store(p[3])
    p[0] = ast.With(p[1], p[3], [], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''except_clause : EXCEPT'''] = '''    p[0] = ast.ExceptHandler(None, None, [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''except_clause : EXCEPT test'''] = '''    p[0] = ast.ExceptHandler(p[2], None, [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''except_clause : EXCEPT test AS test'''] = '''    ctx_to_store(p[4])
    p[0] = ast.ExceptHandler(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''except_clause : EXCEPT test COMMA test'''] = '''    ctx_to_store(p[4])
    p[0] = ast.ExceptHandler(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''suite : simple_stmt'''] = '''    p[0] = p[1]'''
actions['''suite : NEWLINE INDENT suite_plus DEDENT'''] = '''    p[0] = p[3]'''
actions['''suite_plus : stmt'''] = '''    p[0] = p[1]'''
actions['''suite_plus : suite_plus stmt'''] = '''    p[0] = p[1] + p[2]'''
actions['''testlist_safe : old_test'''] = '''    p[0] = p[1]'''
actions['''testlist_safe : old_test testlist_safe_plus'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_safe : old_test testlist_safe_plus COMMA'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_safe_plus : COMMA old_test'''] = '''    p[0] = [p[2]]'''
actions['''testlist_safe_plus : testlist_safe_plus COMMA old_test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''old_test : or_test'''] = '''    p[0] = p[1]'''
actions['''old_test : old_lambdef'''] = '''    p[0] = p[1]'''
actions['''old_lambdef : LAMBDA COLON old_test'''] = '''    p[0] = ast.Lambda(ast.arguments([], None, None, [], rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''old_lambdef : LAMBDA varargslist COLON old_test'''] = '''    p[0] = ast.Lambda(p[2], p[4], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''test : or_test'''] = '''    p[0] = p[1]'''
actions['''test : or_test IF or_test ELSE test'''] = '''    p[0] = ast.IfExp(p[3], p[1], p[5], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''test : lambdef'''] = '''    p[0] = p[1]'''
actions['''or_test : and_test'''] = '''    p[0] = p[1]'''
actions['''or_test : and_test or_test_star'''] = '''    theor = ast.Or(rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(theor, p[2][0])
    p[0] = ast.BoolOp(theor, [p[1]] + p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''or_test_star : OR and_test'''] = '''    p[0] = [p[2]]'''
actions['''or_test_star : or_test_star OR and_test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''and_test : not_test'''] = '''    p[0] = p[1]'''
actions['''and_test : not_test and_test_star'''] = '''    theand = ast.And(rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(theand, p[2][0])
    p[0] = ast.BoolOp(theand, [p[1]] + p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''and_test_star : AND not_test'''] = '''    p[0] = [p[2]]'''
actions['''and_test_star : and_test_star AND not_test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''not_test : NOT not_test'''] = '''    thenot = ast.Not(rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(thenot, p[2])
    p[0] = ast.UnaryOp(thenot, p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''not_test : comparison'''] = '''    p[0] = p[1]'''
actions['''comparison : expr'''] = '''    p[0] = p[1]'''
actions['''comparison : expr comparison_star'''] = '''    ops, exprs = p[2]
    p[0] = ast.Compare(p[1], ops, exprs, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''comparison_star : comp_op expr'''] = '''    inherit_lineno(p[1], p[2])
    p[0] = ([p[1]], [p[2]])'''
actions['''comparison_star : comparison_star comp_op expr'''] = '''    ops, exprs = p[1]
    inherit_lineno(p[2], p[3])
    p[0] = (ops + [p[2]], exprs + [p[3]])'''
actions['''comp_op : LESS'''] = '''    p[0] = ast.Lt(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : GREATER'''] = '''    p[0] = ast.Gt(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : EQEQUAL'''] = '''    p[0] = ast.Eq(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : GREATEREQUAL'''] = '''    p[0] = ast.GtE(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : LESSEQUAL'''] = '''    p[0] = ast.LtE(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : NOTEQUAL'''] = '''    p[0] = ast.NotEq(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : IN'''] = '''    p[0] = ast.In(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : NOT IN'''] = '''    p[0] = ast.NotIn(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : IS'''] = '''    p[0] = ast.Is(rule=inspect.currentframe().f_code.co_name)'''
actions['''comp_op : IS NOT'''] = '''    p[0] = ast.IsNot(rule=inspect.currentframe().f_code.co_name)'''
actions['''expr : xor_expr'''] = '''    p[0] = p[1]'''
actions['''expr : xor_expr expr_star'''] = '''    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)'''
actions['''expr_star : VBAR xor_expr'''] = '''    p[0] = [ast.BitOr(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''expr_star : expr_star VBAR xor_expr'''] = '''    p[0] = p[1] + [ast.BitOr(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''xor_expr : and_expr'''] = '''    p[0] = p[1]'''
actions['''xor_expr : and_expr xor_expr_star'''] = '''    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)'''
actions['''xor_expr_star : CIRCUMFLEX and_expr'''] = '''    p[0] = [ast.BitXor(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''xor_expr_star : xor_expr_star CIRCUMFLEX and_expr'''] = '''    p[0] = p[1] + [ast.BitXor(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''and_expr : shift_expr'''] = '''    p[0] = p[1]'''
actions['''and_expr : shift_expr and_expr_star'''] = '''    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 0)'''
actions['''and_expr_star : AMPER shift_expr'''] = '''    p[0] = [ast.BitAnd(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''and_expr_star : and_expr_star AMPER shift_expr'''] = '''    p[0] = p[1] + [ast.BitAnd(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''shift_expr : arith_expr'''] = '''    p[0] = p[1]'''
actions['''shift_expr : arith_expr shift_expr_star'''] = '''    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)'''
actions['''shift_expr_star : LEFTSHIFT arith_expr'''] = '''    p[0] = [ast.LShift(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''shift_expr_star : RIGHTSHIFT arith_expr'''] = '''    p[0] = [ast.RShift(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''shift_expr_star : shift_expr_star LEFTSHIFT arith_expr'''] = '''    p[0] = p[1] + [ast.LShift(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''shift_expr_star : shift_expr_star RIGHTSHIFT arith_expr'''] = '''    p[0] = p[1] + [ast.RShift(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''arith_expr : term'''] = '''    p[0] = p[1]'''
actions['''arith_expr : term arith_expr_star'''] = '''    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)'''
actions['''arith_expr_star : PLUS term'''] = '''    p[0] = [ast.Add(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''arith_expr_star : MINUS term'''] = '''    p[0] = [ast.Sub(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''arith_expr_star : arith_expr_star PLUS term'''] = '''    p[0] = p[1] + [ast.Add(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''arith_expr_star : arith_expr_star MINUS term'''] = '''    p[0] = p[1] + [ast.Sub(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''term : factor'''] = '''    p[0] = p[1]'''
actions['''term : factor term_star'''] = '''    p[0] = unwrap_left_associative([p[1]] + p[2], rule=inspect.currentframe().f_code.co_name, alt=len(p[2]) > 2)'''
actions['''term_star : STAR factor'''] = '''    p[0] = [ast.Mult(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''term_star : SLASH factor'''] = '''    p[0] = [ast.Div(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''term_star : PERCENT factor'''] = '''    p[0] = [ast.Mod(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''term_star : DOUBLESLASH factor'''] = '''    p[0] = [ast.FloorDiv(rule=inspect.currentframe().f_code.co_name, **p[1][1]), p[2]]'''
actions['''term_star : term_star STAR factor'''] = '''    p[0] = p[1] + [ast.Mult(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''term_star : term_star SLASH factor'''] = '''    p[0] = p[1] + [ast.Div(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''term_star : term_star PERCENT factor'''] = '''    p[0] = p[1] + [ast.Mod(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''term_star : term_star DOUBLESLASH factor'''] = '''    p[0] = p[1] + [ast.FloorDiv(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3]]'''
actions['''factor : PLUS factor'''] = '''    op = ast.UAdd(rule=inspect.currentframe().f_code.co_name, **p[1][1])
    p[0] = ast.UnaryOp(op, p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], op)'''
actions['''factor : MINUS factor'''] = '''    if isinstance(p[2], ast.Num) and not hasattr(p[2], "unary"):
        p[2].n *= -1
        p[0] = p[2]
        p[0].unary = True
        inherit_lineno(p[0], p[1][1])
    else:
        op = ast.USub(rule=inspect.currentframe().f_code.co_name, **p[1][1])
        p[0] = ast.UnaryOp(op, p[2], rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], op)'''
actions['''factor : TILDE factor'''] = '''    op = ast.Invert(rule=inspect.currentframe().f_code.co_name, **p[1][1])
    p[0] = ast.UnaryOp(op, p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], op)'''
actions['''factor : power'''] = '''    p[0] = p[1]'''
actions['''power : atom'''] = '''    p[0] = p[1]'''
actions['''power : atom DOUBLESTAR factor'''] = '''    p[0] = ast.BinOp(p[1], ast.Pow(rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''power : atom power_star'''] = '''    p[0] = unpack_trailer(p[1], p[2])'''
actions['''power : atom power_star DOUBLESTAR factor'''] = '''    p[0] = ast.BinOp(unpack_trailer(p[1], p[2]), ast.Pow(rule=inspect.currentframe().f_code.co_name, **p[3][1]), p[4], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''power_star : trailer'''] = '''    p[0] = [p[1]]'''
actions['''power_star : power_star trailer'''] = '''    p[0] = p[1] + [p[2]]'''
actions['''atom : LPAR RPAR'''] = '''    p[0] = ast.Tuple([], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=True, **p[1][1])'''
actions['''atom : LPAR yield_expr RPAR'''] = '''    p[0] = p[2]
    if isinstance(p[0], ast.Tuple):
        p[0].paren = True
    p[0].alt = p[1][1]'''
actions['''atom : LPAR testlist_comp RPAR'''] = '''    p[0] = p[2]
    if isinstance(p[0], ast.Tuple):
        p[0].paren = True
    p[0].alt = p[1][1]'''
actions['''atom : LSQB RSQB'''] = '''    p[0] = ast.List([], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom : LSQB listmaker RSQB'''] = '''    if isinstance(p[2], ast.ListComp):
        p[0] = p[2]
        p[0].alt = p[1][1]
    else:
        p[0] = ast.List(p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom : LBRACE RBRACE'''] = '''    p[0] = ast.Dict([], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom : LBRACE dictorsetmaker RBRACE'''] = '''    if isinstance(p[2], (ast.SetComp, ast.DictComp)):
        p[0] = p[2]
        p[0].alt = p[1][1]
    else:
        keys, values = p[2]
        if keys is None:
            p[0] = ast.Set(values, rule=inspect.currentframe().f_code.co_name, **p[1][1])
        else:
            p[0] = ast.Dict(keys, values, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom : BACKQUOTE testlist1 BACKQUOTE'''] = '''    p[0] = ast.Repr(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom : NAME'''] = '''    p[0] = ast.Name(p[1][0], ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom : NUMBER'''] = '''    p[0] = ast.Num(p[1][0], rule=inspect.currentframe().f_code.co_name, **p[1][2])'''
actions['''atom : atom_plus'''] = '''    p[0] = p[1]'''
actions['''atom_plus : STRING'''] = '''    p[0] = ast.Str(p[1][0], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''atom_plus : atom_plus STRING'''] = '''    p[1].s = p[1].s + p[2][0]
    p[0] = p[1]'''
actions['''listmaker : test list_for'''] = '''    p[0] = ast.ListComp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''listmaker : test'''] = '''    p[0] = [p[1]]'''
actions['''listmaker : test COMMA'''] = '''    p[0] = [p[1]]'''
actions['''listmaker : test listmaker_star'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''listmaker : test listmaker_star COMMA'''] = '''    p[0] = [p[1]] + p[2]'''
actions['''listmaker_star : COMMA test'''] = '''    p[0] = [p[2]]'''
actions['''listmaker_star : listmaker_star COMMA test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''testlist_comp : test comp_for'''] = '''    p[0] = ast.GeneratorExp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_comp : test'''] = '''    p[0] = p[1]'''
actions['''testlist_comp : test COMMA'''] = '''    p[0] = ast.Tuple([p[1]], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_comp : test testlist_comp_star'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_comp : test testlist_comp_star COMMA'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_comp_star : COMMA test'''] = '''    p[0] = [p[2]]'''
actions['''testlist_comp_star : testlist_comp_star COMMA test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''lambdef : LAMBDA COLON test'''] = '''    p[0] = ast.Lambda(ast.arguments([], None, None, [], rule=inspect.currentframe().f_code.co_name, **p[2][1]), p[3], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''lambdef : LAMBDA varargslist COLON test'''] = '''    p[0] = ast.Lambda(p[2], p[4], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''trailer : LPAR RPAR'''] = '''    p[0] = ast.Call(None, [], [], None, None, rule=inspect.currentframe().f_code.co_name)'''
actions['''trailer : LPAR arglist RPAR'''] = '''    p[0] = p[2]'''
actions['''trailer : LSQB subscriptlist RSQB'''] = '''    p[0] = ast.Subscript(None, p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name)'''
actions['''trailer : DOT NAME'''] = '''    p[0] = ast.Attribute(None, p[2][0], ast.Load(), rule=inspect.currentframe().f_code.co_name)'''
actions['''subscriptlist : subscript'''] = '''    p[0] = p[1]'''
actions['''subscriptlist : subscript COMMA'''] = '''    if isinstance(p[1], ast.Index):
        tup = ast.Tuple([p[1].value], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
        inherit_lineno(tup, p[1].value)
        p[0] = ast.Index(tup, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], tup)
    else:
        p[0] = ast.ExtSlice([p[1]], rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], p[1])'''
actions['''subscriptlist : subscript subscriptlist_star'''] = '''    args = [p[1]] + p[2]
    if all(isinstance(x, ast.Index) for x in args):
        tup = ast.Tuple([x.value for x in args], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
        inherit_lineno(tup, args[0].value)
        p[0] = ast.Index(tup, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], tup)
    else:
        p[0] = ast.ExtSlice(args, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], p[1])'''
actions['''subscriptlist : subscript subscriptlist_star COMMA'''] = '''    args = [p[1]] + p[2]
    if all(isinstance(x, ast.Index) for x in args):
        tup = ast.Tuple([x.value for x in args], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
        inherit_lineno(tup, args[0].value)
        p[0] = ast.Index(tup, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], tup)
    else:
        p[0] = ast.ExtSlice(args, rule=inspect.currentframe().f_code.co_name)
        inherit_lineno(p[0], p[1])'''
actions['''subscriptlist_star : COMMA subscript'''] = '''    p[0] = [p[2]]'''
actions['''subscriptlist_star : subscriptlist_star COMMA subscript'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''subscript : DOT DOT DOT'''] = '''    p[0] = ast.Ellipsis(rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''subscript : test'''] = '''    p[0] = ast.Index(p[1], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''subscript : COLON'''] = '''    p[0] = ast.Slice(None, None, None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''subscript : COLON sliceop'''] = '''    p[0] = ast.Slice(None, None, p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''subscript : COLON test'''] = '''    p[0] = ast.Slice(None, p[2], None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''subscript : COLON test sliceop'''] = '''    p[0] = ast.Slice(None, p[2], p[3], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''subscript : test COLON'''] = '''    p[0] = ast.Slice(p[1], None, None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''subscript : test COLON sliceop'''] = '''    p[0] = ast.Slice(p[1], None, p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''subscript : test COLON test'''] = '''    p[0] = ast.Slice(p[1], p[3], None, rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''subscript : test COLON test sliceop'''] = '''    p[0] = ast.Slice(p[1], p[3], p[4], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''sliceop : COLON'''] = '''    p[0] = ast.Name("None", ast.Load(), rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''sliceop : COLON test'''] = '''    p[0] = p[2]'''
actions['''exprlist : expr'''] = '''    p[0] = p[1]'''
actions['''exprlist : expr COMMA'''] = '''    p[0] = ast.Tuple([p[1]], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''exprlist : expr exprlist_star'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''exprlist : expr exprlist_star COMMA'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''exprlist_star : COMMA expr'''] = '''    p[0] = [p[2]]'''
actions['''exprlist_star : exprlist_star COMMA expr'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''testlist : test'''] = '''    p[0] = p[1]'''
actions['''testlist : test COMMA'''] = '''    p[0] = ast.Tuple([p[1]], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist : test testlist_star'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist : test testlist_star COMMA'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist_star : COMMA test'''] = '''    p[0] = [p[2]]'''
actions['''testlist_star : testlist_star COMMA test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''dictorsetmaker : test COLON test comp_for'''] = '''    p[0] = ast.DictComp(p[1], p[3], p[4], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''dictorsetmaker : test COLON test'''] = '''    p[0] = ([p[1]], [p[3]])'''
actions['''dictorsetmaker : test COLON test COMMA'''] = '''    p[0] = ([p[1]], [p[3]])'''
actions['''dictorsetmaker : test COLON test dictorsetmaker_star'''] = '''    keys, values = p[4]
    p[0] = ([p[1]] + keys, [p[3]] + values)'''
actions['''dictorsetmaker : test COLON test dictorsetmaker_star COMMA'''] = '''    keys, values = p[4]
    p[0] = ([p[1]] + keys, [p[3]] + values)'''
actions['''dictorsetmaker : test comp_for'''] = '''    p[0] = ast.SetComp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''dictorsetmaker : test'''] = '''    p[0] = (None, [p[1]])'''
actions['''dictorsetmaker : test COMMA'''] = '''    p[0] = (None, [p[1]])'''
actions['''dictorsetmaker : test dictorsetmaker_star2'''] = '''    keys, values = p[2]
    p[0] = (keys, [p[1]] + values)'''
actions['''dictorsetmaker : test dictorsetmaker_star2 COMMA'''] = '''    keys, values = p[2]
    p[0] = (keys, [p[1]] + values)'''
actions['''dictorsetmaker_star : COMMA test COLON test'''] = '''    p[0] = ([p[2]], [p[4]])'''
actions['''dictorsetmaker_star : dictorsetmaker_star COMMA test COLON test'''] = '''    keys, values = p[1]
    p[0] = (keys + [p[3]], values + [p[5]])'''
actions['''dictorsetmaker_star2 : COMMA test'''] = '''    p[0] = (None, [p[2]])'''
actions['''dictorsetmaker_star2 : dictorsetmaker_star2 COMMA test'''] = '''    keys, values = p[1]
    p[0] = (keys, values + [p[3]])'''
actions['''classdef : CLASS NAME COLON suite'''] = '''    p[0] = ast.ClassDef(p[2][0], [], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''classdef : CLASS NAME LPAR RPAR COLON suite'''] = '''    p[0] = ast.ClassDef(p[2][0], [], p[6], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''classdef : CLASS NAME LPAR testlist RPAR COLON suite'''] = '''    if isinstance(p[4], ast.Tuple):
        p[0] = ast.ClassDef(p[2][0], p[4].elts, p[7], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])
    else:
        p[0] = ast.ClassDef(p[2][0], [p[4]], p[7], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''arglist : argument'''] = '''    if notkeyword(p[1]):
        p[0] = ast.Call(None, [p[1]], [], None, None, rule=inspect.currentframe().f_code.co_name)
    else:
        p[0] = ast.Call(None, [], [p[1]], None, None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : argument COMMA'''] = '''    if notkeyword(p[1]):
        p[0] = ast.Call(None, [p[1]], [], None, None, rule=inspect.currentframe().f_code.co_name)
    else:
        p[0] = ast.Call(None, [], [p[1]], None, None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : STAR test'''] = '''    p[0] = ast.Call(None, [], [], p[2], None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : STAR test COMMA DOUBLESTAR test'''] = '''    p[0] = ast.Call(None, [], [], p[2], p[5], rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : STAR test arglist_star'''] = '''    p[0] = ast.Call(None, filter(notkeyword, p[3]), filter(iskeyword, p[3]), p[2], None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : STAR test arglist_star COMMA DOUBLESTAR test'''] = '''    p[0] = ast.Call(None, filter(notkeyword, p[3]), filter(iskeyword, p[3]), p[2], p[6], rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : DOUBLESTAR test'''] = '''    p[0] = ast.Call(None, [], [], None, p[2], rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 argument'''] = '''    args = p[1] + [p[2]]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args), None, None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 argument COMMA'''] = '''    args = p[1] + [p[2]]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args), None, None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 STAR test'''] = '''    p[0] = ast.Call(None, filter(notkeyword, p[1]), filter(iskeyword, p[1]), p[3], None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 STAR test COMMA DOUBLESTAR test'''] = '''    p[0] = ast.Call(None, filter(notkeyword, p[1]), filter(iskeyword, p[1]), p[3], p[6], rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 STAR test arglist_star3'''] = '''    args = p[1] + p[4]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args), p[3], None, rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 STAR test arglist_star3 COMMA DOUBLESTAR test'''] = '''    args = p[1] + p[4]
    p[0] = ast.Call(None, filter(notkeyword, args), filter(iskeyword, args), p[3], p[7], rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist : arglist_star2 DOUBLESTAR test'''] = '''    p[0] = ast.Call(None, filter(notkeyword, p[1]), filter(iskeyword, p[1]), None, p[3], rule=inspect.currentframe().f_code.co_name)'''
actions['''arglist_star : COMMA argument'''] = '''    p[0] = [p[2]]'''
actions['''arglist_star : arglist_star COMMA argument'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''arglist_star3 : COMMA argument'''] = '''    p[0] = [p[2]]'''
actions['''arglist_star3 : arglist_star3 COMMA argument'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''arglist_star2 : argument COMMA'''] = '''    p[0] = [p[1]]'''
actions['''arglist_star2 : arglist_star2 argument COMMA'''] = '''    p[0] = p[1] + [p[2]]'''
actions['''argument : test'''] = '''    p[0] = p[1]'''
actions['''argument : test comp_for'''] = '''    p[0] = ast.GeneratorExp(p[1], p[2], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''argument : test EQUAL test'''] = '''    p[0] = ast.keyword(p[1].id, p[3], rule=inspect.currentframe().f_code.co_name)
    inherit_lineno(p[0], p[1])'''
actions['''list_iter : list_for'''] = '''    p[0] = ([], p[1])'''
actions['''list_iter : list_if'''] = '''    p[0] = p[1]'''
actions['''list_for : FOR exprlist IN testlist_safe'''] = '''    ctx_to_store(p[2])
    p[0] = [ast.comprehension(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])]'''
actions['''list_for : FOR exprlist IN testlist_safe list_iter'''] = '''    ctx_to_store(p[2])
    ifs, iters = p[5]
    p[0] = [ast.comprehension(p[2], p[4], ifs, rule=inspect.currentframe().f_code.co_name, **p[1][1])] + iters'''
actions['''list_if : IF old_test'''] = '''    p[0] = ([p[2]], [])'''
actions['''list_if : IF old_test list_iter'''] = '''    ifs, iters = p[3]
    p[0] = ([p[2]] + ifs, iters)'''
actions['''comp_iter : comp_for'''] = '''    p[0] = ([], p[1])'''
actions['''comp_iter : comp_if'''] = '''    p[0] = p[1]'''
actions['''comp_for : FOR exprlist IN or_test'''] = '''    ctx_to_store(p[2])
    p[0] = [ast.comprehension(p[2], p[4], [], rule=inspect.currentframe().f_code.co_name, **p[1][1])]'''
actions['''comp_for : FOR exprlist IN or_test comp_iter'''] = '''    ctx_to_store(p[2])
    ifs, iters = p[5]
    p[0] = [ast.comprehension(p[2], p[4], ifs, rule=inspect.currentframe().f_code.co_name, **p[1][1])] + iters'''
actions['''comp_if : IF old_test'''] = '''    p[0] = ([p[2]], [])'''
actions['''comp_if : IF old_test comp_iter'''] = '''    ifs, iters = p[3]
    p[0] = ([p[2]] + ifs, iters)'''
actions['''testlist1 : test'''] = '''    p[0] = p[1]'''
actions['''testlist1 : test testlist1_star'''] = '''    p[0] = ast.Tuple([p[1]] + p[2], ast.Load(), rule=inspect.currentframe().f_code.co_name, paren=False)
    inherit_lineno(p[0], p[1])'''
actions['''testlist1_star : COMMA test'''] = '''    p[0] = [p[2]]'''
actions['''testlist1_star : testlist1_star COMMA test'''] = '''    p[0] = p[1] + [p[3]]'''
actions['''encoding_decl : NAME'''] = '''    p[0] = p[1]'''
actions['''yield_expr : YIELD'''] = '''    p[0] = ast.Yield(None, rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
actions['''yield_expr : YIELD testlist'''] = '''    p[0] = ast.Yield(p[2], rule=inspect.currentframe().f_code.co_name, **p[1][1])'''
