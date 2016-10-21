#!/usr/bin/env python

import ast
import sys
import re
import glob

import histogrammar.hgawk_grammar as grammar

def check(source, fileName=None):
    theirs = ast.parse(source)
    theirsdump = ast.dump(theirs)
    theirsres = {}
    if fileName is None:
        try:
            exec compile(theirs, "<check>", "exec") in theirsres
        except Exception as err:
            theirsres = err
        else:
            for key in theirsres.keys():
                if callable(theirsres[key]):
                    del theirsres[key]

    if fileName is None:
        print >> sys.stderr, "test:  ", source
    else:
        print >> sys.stderr, "test:  ", fileName

    mine = grammar.parse(source)
    mineres = {}
    try:
        minedump = ast.dump(mine)
    except Exception as err:
        minedump = err
    else:
        if fileName is None:
            try:
                exec compile(mine, "<check>", "exec") in mineres
            except Exception as err:
                mineres = err
            else:
                for key in mineres.keys():
                    if callable(mineres[key]):
                        del mineres[key]

    # verify that the dump is the same
    if minedump == theirsdump and \
       ((isinstance(mineres, Exception) and isinstance(theirsres, Exception) and str(mineres) == str(theirsres)) or \
        (isinstance(mineres, dict) and isinstance(theirsres, dict) and mineres == theirsres)):
        pass
    else:
        print >> sys.stderr, "my dump:     ", minedump
        print >> sys.stderr, "their dump:  ", theirsdump
        if isinstance(mineres, Exception):
            print >> sys.stderr, "my except:   ", str(mineres)
        if isinstance(theirsres, Exception):
            print >> sys.stderr, "their except:", str(theirsres)
        if not isinstance(mineres, Exception) and not isinstance(theirsres, Exception):
            t = {}
            m = {}
            for key in theirsres:
                if key not in mineres:
                    t[key] = theirsres[key]
                elif mineres[key] != theirsres[key]:
                    m[key] = mineres[key]
                    t[key] = theirsres[key]
            for key in mineres:
                if key not in theirsres:
                    m[key] = mineres[key]
            print >> sys.stderr, "my result:   ", m
            print >> sys.stderr, "their result:", m
        print >> sys.stderr
        sys.exit(-1)

    # verify that even the line numbers are the same
    global same, treeOne, treeTwo
    same = True
    treeOne = ""
    treeTwo = ""
    def deepcompare(one, two, indent):
        global same, treeOne, treeTwo
        if isinstance(one, ast.AST):
            if not (isinstance(two, ast.AST) and one._fields == two._fields and one.__class__ == two.__class__):
                same = False
            if not (getattr(one, "lineno", "?") == getattr(two, "lineno", "?") and getattr(one, "col_offset", "?") == getattr(two, "col_offset", "?")):
                if hasattr(one, "lineno") and hasattr(one, "col_offset"):
                    # Python's lineno/col_offset for strings with line breaks is wrong.
                    # Don't count it against my implementation for getting it right.
                    if not isinstance(one, ast.Str) and not (isinstance(one, ast.Expr) and isinstance(one.value, ast.Str)):
                        same = False
            if not (hasattr(two, "lineno") and hasattr(two, "col_offset")):
                raise Exception
            treeOne += one.__class__.__name__ + " " + str(getattr(one, "lineno", "?")) + ":" + str(getattr(one, "col_offset", "?")) + "\n"
            treeTwo += two.__class__.__name__ + " " + str(getattr(two, "lineno", "?")) + ":" + str(getattr(two, "col_offset", "?")) + " (" + str(getattr(two, "rule", "???")) + ")\n"
            for attrib in one._fields:
                treeOne += indent + "  " + attrib + ": "
                treeTwo += indent + "  " + attrib + ": "
                valueOne = getattr(one, attrib)
                valueTwo = getattr(two, attrib)
                if isinstance(valueOne, list):
                    if not (isinstance(valueTwo, list) and len(valueOne) == len(valueTwo)):
                        same = False
                    if len(valueOne) == 0:
                        treeOne += "[]\n"
                    else:
                        treeOne += "\n"
                    if len(valueTwo) == 0:
                        treeTwo += "[]\n"
                    else:
                        treeTwo += "\n"
                    for x, y in zip(valueOne, valueTwo):
                        treeOne += indent + "    - "
                        treeTwo += indent + "    - "
                        deepcompare(x, y, indent + "        ")
                elif isinstance(valueOne, (ast.Load, ast.Store, ast.Param, ast.Del)):
                    if not (isinstance(valueTwo, (ast.Load, ast.Store, ast.Param, ast.Del))):
                        same = False
                    treeOne += valueOne.__class__.__name__ + "\n"
                    treeTwo += valueTwo.__class__.__name__ + "\n"
                elif isinstance(valueOne, ast.AST):
                    if not (isinstance(valueTwo, ast.AST)):
                        same = False
                    deepcompare(valueOne, valueTwo, indent + "    ")
                elif valueOne is None or isinstance(valueOne, (int, long, float, complex, basestring)):
                    if not (valueOne == valueTwo):
                        same = False
                    treeOne += repr(valueOne) + "\n"
                    treeTwo += repr(valueTwo) + "\n"
                else:
                    raise Exception
        else:
            if not (one == two):
                same = False

    if fileName is not None:
        return

    deepcompare(theirs, mine, "")
    if not same:
        print >> sys.stderr, source
        print >> sys.stderr
        treeOne = treeOne.split("\n")
        treeTwo = treeTwo.split("\n")
        width = max(len(x) for x in treeOne) + 3
        while len(treeOne) < len(treeTwo):
            treeOne.append("")
        while len(treeTwo) < len(treeOne):
            treeTwo.append("")
        for x, y in zip(treeOne, treeTwo):
            diff = x != re.sub("\s*\(.*\)", "", y)
            while len(x) < width:
                x += " "
            if diff:
                print >> sys.stderr, x + "| " + y
            else:
                print >> sys.stderr, x + "  " + y
        sys.exit(-1)

if __name__ == "__main__":
    sys.stdout = open("/dev/null", "wb")

    check('''3 <> 4''')
    check('''.3''')

    check('''-3''')
    check('''- 3''')
    check('''-  3''')
    check('''--3''')
    check('''-- 3''')
    check('''- -3''')
    check('''- - 3''')
    check('''- -  3''')
    check('''+3''')
    check('''+ 3''')
    check('''+  3''')
    check('''++3''')
    check('''++ 3''')
    check('''+ +3''')
    check('''+ + 3''')
    check('''+ +  3''')
    check('''+-3''')
    check('''+- 3''')
    check('''+ -3''')
    check('''+ - 3''')
    check('''+ -  3''')
    check('''-+3''')
    check('''-+ 3''')
    check('''- +3''')
    check('''- + 3''')
    check('''- +  3''')
    check('''-3.14''')
    check('''- 3.14''')
    check('''-  3.14''')
    check('''--3.14''')
    check('''-- 3.14''')
    check('''- -3.14''')
    check('''- - 3.14''')
    check('''- -  3.14''')
    check('''+3.14''')
    check('''+ 3.14''')
    check('''+  3.14''')
    check('''++3.14''')
    check('''++ 3.14''')
    check('''+ +3.14''')
    check('''+ + 3.14''')
    check('''+ +  3.14''')
    check('''+-3.14''')
    check('''+- 3.14''')
    check('''+ -3.14''')
    check('''+ - 3.14''')
    check('''+ -  3.14''')
    check('''-+3.14''')
    check('''-+ 3.14''')
    check('''- +3.14''')
    check('''- + 3.14''')
    check('''- +  3.14''')
    check('''-3e1''')
    check('''- 3e1''')
    check('''-  3e1''')
    check('''--3e1''')
    check('''-- 3e1''')
    check('''- -3e1''')
    check('''- - 3e1''')
    check('''- -  3e1''')
    check('''+3e1''')
    check('''+ 3e1''')
    check('''+  3e1''')
    check('''++3e1''')
    check('''++ 3e1''')
    check('''+ +3e1''')
    check('''+ + 3e1''')
    check('''+ +  3e1''')
    check('''+-3e1''')
    check('''+- 3e1''')
    check('''+ -3e1''')
    check('''+ - 3e1''')
    check('''+ -  3e1''')
    check('''-+3e1''')
    check('''-+ 3e1''')
    check('''- +3e1''')
    check('''- + 3e1''')
    check('''- +  3e1''')

    check('''3''')
    check('''3,''')
    check('''3, 4''')
    check('''3, 4,''')
    check('''3, 4, 5''')
    check('''3, 4, 5,''')
    check('''3, 4, 5, 6''')
    check('''3, 4, 5, 6,''')

    check('''()''')
    check('''(3)''')
    check('''(3,)''')
    check('''(3, 4)''')
    check('''(3, 4,)''')
    check('''(3, 4, 5)''')
    check('''(3, 4, 5,)''')
    check('''(3, 4, 5, 6)''')
    check('''(3, 4, 5, 6,)''')

    check('''(),''')
    check('''(3),''')
    check('''(3,),''')
    check('''(3, 4),''')
    check('''(3, 4,),''')
    check('''(3, 4, 5),''')
    check('''(3, 4, 5,),''')
    check('''(3, 4, 5, 6),''')
    check('''(3, 4, 5, 6,),''')

    check('''((1), 2, 3, 4, 5)''')
    check('''((1, 2), 3, 4, 5)''')
    check('''((1, 2, 3), 4, 5)''')
    check('''((1, 2, 3, 4), 5)''')
    check('''((1, 2, 3, 4, 5))''')
    check('''(((1), 2, 3, 4, 5))''')
    check('''(((1, 2), 3, 4, 5))''')
    check('''(((1, 2, 3), 4, 5))''')
    check('''(((1, 2, 3, 4), 5))''')
    check('''(((1, 2, 3, 4, 5)))''')

    check('''(1, 2, 3, 4, (5))''')
    check('''(1, 2, 3, (4, 5))''')
    check('''(1, 2, (3, 4, 5))''')
    check('''(1, (2, 3, 4, 5))''')
    check('''((1, 2, 3, 4, (5)))''')
    check('''((1, 2, 3, (4, 5)))''')
    check('''((1, 2, (3, 4, 5)))''')
    check('''((1, (2, 3, 4, 5)))''')

    check('''[]''')
    check('''[3]''')
    check('''[3,]''')
    check('''[3, 4]''')
    check('''[3, 4,]''')
    check('''[3, 4, 5]''')
    check('''[3, 4, 5,]''')
    check('''[3, 4, 5, 6]''')
    check('''[3, 4, 5, 6,]''')

    check('''[],''')
    check('''[3],''')
    check('''[3,],''')
    check('''[3, 4],''')
    check('''[3, 4,],''')
    check('''[3, 4, 5],''')
    check('''[3, 4, 5,],''')
    check('''[3, 4, 5, 6],''')
    check('''[3, 4, 5, 6,],''')

    check('''[[1], 2, 3, 4, 5]''')
    check('''[[1, 2], 3, 4, 5]''')
    check('''[[1, 2, 3], 4, 5]''')
    check('''[[1, 2, 3, 4], 5]''')
    check('''[[1, 2, 3, 4, 5]]''')
    check('''[[[1], 2, 3, 4, 5]]''')
    check('''[[[1, 2], 3, 4, 5]]''')
    check('''[[[1, 2, 3], 4, 5]]''')
    check('''[[[1, 2, 3, 4], 5]]''')
    check('''[[[1, 2, 3, 4, 5]]]''')

    check('''[1, 2, 3, 4, [5]]''')
    check('''[1, 2, 3, [4, 5]]''')
    check('''[1, 2, [3, 4, 5]]''')
    check('''[1, [2, 3, 4, 5]]''')
    check('''[[1, 2, 3, 4, [5]]]''')
    check('''[[1, 2, 3, [4, 5]]]''')
    check('''[[1, 2, [3, 4, 5]]]''')
    check('''[[1, [2, 3, 4, 5]]]''')

    check('''[(1), 2, 3, 4, 5]''')
    check('''[(1, 2), 3, 4, 5]''')
    check('''[(1, 2, 3), 4, 5]''')
    check('''[(1, 2, 3, 4), 5]''')
    check('''[(1, 2, 3, 4, 5)]''')
    check('''[([1], 2, 3, 4, 5)]''')
    check('''[([1, 2], 3, 4, 5)]''')
    check('''[([1, 2, 3], 4, 5)]''')
    check('''[([1, 2, 3, 4], 5)]''')
    check('''[([1, 2, 3, 4, 5])]''')

    check('''[1, 2, 3, 4, (5)]''')
    check('''[1, 2, 3, (4, 5)]''')
    check('''[1, 2, (3, 4, 5)]''')
    check('''[1, (2, 3, 4, 5)]''')
    check('''[(1, 2, 3, 4, [5])]''')
    check('''[(1, 2, 3, [4, 5])]''')
    check('''[(1, 2, [3, 4, 5])]''')
    check('''[(1, [2, 3, 4, 5])]''')

    check('''([1], 2, 3, 4, 5)''')
    check('''([1, 2], 3, 4, 5)''')
    check('''([1, 2, 3], 4, 5)''')
    check('''([1, 2, 3, 4], 5)''')
    check('''([1, 2, 3, 4, 5])''')
    check('''([(1), 2, 3, 4, 5])''')
    check('''([(1, 2), 3, 4, 5])''')
    check('''([(1, 2, 3), 4, 5])''')
    check('''([(1, 2, 3, 4), 5])''')
    check('''([(1, 2, 3, 4, 5)])''')

    check('''(1, 2, 3, 4, [5])''')
    check('''(1, 2, 3, [4, 5])''')
    check('''(1, 2, [3, 4, 5])''')
    check('''(1, [2, 3, 4, 5])''')
    check('''([1, 2, 3, 4, (5)])''')
    check('''([1, 2, 3, (4, 5)])''')
    check('''([1, 2, (3, 4, 5)])''')
    check('''([1, (2, 3, 4, 5)])''')

    check('''{}''')
    check('''{3}''')
    check('''{3,}''')
    check('''{3, 4}''')
    check('''{3, 4,}''')
    check('''{3, 4, 5}''')
    check('''{3, 4, 5,}''')
    check('''{3, 4, 5, 6}''')
    check('''{3, 4, 5, 6,}''')

    check('''{},''')
    check('''{3},''')
    check('''{3,},''')
    check('''{3, 4},''')
    check('''{3, 4,},''')
    check('''{3, 4, 5},''')
    check('''{3, 4, 5,},''')
    check('''{3, 4, 5, 6},''')
    check('''{3, 4, 5, 6,},''')

    check('''{{1}, 2, 3, 4, 5}''')
    check('''{{1, 2}, 3, 4, 5}''')
    check('''{{1, 2, 3}, 4, 5}''')
    check('''{{1, 2, 3, 4}, 5}''')
    check('''{{1, 2, 3, 4, 5}}''')
    check('''{{{1}, 2, 3, 4, 5}}''')
    check('''{{{1, 2}, 3, 4, 5}}''')
    check('''{{{1, 2, 3}, 4, 5}}''')
    check('''{{{1, 2, 3, 4}, 5}}''')
    check('''{{{1, 2, 3, 4, 5}}}''')

    check('''{1, 2, 3, 4, {5}}''')
    check('''{1, 2, 3, {4, 5}}''')
    check('''{1, 2, {3, 4, 5}}''')
    check('''{1, {2, 3, 4, 5}}''')
    check('''{{1, 2, 3, 4, {5}}}''')
    check('''{{1, 2, 3, {4, 5}}}''')
    check('''{{1, 2, {3, 4, 5}}}''')
    check('''{{1, {2, 3, 4, 5}}}''')

    check('''{(1), 2, 3, 4, 5}''')
    check('''{(1, 2), 3, 4, 5}''')
    check('''{(1, 2, 3), 4, 5}''')
    check('''{(1, 2, 3, 4), 5}''')
    check('''{(1, 2, 3, 4, 5)}''')
    check('''{({1}, 2, 3, 4, 5)}''')
    check('''{({1, 2}, 3, 4, 5)}''')
    check('''{({1, 2, 3}, 4, 5)}''')
    check('''{({1, 2, 3, 4}, 5)}''')
    check('''{({1, 2, 3, 4, 5})}''')

    check('''{1, 2, 3, 4, (5)}''')
    check('''{1, 2, 3, (4, 5)}''')
    check('''{1, 2, (3, 4, 5)}''')
    check('''{1, (2, 3, 4, 5)}''')
    check('''{(1, 2, 3, 4, {5})}''')
    check('''{(1, 2, 3, {4, 5})}''')
    check('''{(1, 2, {3, 4, 5})}''')
    check('''{(1, {2, 3, 4, 5})}''')

    check('''({1}, 2, 3, 4, 5)''')
    check('''({1, 2}, 3, 4, 5)''')
    check('''({1, 2, 3}, 4, 5)''')
    check('''({1, 2, 3, 4}, 5)''')
    check('''({1, 2, 3, 4, 5})''')
    check('''({(1), 2, 3, 4, 5})''')
    check('''({(1, 2), 3, 4, 5})''')
    check('''({(1, 2, 3), 4, 5})''')
    check('''({(1, 2, 3, 4), 5})''')
    check('''({(1, 2, 3, 4, 5)})''')

    check('''(1, 2, 3, 4, {5})''')
    check('''(1, 2, 3, {4, 5})''')
    check('''(1, 2, {3, 4, 5})''')
    check('''(1, {2, 3, 4, 5})''')
    check('''({1, 2, 3, 4, (5)})''')
    check('''({1, 2, 3, (4, 5)})''')
    check('''({1, 2, (3, 4, 5)})''')
    check('''({1, (2, 3, 4, 5)})''')

    check('''{a: 3}''')
    check('''{a: 3,}''')
    check('''{a: 3, b: 4}''')
    check('''{a: 3, b: 4,}''')
    check('''{a: 3, b: 4, c: 5}''')
    check('''{a: 3, b: 4, c: 5,}''')
    check('''{a: 3, b: 4, c: 5, d: 6}''')
    check('''{a: 3, b: 4, c: 5, d: 6,}''')

    check('''{a: 3},''')
    check('''{a: 3,},''')
    check('''{a: 3, b: 4},''')
    check('''{a: 3, b: 4,},''')
    check('''{a: 3, b: 4, c: 5},''')
    check('''{a: 3, b: 4, c: 5,},''')
    check('''{a: 3, b: 4, c: 5, d: 6},''')
    check('''{a: 3, b: 4, c: 5, d: 6,},''')

    check('''{{x: 1}, 2, 3, 4, 5}''')
    check('''{{x: 1, y: 2}, 3, 4, 5}''')
    check('''{{x: 1, y: 2, a: 3}, 4, 5}''')
    check('''{{x: 1, y: 2, a: 3, b: 4}, 5}''')
    check('''{{x: 1, y: 2, a: 3, b: 4, c: 5}}''')
    check('''{{{x: 1}, 2, 3, 4, 5}}''')
    check('''{{{x: 1, y: 2}, 3, 4, 5}}''')
    check('''{{{x: 1, y: 2, a: 3}, 4, 5}}''')
    check('''{{{x: 1, y: 2, a: 3, b: 4}, 5}}''')
    check('''{{{x: 1, y: 2, a: 3, b: 4, c: 5}}}''')

    check('''{1, 2, 3, 4, {c: 5}}''')
    check('''{1, 2, 3, {b: 4, c: 5}}''')
    check('''{1, 2, {a: 3, b: 4, c: 5}}''')
    check('''{1, {y: 2, a: 3, b: 4, c: 5}}''')
    check('''{{1, 2, 3, 4, {c: 5}}}''')
    check('''{{1, 2, 3, {b: 4, c: 5}}}''')
    check('''{{1, 2, {a: 3, b: 4, c: 5}}}''')
    check('''{{1, {y: 2, a: 3, b: 4, c: 5}}}''')

    check('''{x: (1), y: 2, a: 3, b: 4, c: 5}''')
    check('''{x: (1, 2), a: 3, b: 4, c: 5}''')
    check('''{x: (1, 2, 3), b: 4, c: 5}''')
    check('''{x: (1, 2, 3, 4), c: 5}''')
    check('''{x: (1, 2, 3, 4, 5)}''')
    check('''{({x: 1}, 2, 3, 4, 5)}''')
    check('''{({x: 1, y: 2}, 3, 4, 5)}''')
    check('''{({x: 1, y: 2, a: 3}, 4, 5)}''')
    check('''{({x: 1, y: 2, a: 3, b: 4}, 5)}''')
    check('''{({x: 1, y: 2, a: 3, b: 4, c: 5})}''')

    check('''{x: 1, y: 2, a: 3, b: 4, c: (5)}''')
    check('''{x: 1, y: 2, a: 3, b: (4, 5)}''')
    check('''{x: 1, y: 2, a: (3, 4, 5)}''')
    check('''{x: 1, y: (2, 3, 4, 5)}''')
    check('''{(1, 2, 3, 4, {c: 5})}''')
    check('''{(1, 2, 3, {b: 4, c: 5})}''')
    check('''{(1, 2, {a: 3, b: 4, c: 5})}''')
    check('''{(1, {y: 2, a: 3, b: 4, c: 5})}''')

    check('''({x: 1}, 2, 3, 4, 5)''')
    check('''({x: 1, y: 2}, 3, 4, 5)''')
    check('''({x: 1, y: 2, a: 3}, 4, 5)''')
    check('''({x: 1, y: 2, a: 3, b: 4}, 5)''')
    check('''({x: 1, y: 2, a: 3, b: 4, c: 5})''')
    check('''({x: (1), y: 2, a: 3, b: 4, c: 5})''')
    check('''({x: (1, 2), a: 3, b: 4, c: 5})''')
    check('''({x: (1, 2, 3), b: 4, c: 5})''')
    check('''({x: (1, 2, 3, 4), c: 5})''')
    check('''({x: (1, 2, 3, 4, 5)})''')

    check('''(1, 2, 3, 4, {c: 5})''')
    check('''(1, 2, 3, {b: 4, c: 5})''')
    check('''(1, 2, {a: 3, b: 4, c: 5})''')
    check('''(1, {y: 2, a: 3, b: 4, c: 5})''')
    check('''({x: 1, y: 2, a: 3, b: (4, 5)})''')
    check('''({x: 1, y: 2, a: (3, 4, 5)})''')
    check('''({x: 1, y: (2, 3, 4, 5)})''')

    check('''a = 3''')
    check('''a, = 3,''')
    check('''a, b = 3, 4''')
    check('''a, b, = 3, 4,''')
    check('''a, b, c = 3, 4, 5''')
    check('''a, b, c, = 3, 4, 5,''')
    check('''a, b, c, d = 3, 4, 5, 6''')
    check('''a, b, c, d, = 3, 4, 5, 6,''')

    check('''z = a = 3''')
    check('''z = a, = 3,''')
    check('''z = a, b = 3, 4''')
    check('''z = a, b, = 3, 4,''')
    check('''z = a, b, c = 3, 4, 5''')
    check('''z = a, b, c, = 3, 4, 5,''')
    check('''z = a, b, c, d = 3, 4, 5, 6''')
    check('''z = a, b, c, d, = 3, 4, 5, 6,''')

    check('''a = (b, c) = (d, [e, f]) = [1, (2, 3)]''')
    check('''a = (b, c,) = (d, [e, f]) = [1, (2, 3)]''')
    check('''a = b, c, = (d, [e, f]) = [1, (2, 3)]''')
    check('''a, = b, c, = (d, [e, f]) = [1, (2, 3)]''')
    check('''a = b, c, = d, [e, f] = [1, (2, 3)]''')
    check('''a = b, c, z = d, [e, f] = [1, (2, 3)]''')

    check('''a = (b, c) = (d, [e, f]) = [1, (2, 3,)]''')
    check('''a = (b, c,) = (d, [e, f]) = [1, (2, 3),]''')
    check('''a = b, c, = (d, [e, f]) = [1, (2, 3)],''')
    check('''a, = b, c, = (d, [e, f]) = [(2, 3), 1]''')
    check('''a = b, c, = d, [e, f] = [(2, 3), 1,]''')
    check('''a = b, c, z = d, [e, f] = [(2, 3), 1, ()]''')

    check('''''')
    check('''
''')
    check('''

''')
    check('''3
''')
    check('''3

''')
    check('''3


''')
    check('''3



''')
    check('''
3''')
    check('''

3''')
    check('''


3''')
    check('''



3''')
    check('''x = 3
''')
    check('''x = 3

''')
    check('''x = 3


''')
    check('''x = 3



''')
    check('''
x = 3''')
    check('''

x = 3''')
    check('''


x = 3''')
    check('''



x = 3''')

    check('''x; y''')
    check('''x; y; z''')
    check('''x; y; z; w''')
    check('''x
y''')
    check('''x
y
z''')
    check('''x
y
z
w''')
    check('''x

y
z
w''')
    check('''x; y
z
w''')
    check('''x
y
z
w;''')
    check('''x;
y;
z;
w;''')
    check('''x; y;
z;
w;''')
    check('''x; y;

z;
w''')

    check('''if x: y''')
    check('''if x:
    y''')
    check('''if x:

    y''')
    check('''if x:
    y
''')
    check('''if x:
    y
    ''')
    check('''
if x:
    y''')
    check('''if x:
    y;''')
    check('''if x:
    y;
''')
    check('''if x:
    y
    z
''')
    check('''if x:
    y
    z
w
''')
    check('''if x:
    y;
    z
w
''')
    check('''if x:
    y
    z;
w
''')
    check('''if x:
    y
    z
w;
''')
    check('''if x:
    if y:
        z
w
''')
    check('''if x:
    if y:
        z
    w
u
''')
    check('''if x:
    xx
    if y:
        z
    w
u
''')
    check('''if x:
    xx;
    if y:
        z
    w
u
''')
    check('''if x:
    xx
    if y:
        z;
    w
u
''')
    check('''if x:
    xx
    if y:
        z
    w;
u
''')

    check('''if x:
    y
else:
    z''')
    check('''if x: y
else: z''')
    check('''if x:
    if y:
        w
else:
    z''')
    check('''if x:
    if y:
        w
    ww
else:
    z''')
    check('''if x:
    uu
    if y:
        w
    ww
else:
    z''')
    check('''if x:
    y
else:
    if z:
        w''')
    check('''if x:
    y
else:
  if z:
   w''')
    check('''if x:
    y
else:
  if z:
   w
  else:
      u''')

    check('''if x:
    y
elif z:
    w''')
    check('''if x:
    y
elif z:
    w
else:
    u''')
    check('''if x:
    y
elif z:
    w
elif u:
    a''')
    check('''if x:
    y
elif z:
    w
elif u:
    a
else:
    b''')
    check('''if x:
    y
elif z:
    w
elif u:
    a
elif b:
    c''')
    check('''if x:
    y
elif z:
    w
elif u:
    a
elif b:
    c
else:
    d''')

    check('''while x: y''')
    check('''while x:
    y''')
    check('''while x:
    y
else:
    z''')

    check('''for x in 1, 2, 3:
    z''')
    check('''for x, in 1, 2, 3:
    z''')
    check('''for x, y in 1, 2, 3:
    z''')
    check('''for x, y, in 1, 2, 3:
    z''')
    check('''for x, y, z in 1, 2, 3:
    z''')
    check('''for x, y, z, in 1, 2, 3:
    z''')
    check('''for x in 1, 2, 3:
    z
else:
    w''')

    check('''try:
    x
except:
    y''')
    check('''try:
    x
except A:
    y''')
    check('''try:
    x
except A as a:
    y''')
    check('''try:
    x
except A, a:
    y''')
    check('''try:
    x
except A:
    y
except B:
    z''')
    check('''try:
    x
except A as a:
    y
except B as b:
    z''')
    check('''try:
    x
except A, a:
    y
except B, b:
    z''')
    check('''try:
    x
except:
    y
else:
    z''')
    check('''try:
    x
except A:
    y
else:
    z''')
    check('''try:
    x
except A as a:
    y
else:
    z''')
    check('''try:
    x
except A, a:
    y
else:
    z''')
    check('''try:
    x
finally:
    z''')
    check('''try:
    x
except:
    y
finally:
    z''')
    check('''try:
    x
except A:
    y
finally:
    z''')
    check('''try:
    x
except A as a:
    y
finally:
    z''')
    check('''try:
    x
except A, a:
    y
finally:
    z''')
    check('''try:
    x
except:
    y
else:
    yy
finally:
    z''')
    check('''try:
    x
except A:
    y
else:
    yy
finally:
    z''')
    check('''try:
    x
except A as a:
    y
else:
    yy
finally:
    z''')
    check('''try:
    x
except A, a:
    y
else:
    yy
finally:
    z''')

    check('''with a: b''')
    check('''with a, b: c''')
    check('''with a, b, c: d''')
    check('''with a, b, c, d: e''')
    check('''with a as aa: b''')
    check('''with a as aa, b as bb: c''')
    check('''with a as aa, b as bb, c as cc: d''')
    check('''with a as aa, b as bb, c as cc, d as dd: e''')

    check('''def f(): 3''')
    check('''def f(x): 3''')
    check('''def f(x,): 3''')
    check('''def f(x, y): 3''')
    check('''def f(x, y,): 3''')
    check('''def f(x, y, z): 3''')
    check('''def f(x, y, z, w): 3''')
    check('''def f(x = 1): 3''')
    check('''def f(x = 1,): 3''')
    check('''def f(x, y = 2): 3''')
    check('''def f(x, y = 2,): 3''')
    check('''def f(x = 1, y = 2): 3''')
    check('''def f(x = 1, y = 2,): 3''')
    check('''def f(x = 1, y = 2, z = 3): 3''')
    check('''def f(x, y = 2, z = 3): 3''')
    check('''def f(x, y, z = 3): 3''')

    check('''def f(*args): 3''')
    check('''def f(x, *args): 3''')
    check('''def f(x, y, *args): 3''')
    check('''def f(x, y, z, *args): 3''')
    check('''def f(x, y, z, w, *args): 3''')
    check('''def f(x = 1, *args): 3''')
    check('''def f(x, y = 2, *args): 3''')
    check('''def f(x = 1, y = 2, *args): 3''')
    check('''def f(x = 1, y = 2, z = 3, *args): 3''')
    check('''def f(x, y = 2, z = 3, *args): 3''')
    check('''def f(x, y, z = 3, *args): 3''')

    check('''def f(**kwds): 3''')
    check('''def f(x, **kwds): 3''')
    check('''def f(x, y, **kwds): 3''')
    check('''def f(x, y, z, **kwds): 3''')
    check('''def f(x, y, z, w, **kwds): 3''')
    check('''def f(x = 1, **kwds): 3''')
    check('''def f(x, y = 2, **kwds): 3''')
    check('''def f(x = 1, y = 2, **kwds): 3''')
    check('''def f(x = 1, y = 2, z = 3, **kwds): 3''')
    check('''def f(x, y = 2, z = 3, **kwds): 3''')
    check('''def f(x, y, z = 3, **kwds): 3''')

    check('''def f(*args, **kwds): 3''')
    check('''def f(x, *args, **kwds): 3''')
    check('''def f(x, y, *args, **kwds): 3''')
    check('''def f(x, y, z, *args, **kwds): 3''')
    check('''def f(x, y, z, w, *args, **kwds): 3''')
    check('''def f(x = 1, *args, **kwds): 3''')
    check('''def f(x, y = 2, *args, **kwds): 3''')
    check('''def f(x = 1, y = 2, *args, **kwds): 3''')
    check('''def f(x = 1, y = 2, z = 3, *args, **kwds): 3''')
    check('''def f(x, y = 2, z = 3, *args, **kwds): 3''')
    check('''def f(x, y, z = 3, *args, **kwds): 3''')

    check('''def f((a)): 3''')
    check('''def f((a,)): 3''')
    check('''def f((a, b)): 3''')
    check('''def f((a, b,)): 3''')
    check('''def f((a, b, c)): 3''')
    check('''def f((a, b, c, d)): 3''')
    check('''def f((a),): 3''')
    check('''def f((a,),): 3''')
    check('''def f((a, b),): 3''')
    check('''def f((a, b,),): 3''')
    check('''def f((a, b, c),): 3''')
    check('''def f((a, b, c, d),): 3''')
    check('''def f(x, (a)): 3''')
    check('''def f(x, (a,)): 3''')
    check('''def f(x, (a, b)): 3''')
    check('''def f(x, (a, b,)): 3''')
    check('''def f(x, (a, b, c)): 3''')
    check('''def f(x, (a, b, c, d)): 3''')
    check('''def f(x, (a),): 3''')
    check('''def f(x, (a,),): 3''')
    check('''def f(x, (a, b),): 3''')
    check('''def f(x, (a, b,),): 3''')
    check('''def f(x, (a, b, c),): 3''')
    check('''def f(x, (a, b, c, d),): 3''')
    check('''def f((a), x): 3''')
    check('''def f((a,), x): 3''')
    check('''def f((a, b), x): 3''')
    check('''def f((a, b,), x): 3''')
    check('''def f((a, b, c), x): 3''')
    check('''def f((a, b, c, d), x): 3''')

    check('''def f((a), *args): 3''')
    check('''def f((a,), *args): 3''')
    check('''def f((a, b), *args): 3''')
    check('''def f((a, b,), *args): 3''')
    check('''def f((a, b, c), *args): 3''')
    check('''def f((a, b, c, d), *args): 3''')
    check('''def f(x, (a), *args): 3''')
    check('''def f(x, (a,), *args): 3''')
    check('''def f(x, (a, b), *args): 3''')
    check('''def f(x, (a, b,), *args): 3''')
    check('''def f(x, (a, b, c), *args): 3''')
    check('''def f(x, (a, b, c, d), *args): 3''')
    check('''def f((a), x, *args): 3''')
    check('''def f((a,), x, *args): 3''')
    check('''def f((a, b), x, *args): 3''')
    check('''def f((a, b,), x, *args): 3''')
    check('''def f((a, b, c), x, *args): 3''')
    check('''def f((a, b, c, d), x, *args): 3''')
    check('''def f((a), **kwds): 3''')
    check('''def f((a,), **kwds): 3''')
    check('''def f((a, b), **kwds): 3''')
    check('''def f((a, b,), **kwds): 3''')
    check('''def f((a, b, c), **kwds): 3''')
    check('''def f((a, b, c, d), **kwds): 3''')
    check('''def f(x, (a), **kwds): 3''')
    check('''def f(x, (a,), **kwds): 3''')
    check('''def f(x, (a, b), **kwds): 3''')
    check('''def f(x, (a, b,), **kwds): 3''')
    check('''def f(x, (a, b, c), **kwds): 3''')
    check('''def f(x, (a, b, c, d), **kwds): 3''')
    check('''def f((a), x, **kwds): 3''')
    check('''def f((a,), x, **kwds): 3''')
    check('''def f((a, b), x, **kwds): 3''')
    check('''def f((a, b,), x, **kwds): 3''')
    check('''def f((a, b, c), x, **kwds): 3''')
    check('''def f((a, b, c, d), x, **kwds): 3''')
    check('''def f((a), *args, **kwds): 3''')
    check('''def f((a,), *args, **kwds): 3''')
    check('''def f((a, b), *args, **kwds): 3''')
    check('''def f((a, b,), *args, **kwds): 3''')
    check('''def f((a, b, c), *args, **kwds): 3''')
    check('''def f((a, b, c, d), *args, **kwds): 3''')
    check('''def f(x, (a), *args, **kwds): 3''')
    check('''def f(x, (a,), *args, **kwds): 3''')
    check('''def f(x, (a, b), *args, **kwds): 3''')
    check('''def f(x, (a, b,), *args, **kwds): 3''')
    check('''def f(x, (a, b, c), *args, **kwds): 3''')
    check('''def f(x, (a, b, c, d), *args, **kwds): 3''')
    check('''def f((a), x, *args, **kwds): 3''')
    check('''def f((a,), x, *args, **kwds): 3''')
    check('''def f((a, b), x, *args, **kwds): 3''')
    check('''def f((a, b,), x, *args, **kwds): 3''')
    check('''def f((a, b, c), x, *args, **kwds): 3''')
    check('''def f((a, b, c, d), x, *args, **kwds): 3''')
    check('''def f(a, (b, (c, (d))), e=2, *args, **kwds): 3''')

    check('''@a
def f(): 3''')
    check('''@a.b
def f(): 3''')
    check('''@a.b.c
def f(): 3''')
    check('''@a.b.c.d
def f(): 3''')
    check('''@a.b.c.d.e
def f(): 3''')
    check('''@a.b.c.d.e
@a.b.c.d.e
def f(): 3''')
    check('''@a()
def f(): 3''')
    check('''@a(x)
def f(): 3''')
    check('''@a(x,)
def f(): 3''')
    check('''@a(x, y)
def f(): 3''')
    check('''@a(x, y,)
def f(): 3''')
    check('''@a(x, y, z)
def f(): 3''')
    check('''@a(x, y, z,)
def f(): 3''')
    check('''@a(x, y, z, w)
def f(): 3''')
    check('''@a(x, y, z, w,)
def f(): 3''')
    check('''@a(x, y, z, w, u)
def f(): 3''')
    check('''@a(x, y, z, w, u,)
def f(): 3''')
    check('''@a(x=1)
def f(): 3''')
    check('''@a(x=1,)
def f(): 3''')
    check('''@a(x, y=1)
def f(): 3''')
    check('''@a(x, y=1,)
def f(): 3''')
    check('''@a(x, y, z=1)
def f(): 3''')
    check('''@a(x, y, z=1,)
def f(): 3''')
    check('''@a(x, y, z, w=1)
def f(): 3''')
    check('''@a(x, y, z, w=1,)
def f(): 3''')
    check('''@a(x, y, z, w, u=1)
def f(): 3''')
    check('''@a(x, y, z, w, u=1,)
def f(): 3''')
    check('''@a(x=2, y=1)
def f(): 3''')
    check('''@a(x=2, y=1,)
def f(): 3''')
    check('''@a(x, y=2, z=1)
def f(): 3''')
    check('''@a(x, y=2, z=1,)
def f(): 3''')
    check('''@a(x, y, z=2, w=1)
def f(): 3''')
    check('''@a(x, y, z=2, w=1,)
def f(): 3''')
    check('''@a(x, y, z, w=2, u=1)
def f(): 3''')
    check('''@a(x, y, z, w=2, u=1,)
def f(): 3''')
    check('''@a(*args)
def f(): 3''')
    check('''@a(x, *args)
def f(): 3''')
    check('''@a(x, y, *args)
def f(): 3''')
    check('''@a(x, y, z, *args)
def f(): 3''')
    check('''@a(x, y, z, w, *args)
def f(): 3''')
    check('''@a(x, y, z, w, u, *args)
def f(): 3''')
    check('''@a(x=1, *args)
def f(): 3''')
    check('''@a(x, y=1, *args)
def f(): 3''')
    check('''@a(x, y, z=1, *args)
def f(): 3''')
    check('''@a(x, y, z, w=1, *args)
def f(): 3''')
    check('''@a(x, y, z, w, u=1, *args)
def f(): 3''')
    check('''@a(x=2, y=1, *args)
def f(): 3''')
    check('''@a(x, y=2, z=1, *args)
def f(): 3''')
    check('''@a(x, y, z=2, w=1, *args)
def f(): 3''')
    check('''@a(x, y, z, w=2, u=1, *args)
def f(): 3''')
    check('''@a(**kwds)
def f(): 3''')
    check('''@a(x, **kwds)
def f(): 3''')
    check('''@a(x, y, **kwds)
def f(): 3''')
    check('''@a(x, y, z, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w, u, **kwds)
def f(): 3''')
    check('''@a(x=1, **kwds)
def f(): 3''')
    check('''@a(x, y=1, **kwds)
def f(): 3''')
    check('''@a(x, y, z=1, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w=1, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w, u=1, **kwds)
def f(): 3''')
    check('''@a(x=2, y=1, **kwds)
def f(): 3''')
    check('''@a(x, y=2, z=1, **kwds)
def f(): 3''')
    check('''@a(x, y, z=2, w=1, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w=2, u=1, **kwds)
def f(): 3''')
    check('''@a(*args, **kwds)
def f(): 3''')
    check('''@a(x, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w, u, *args, **kwds)
def f(): 3''')
    check('''@a(x=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w, u=1, *args, **kwds)
def f(): 3''')
    check('''@a(x=2, y=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y=2, z=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z=2, w=1, *args, **kwds)
def f(): 3''')
    check('''@a(x, y, z, w=2, u=1, *args, **kwds)
def f(): 3''')
    check('''@a(x=2, *args, y=1)
def f(): 3''')
    check('''@a(x, *args, y=2, z=1)
def f(): 3''')
    check('''@a(x, y, *args, z=2, w=1)
def f(): 3''')
    check('''@a(x, y, z, *args, w=2, u=1)
def f(): 3''')
    check('''@a(x=2, *args, y=1, **kwds)
def f(): 3''')
    check('''@a(x, *args, y=2, z=1, **kwds)
def f(): 3''')
    check('''@a(x, y, *args, z=2, w=1, **kwds)
def f(): 3''')
    check('''@a(x, y, z, *args, w=2, u=1, **kwds)
def f(): 3''')
    check('''@a(*args, y=1)
def f(): 3''')
    check('''@a(*args, y=2, z=1)
def f(): 3''')
    check('''@a(*args, z=2, w=1)
def f(): 3''')
    check('''@a(*args, w=2, u=1)
def f(): 3''')
    check('''@a(*args, y=1, **kwds)
def f(): 3''')
    check('''@a(*args, y=2, z=1, **kwds)
def f(): 3''')
    check('''@a(*args, z=2, w=1, **kwds)
def f(): 3''')
    check('''@a(*args, w=2, u=1, **kwds)
def f(): 3''')

    check('''f()''')
    check('''f(x)''')
    check('''f(x,)''')
    check('''f(x, y)''')
    check('''f(x, y,)''')
    check('''f(x, y, z)''')
    check('''f(x, y, z,)''')
    check('''f(x, y, z, w)''')
    check('''f(x, y, z, w,)''')
    check('''f(x, y, z, w, u)''')
    check('''f(x, y, z, w, u,)''')
    check('''f(x=1)''')
    check('''f(x=1,)''')
    check('''f(x, y=1)''')
    check('''f(x, y=1,)''')
    check('''f(x, y, z=1)''')
    check('''f(x, y, z=1,)''')
    check('''f(x, y, z, w=1)''')
    check('''f(x, y, z, w=1,)''')
    check('''f(x, y, z, w, u=1)''')
    check('''f(x, y, z, w, u=1,)''')
    check('''f(x=2, y=1)''')
    check('''f(x=2, y=1,)''')
    check('''f(x, y=2, z=1)''')
    check('''f(x, y=2, z=1,)''')
    check('''f(x, y, z=2, w=1)''')
    check('''f(x, y, z=2, w=1,)''')
    check('''f(x, y, z, w=2, u=1)''')
    check('''f(x, y, z, w=2, u=1,)''')
    check('''f(*args)''')
    check('''f(x, *args)''')
    check('''f(x, y, *args)''')
    check('''f(x, y, z, *args)''')
    check('''f(x, y, z, w, *args)''')
    check('''f(x, y, z, w, u, *args)''')
    check('''f(x=1, *args)''')
    check('''f(x, y=1, *args)''')
    check('''f(x, y, z=1, *args)''')
    check('''f(x, y, z, w=1, *args)''')
    check('''f(x, y, z, w, u=1, *args)''')
    check('''f(x=2, y=1, *args)''')
    check('''f(x, y=2, z=1, *args)''')
    check('''f(x, y, z=2, w=1, *args)''')
    check('''f(x, y, z, w=2, u=1, *args)''')
    check('''f(**kwds)''')
    check('''f(x, **kwds)''')
    check('''f(x, y, **kwds)''')
    check('''f(x, y, z, **kwds)''')
    check('''f(x, y, z, w, **kwds)''')
    check('''f(x, y, z, w, u, **kwds)''')
    check('''f(x=1, **kwds)''')
    check('''f(x, y=1, **kwds)''')
    check('''f(x, y, z=1, **kwds)''')
    check('''f(x, y, z, w=1, **kwds)''')
    check('''f(x, y, z, w, u=1, **kwds)''')
    check('''f(x=2, y=1, **kwds)''')
    check('''f(x, y=2, z=1, **kwds)''')
    check('''f(x, y, z=2, w=1, **kwds)''')
    check('''f(x, y, z, w=2, u=1, **kwds)''')
    check('''f(*args, **kwds)''')
    check('''f(x, *args, **kwds)''')
    check('''f(x, y, *args, **kwds)''')
    check('''f(x, y, z, *args, **kwds)''')
    check('''f(x, y, z, w, *args, **kwds)''')
    check('''f(x, y, z, w, u, *args, **kwds)''')
    check('''f(x=1, *args, **kwds)''')
    check('''f(x, y=1, *args, **kwds)''')
    check('''f(x, y, z=1, *args, **kwds)''')
    check('''f(x, y, z, w=1, *args, **kwds)''')
    check('''f(x, y, z, w, u=1, *args, **kwds)''')
    check('''f(x=2, y=1, *args, **kwds)''')
    check('''f(x, y=2, z=1, *args, **kwds)''')
    check('''f(x, y, z=2, w=1, *args, **kwds)''')
    check('''f(x, y, z, w=2, u=1, *args, **kwds)''')
    check('''f(x=2, *args, y=1)''')
    check('''f(x, *args, y=2, z=1)''')
    check('''f(x, y, *args, z=2, w=1)''')
    check('''f(x, y, z, *args, w=2, u=1)''')
    check('''f(x=2, *args, y=1, **kwds)''')
    check('''f(x, *args, y=2, z=1, **kwds)''')
    check('''f(x, y, *args, z=2, w=1, **kwds)''')
    check('''f(x, y, z, *args, w=2, u=1, **kwds)''')
    check('''f(*args, y=1)''')
    check('''f(*args, y=2, z=1)''')
    check('''f(*args, z=2, w=1)''')
    check('''f(*args, w=2, u=1)''')
    check('''f(*args, y=1, **kwds)''')
    check('''f(*args, y=2, z=1, **kwds)''')
    check('''f(*args, z=2, w=1, **kwds)''')
    check('''f(*args, w=2, u=1, **kwds)''')
    check('''f(a)(b)''')
    check('''f(a)(b)(c)''')

    check('''a''')
    check('''a.b''')
    check('''a.b.c''')
    check('''a.b.c.d''')
    check('''a.b.c.d.e''')
    check('''a.b = 3''')
    check('''a.b.c = 3''')
    check('''a.b.c.d = 3''')
    check('''a.b.c.d.e = 3''')
    check('''del a''')
    check('''del a.b''')
    check('''del a.b.c''')
    check('''del a.b.c.d''')
    check('''del a.b.c.d.e''')
    check('''a[1] = 3''')
    check('''a[1][2] = 3''')
    check('''a[1][2][3] = 3''')
    check('''a[1][2][3][4] = 3''')
    check('''del a[1]''')
    check('''del a[1][2]''')
    check('''del a[1][2][3]''')
    check('''del a[1][2][3][4]''')
    check('''(9).stuff''')
    check('''((9)).stuff''')
    check('''(((9))).stuff''')

    check('''a[1]''')
    check('''a["hey"]''')
    check('''a[1:2]''')
    check('''a[:]''')
    check('''a[1:]''')
    check('''a[:1]''')
    check('''a[::]''')
    check('''a[1::]''')
    check('''a[:1:]''')
    check('''a[::1]''')
    check('''a[1:2:]''')
    check('''a[:1:2]''')
    check('''a[1::2]''')
    check('''a[1:2:3]''')
    check('''a[...]''')
    check('''a[1,]''')
    check('''a["hey",]''')
    check('''a[1:2,]''')
    check('''a[:,]''')
    check('''a[1:,]''')
    check('''a[:1,]''')
    check('''a[::,]''')
    check('''a[1::,]''')
    check('''a[:1:,]''')
    check('''a[::1,]''')
    check('''a[1:2:,]''')
    check('''a[:1:2,]''')
    check('''a[1::2,]''')
    check('''a[1:2:3,]''')
    check('''a[...,]''')
    check('''a[1,5]''')
    check('''a["hey",5]''')
    check('''a[1:2,5]''')
    check('''a[:,5]''')
    check('''a[1:,5]''')
    check('''a[:1,5]''')
    check('''a[::,5]''')
    check('''a[1::,5]''')
    check('''a[:1:,5]''')
    check('''a[::1,5]''')
    check('''a[1:2:,5]''')
    check('''a[:1:2,5]''')
    check('''a[1::2,5]''')
    check('''a[1:2:3,5]''')
    check('''a[...,5]''')
    check('''a[1,5,]''')
    check('''a["hey",5,]''')
    check('''a[1:2,5,]''')
    check('''a[:,5,]''')
    check('''a[1:,5,]''')
    check('''a[:1,5,]''')
    check('''a[::,5,]''')
    check('''a[1::,5,]''')
    check('''a[:1:,5,]''')
    check('''a[::1,5,]''')
    check('''a[1:2:,5,]''')
    check('''a[:1:2,5,]''')
    check('''a[1::2,5,]''')
    check('''a[1:2:3,5,]''')
    check('''a[...,5,]''')
    check('''a[1,"a":"b"]''')
    check('''a["hey","a":"b"]''')
    check('''a[1:2,"a":"b"]''')
    check('''a[:,"a":"b"]''')
    check('''a[1:,"a":"b"]''')
    check('''a[:1,"a":"b"]''')
    check('''a[::,"a":"b"]''')
    check('''a[1::,"a":"b"]''')
    check('''a[:1:,"a":"b"]''')
    check('''a[::1,"a":"b"]''')
    check('''a[1:2:,"a":"b"]''')
    check('''a[:1:2,"a":"b"]''')
    check('''a[1::2,"a":"b"]''')
    check('''a[1:2:3,"a":"b"]''')
    check('''a[...,"a":"b"]''')
    check('''a[1,"a":"b",]''')
    check('''a["hey","a":"b",]''')
    check('''a[1:2,"a":"b",]''')
    check('''a[:,"a":"b",]''')
    check('''a[1:,"a":"b",]''')
    check('''a[:1,"a":"b",]''')
    check('''a[::,"a":"b",]''')
    check('''a[1::,"a":"b",]''')
    check('''a[:1:,"a":"b",]''')
    check('''a[::1,"a":"b",]''')
    check('''a[1:2:,"a":"b",]''')
    check('''a[:1:2,"a":"b",]''')
    check('''a[1::2,"a":"b",]''')
    check('''a[1:2:3,"a":"b",]''')
    check('''a[...,"a":"b",]''')
    check('''a[1,...]''')
    check('''a["hey",...]''')
    check('''a[1:2,...]''')
    check('''a[:,...]''')
    check('''a[1:,...]''')
    check('''a[:1,...]''')
    check('''a[::,...]''')
    check('''a[1::,...]''')
    check('''a[:1:,...]''')
    check('''a[::1,...]''')
    check('''a[1:2:,...]''')
    check('''a[:1:2,...]''')
    check('''a[1::2,...]''')
    check('''a[1:2:3,...]''')
    check('''a[...,...]''')
    check('''a[1,...,]''')
    check('''a["hey",...,]''')
    check('''a[1:2,...,]''')
    check('''a[:,...,]''')
    check('''a[1:,...,]''')
    check('''a[:1,...,]''')
    check('''a[::,...,]''')
    check('''a[1::,...,]''')
    check('''a[:1:,...,]''')
    check('''a[::1,...,]''')
    check('''a[1:2:,...,]''')
    check('''a[:1:2,...,]''')
    check('''a[1::2,...,]''')
    check('''a[1:2:3,...,]''')
    check('''a[...,...,]''')
    check('''a[1,5,6]''')
    check('''a["hey",5,6]''')
    check('''a[1:2,5,6]''')
    check('''a[:,5,6]''')
    check('''a[1:,5,6]''')
    check('''a[:1,5,6]''')
    check('''a[::,5,6]''')
    check('''a[1::,5,6]''')
    check('''a[:1:,5,6]''')
    check('''a[::1,5,6]''')
    check('''a[1:2:,5,6]''')
    check('''a[:1:2,5,6]''')
    check('''a[1::2,5,6]''')
    check('''a[1:2:3,5,6]''')
    check('''a[...,5,6]''')
    check('''a[1,5,6,]''')
    check('''a["hey",5,6,]''')
    check('''a[1:2,5,6,]''')
    check('''a[:,5,6,]''')
    check('''a[1:,5,6,]''')
    check('''a[:1,5,6,]''')
    check('''a[::,5,6,]''')
    check('''a[1::,5,6,]''')
    check('''a[:1:,5,6,]''')
    check('''a[::1,5,6,]''')
    check('''a[1:2:,5,6,]''')
    check('''a[:1:2,5,6,]''')
    check('''a[1::2,5,6,]''')
    check('''a[1:2:3,5,6,]''')
    check('''a[...,5,6,]''')
    check('''a[1, "hey", 2:3, 4:5:6, ..., 7]''')

    check('''a(1)[2].three''')
    check('''a[2].three''')
    check('''a.three''')
    check('''a[2]''')
    check('''a(1).three''')
    check('''a(1)''')
    check('''a(1)[2]''')
    check('''a(1).three[2]''')
    check('''a.three[2]''')
    check('''a[2](1).three''')
    check('''a[2](1)''')
    check('''a[2].three(1)''')
    check('''a.three(1)''')
    check('''a.three(1)[2]''')
    check('''a.three[2](1)''')

    check('''import x''')
    check('''import x as y''')
    check('''import x, a''')
    check('''import x as y, a''')
    check('''import x, a as b''')
    check('''import x as y, a as b''')
    check('''import x as y, a as b, c''')
    check('''import x as y, a as b, c, d''')
    check('''from z import *''')
    check('''from z import x''')
    check('''from z import x as y''')
    check('''from z import x, a''')
    check('''from z import x as y, a''')
    check('''from z import x, a as b''')
    check('''from z import x as y, a as b''')
    check('''from .z import *''')
    check('''from .z import x''')
    check('''from .z import x as y''')
    check('''from .z import x, a''')
    check('''from .z import x as y, a''')
    check('''from .z import x, a as b''')
    check('''from .z import x as y, a as b''')
    check('''from ..z import *''')
    check('''from ..z import x''')
    check('''from ..z import x as y''')
    check('''from ..z import x, a''')
    check('''from ..z import x as y, a''')
    check('''from ..z import x, a as b''')
    check('''from ..z import x as y, a as b''')
    check('''from ...z import *''')
    check('''from ...z import x''')
    check('''from ...z import x as y''')
    check('''from ...z import x, a''')
    check('''from ...z import x as y, a''')
    check('''from ...z import x, a as b''')
    check('''from ...z import x as y, a as b''')
    check('''from z.q import *''')
    check('''from z.q import x''')
    check('''from z.q import x as y''')
    check('''from z.q import x, a''')
    check('''from z.q import x as y, a''')
    check('''from z.q import x, a as b''')
    check('''from z.q import x as y, a as b''')
    check('''from .z.q import *''')
    check('''from .z.q import x''')
    check('''from .z.q import x as y''')
    check('''from .z.q import x, a''')
    check('''from .z.q import x as y, a''')
    check('''from .z.q import x, a as b''')
    check('''from .z.q import x as y, a as b''')
    check('''from ..z.q import *''')
    check('''from ..z.q import x''')
    check('''from ..z.q import x as y''')
    check('''from ..z.q import x, a''')
    check('''from ..z.q import x as y, a''')
    check('''from ..z.q import x, a as b''')
    check('''from ..z.q import x as y, a as b''')
    check('''from ...z.q import *''')
    check('''from ...z.q import x''')
    check('''from ...z.q import x as y''')
    check('''from ...z.q import x, a''')
    check('''from ...z.q import x as y, a''')
    check('''from ...z.q import x, a as b''')
    check('''from ...z.q import x as y, a as b''')
    check('''from z import (x)''')
    check('''from z import (x as y)''')
    check('''from z import (x, a)''')
    check('''from z import (x as y, a)''')
    check('''from z import (x, a as b)''')
    check('''from z import (x as y, a as b)''')
    check('''from .z import (x)''')
    check('''from .z import (x as y)''')
    check('''from .z import (x as y,)''')
    check('''from .z import (x, a)''')
    check('''from .z import (x as y, a)''')
    check('''from .z import (x as y, a,)''')
    check('''from .z import (x as y, a, b)''')
    check('''from .z import (x, a as b)''')
    check('''from .z import (x as y, a as b)''')
    check('''from ..z import (x)''')
    check('''from ..z import (x as y)''')
    check('''from ..z import (x, a)''')
    check('''from ..z import (x as y, a)''')
    check('''from ..z import (x, a as b)''')
    check('''from ..z import (x as y, a as b)''')
    check('''from ...z import (x)''')
    check('''from ...z import (x as y)''')
    check('''from ...z import (x, a)''')
    check('''from ...z import (x as y, a)''')
    check('''from ...z import (x, a as b)''')
    check('''from ...z import (x as y, a as b)''')
    check('''from z.q import (x)''')
    check('''from z.q import (x as y)''')
    check('''from z.q import (x, a)''')
    check('''from z.q import (x as y, a)''')
    check('''from z.q import (x, a as b)''')
    check('''from z.q import (x as y, a as b)''')
    check('''from .z.q import (x)''')
    check('''from .z.q import (x as y)''')
    check('''from .z.q import (x as y,)''')
    check('''from .z.q import (x, a)''')
    check('''from .z.q import (x as y, a)''')
    check('''from .z.q import (x as y, a,)''')
    check('''from .z.q import (x as y, a, b)''')
    check('''from .z.q import (x, a as b)''')
    check('''from .z.q import (x as y, a as b)''')
    check('''from ..z.q import (x)''')
    check('''from ..z.q import (x as y)''')
    check('''from ..z.q import (x, a)''')
    check('''from ..z.q import (x as y, a)''')
    check('''from ..z.q import (x, a as b)''')
    check('''from ..z.q import (x as y, a as b)''')
    check('''from ...z.q import (x)''')
    check('''from ...z.q import (x as y)''')
    check('''from ...z.q import (x, a)''')
    check('''from ...z.q import (x as y, a)''')
    check('''from ...z.q import (x, a as b)''')
    check('''from ...z.q import (x as y, a as b)''')
    check('''from . import *''')
    check('''from . import x''')
    check('''from . import x as y''')
    check('''from . import x, a''')
    check('''from . import x as y, a''')
    check('''from . import x, a as b''')
    check('''from . import x as y, a as b''')
    check('''from .. import *''')
    check('''from .. import x''')
    check('''from .. import x as y''')
    check('''from .. import x, a''')
    check('''from .. import x as y, a''')
    check('''from .. import x, a as b''')
    check('''from .. import x as y, a as b''')
    check('''from ... import *''')
    check('''from ... import x''')
    check('''from ... import x as y''')
    check('''from ... import x, a''')
    check('''from ... import x as y, a''')
    check('''from ... import x, a as b''')
    check('''from ... import x as y, a as b''')
    check('''from . import (x)''')
    check('''from . import (x as y)''')
    check('''from . import (x, a)''')
    check('''from . import (x as y, a)''')
    check('''from . import (x, a as b)''')
    check('''from . import (x as y, a as b)''')
    check('''from .. import (x)''')
    check('''from .. import (x as y)''')
    check('''from .. import (x, a)''')
    check('''from .. import (x as y, a)''')
    check('''from .. import (x, a as b)''')
    check('''from .. import (x as y, a as b)''')
    check('''from ... import (x)''')
    check('''from ... import (x as y)''')
    check('''from ... import (x, a)''')
    check('''from ... import (x as y, a)''')
    check('''from ... import (x, a as b)''')
    check('''from ... import (x as y, a as b)''')

    check('''del blah''')
    check('''del blah, blahity''')
    check('''del blah, (blahity, x, y)''')
    check('''del blah, (blahity, (x), y)''')
    check('''del blah, (blahity, ((x), y))''')

    check('''pass''')

    check('''while True: break''')
    check('''while False: continue''')
    check('''def f(): return''')
    check('''def f(x): return x''')
    check('''raise''')
    check('''raise x''')
    check('''raise x, y''')
    check('''raise x, y, z''')
    check('''def f(x): yield''')
    check('''def f(x): yield x''')

    check('''global x''')
    check('''global x, y''')
    check('''global x, y, z''')
    check('''global x, y, z, w''')

    check('''exec "3"''')
    check('''exec "3" in globals()''')
    check('''exec "3" in globals(), locals()''')

    check('''assert x''')
    check('''assert x, y''')

    check('''x += 3''')
    check('''x -= 3''')
    check('''x *= 3''')
    check('''x /= 3''')
    check('''x %= 3''')
    check('''x &= 3''')
    check('''x |= 3''')
    check('''x ^= 3''')
    check('''x <<= 3''')
    check('''x >>= 3''')
    check('''x **= 3''')
    check('''x //= 3''')

    check('''class X: pass''')
    check('''class X:
    def something(): pass''')
    check('''class X(object): pass''')
    check('''class X(A, B): pass''')
    check('''class X(A, B,): pass''')
    check('''class X(): pass''')
    check('''class X([A, B]): pass''')
    check('''@some
class X: pass''')

    check('''x += yield''')
    check('''x = yield''')
    check('''x = y = yield''')
    check('''(yield)''')
    check('''x += yield y''')
    check('''x = yield y''')
    check('''x = y = yield y''')
    check('''(yield y)''')

    check('''lambda: 3''')
    check('''lambda x: 3''')
    check('''lambda x, y=2: 3''')
    check('''lambda x, y=2, *args: 3''')
    check('''lambda x, y=2, *args, **kwds: 3''')

    check('''[x for x in y]''')
    check('''[x for x in y for a in b]''')
    check('''[x for x in y for a in b for c in d]''')
    check('''[x for x in y if x]''')
    check('''[x for x in y if x for a in b]''')
    check('''[x for x in y if x for a in b for c in d]''')
    check('''[x for x in y for a in b if b]''')
    check('''[x for x in y for a in b for c in d if d]''')
    check('''[x for x in y if x if xx]''')
    check('''[x for x in y if x if xx for a in b]''')
    check('''[x for x in y if x if xx for a in b for c in d]''')
    check('''[x for x in y if x if xx if x]''')
    check('''[x for x in y if x if xx for a in b if a]''')
    check('''[x for x in y if x if xx for a in b if b for c in d if c]''')
    check('''(x for x in y)''')
    check('''(x for x in y for a in b)''')
    check('''(x for x in y for a in b for c in d)''')
    check('''(x for x in y if x)''')
    check('''(x for x in y if x for a in b)''')
    check('''(x for x in y if x for a in b for c in d)''')
    check('''(x for x in y for a in b if a)''')
    check('''(x for x in y for a in b for c in d if c)''')
    check('''(x for x in y if x if xx)''')
    check('''(x for x in y if x if xx for a in b)''')
    check('''(x for x in y if x if xx for a in b for c in d)''')
    check('''(x for x in y if x if xx if x)''')
    check('''(x for x in y if x if xx for a in b if a)''')
    check('''(x for x in y if x if xx for a in b if a for c in d if c)''')
    check('''{x for x in y}''')
    check('''{x for x in y for a in b}''')
    check('''{x for x in y for a in b for c in d}''')
    check('''{x for x in y if x}''')
    check('''{x for x in y if x for a in b}''')
    check('''{x for x in y if x for a in b for c in d}''')
    check('''{x for x in y for a in b if b}''')
    check('''{x for x in y for a in b for c in d if d}''')
    check('''{x for x in y if x if xx}''')
    check('''{x for x in y if x if xx for a in b}''')
    check('''{x for x in y if x if xx for a in b for c in d}''')
    check('''{x for x in y if x if xx if x}''')
    check('''{x for x in y if x if xx for a in b if a}''')
    check('''{x for x in y if x if xx for a in b if b for c in d if c}''')
    check('''{x: 3 for x in y}''')
    check('''{x: 3 for x in y for a in b}''')
    check('''{x: 3 for x in y for a in b for c in d}''')
    check('''{x: 3 for x in y if x}''')
    check('''{x: 3 for x in y if x for a in b}''')
    check('''{x: 3 for x in y if x for a in b for c in d}''')
    check('''{x: 3 for x in y for a in b if b}''')
    check('''{x: 3 for x in y for a in b for c in d if d}''')
    check('''{x: 3 for x in y if x if xx}''')
    check('''{x: 3 for x in y if x if xx for a in b}''')
    check('''{x: 3 for x in y if x if xx for a in b for c in d}''')
    check('''{x: 3 for x in y if x if xx if x}''')
    check('''{x: 3 for x in y if x if xx for a in b if a}''')
    check('''{x: 3 for x in y if x if xx for a in b if b for c in d if c}''')
    check('''f(x for x in y)''')
    check('''f(x for x in y,)''')
    check('''f((x for x in y), (x for x in y))''')
    check('''f((x for x in y), (x for x in y),)''')

    check('''None''')
    check('''True''')
    check('''False''')
    check('''[x for x in lambda: True, lambda: False if x()]''')
    check('''[x for x in lambda: True, lambda: False, if x()]''')
    check('''[x for x in lambda: True, lambda: False, lambda: None if x()]''')
    check('''[x for x in lambda x, y: True, lambda: False if x()]''')
    check('''x if a else y''')

    check('''x and y''')
    check('''x and y and z''')
    check('''x and y and z and w''')
    check('''not x''')
    check('''not x and y''')
    check('''x or y''')
    check('''x or y and z''')
    check('''x or y or z''')
    check('''not x or y and z''')
    check('''x or not y and z''')
    check('''x or y and not z''')
    check('''not x or not y and z''')
    check('''not x or y and not z''')
    check('''x or not y and not z''')
    check('''not x or not y and not z''')
    check('''x and y or z''')
    check('''not x and y or z''')
    check('''x and not y or z''')
    check('''x and y or not z''')
    check('''not x and not y or z''')
    check('''not x and y or not z''')
    check('''x and not y or not z''')

    check('''x < y''')
    check('''x > y''')
    check('''x == y''')
    check('''x >= y''')
    check('''x <= y''')
    check('''x != y''')
    check('''x in y''')
    check('''x not in y''')
    check('''x is y''')
    check('''x is not y''')
    check('''1 < y < 2''')
    check('''1 < y == 2''')

    check('''(x) < y''')
    check('''(x) > y''')
    check('''(x) == y''')
    check('''(x) >= y''')
    check('''(x) <= y''')
    check('''(x) != y''')
    check('''(x) in y''')
    check('''(x) not in y''')
    check('''(x) is y''')
    check('''(x) is not y''')
    check('''(1) < y < 2''')
    check('''(1) < y == 2''')

    check('''x < (y)''')
    check('''x > (y)''')
    check('''x == (y)''')
    check('''x >= (y)''')
    check('''x <= (y)''')
    check('''x != (y)''')
    check('''x in (y)''')
    check('''x not in (y)''')
    check('''x is (y)''')
    check('''x is not (y)''')
    check('''1 < (y) < 2''')
    check('''1 < (y) == 2''')
    check('''1 < y < (2)''')
    check('''1 < y == (2)''')

    check('''(x) < (y)''')
    check('''(x) > (y)''')
    check('''(x) == (y)''')
    check('''(x) >= (y)''')
    check('''(x) <= (y)''')
    check('''(x) != (y)''')
    check('''(x) in (y)''')
    check('''(x) not in (y)''')
    check('''(x) is (y)''')
    check('''(x) is not (y)''')
    check('''(1) < (y) < 2''')
    check('''(1) < (y) == 2''')
    check('''(1) < y < (2)''')
    check('''(1) < y == (2)''')

    check('''x | y''')
    check('''x | y | z''')
    check('''x | y | z | w''')
    check('''x ^ y''')
    check('''x ^ y ^ z''')
    check('''x ^ y ^ z ^ w''')
    check('''x ^ y | z ^ w''')
    check('''x & y''')
    check('''x & y & z''')
    check('''x & y & z & w''')
    check('''x & y << z & w''')
    check('''x << y''')
    check('''x << y << z''')
    check('''x << y << z << w''')
    check('''x << y & z << w''')
    check('''x >> y''')
    check('''x >> y >> z''')
    check('''x >> y >> z >> w''')
    check('''x >> y << z >> w''')
    check('''x + y''')
    check('''x + y + z''')
    check('''x + y + z + w''')
    check('''x + y >> z + w''')
    check('''x - y''')
    check('''x - y - z''')
    check('''x - y - z - w''')
    check('''x - y + z - w''')
    check('''x * y''')
    check('''x * y * z''')
    check('''x * y * z * w''')
    check('''x * y - z * w''')
    check('''x / y''')
    check('''x / y / z''')
    check('''x / y / z / w''')
    check('''x / y * z / w''')
    check('''x % y''')
    check('''x % y % z''')
    check('''x % y % z % w''')
    check('''x % y / z % w''')
    check('''x // y''')
    check('''x // y // z''')
    check('''x // y // z // w''')
    check('''x // y % z // w''')
    check('''+x''')
    check('''-x''')
    check('''~x''')
    check('''++x''')
    check('''+-x''')
    check('''+~x''')
    check('''-+x''')
    check('''--x''')
    check('''-~x''')
    check('''~+x''')
    check('''~-x''')
    check('''~~x''')
    check('''+x + y''')
    check('''-x + y''')
    check('''~x + y''')
    check('''++x + y''')
    check('''+-x + y''')
    check('''+~x + y''')
    check('''-+x + y''')
    check('''--x + y''')
    check('''-~x + y''')
    check('''~+x + y''')
    check('''~-x + y''')
    check('''~~x + y''')
    check('''x + +x''')
    check('''x + -x''')
    check('''x + ~x''')
    check('''x + ++x''')
    check('''x + +-x''')
    check('''x + +~x''')
    check('''x + -+x''')
    check('''x + --x''')
    check('''x + -~x''')
    check('''x + ~+x''')
    check('''x + ~-x''')
    check('''x + ~~x''')
    check('''x ** y''')
    check('''x ** y ** z''')
    check('''x ** y ** z ** w''')
    check('''x ** y // z ** w''')
    check('''x() ** y''')
    check('''x() ** y ** z''')
    check('''x() ** y ** z ** w''')
    check('''x() ** y // z ** w''')
    check('''x ** y()''')
    check('''x ** y() ** z''')
    check('''x ** y() ** z ** w''')
    check('''x ** y() // z ** w''')
    check('''x ** y ** z()''')
    check('''x ** y ** z() ** w''')
    check('''x ** y // z() ** w''')
    check('''x ** y ** z ** w()''')
    check('''x ** y // z ** w()''')

    check('''`x`''')
    check('''`x, y`''')
    check('''`x, y, z`''')

    check('''print''')
    check('''print "hello"''')
    check('''print "hello",''')
    check('''print "hello", 1''')
    check('''print "hello", 1,''')
    check('''print >> HEY''')
    check('''print >> HEY, "hello"''')
    check('''print >> HEY, "hello",''')
    check('''print >> HEY, "hello", 1''')
    check('''print >> HEY, "hello", 1,''')

    check('print r"hello"')
    check('print u"hello"')
    check('print ur"hello"')
    check('print R"hello"')
    check('print U"hello"')
    check('print UR"hello"')
    check('print Ur"hello"')
    check('print uR"hello"')
    check('print b"hello"')
    check('print B"hello"')
    check('print br"hello"')
    check('print Br"hello"')
    check('print bR"hello"')
    check('print BR"hello"')
    check("print 'hello'")
    check("print r'hello'")
    check("print u'hello'")
    check("print ur'hello'")
    check("print R'hello'")
    check("print U'hello'")
    check("print UR'hello'")
    check("print Ur'hello'")
    check("print uR'hello'")
    check("print b'hello'")
    check("print B'hello'")
    check("print br'hello'")
    check("print Br'hello'")
    check("print bR'hello'")
    check("print BR'hello'")
    check('print "hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print r"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print u"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905"')
    check('print ur"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905"')
    check('print R"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print U"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905"')
    check('print UR"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905"')
    check('print Ur"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905"')
    check('print uR"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905"')
    check('print b"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print B"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print br"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print Br"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print bR"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print BR"hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo"')
    check('print \'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print r\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print u\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905lo\'')
    check('print ur\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905lo\'')
    check('print R\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print U\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905lo\'')
    check('print UR\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905lo\'')
    check('print Ur\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905lo\'')
    check('print uR\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\vlo\\N{LATIN SMALL LETTER ETH}\\u2212\\U00010905lo\'')
    check('print b\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print B\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print br\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print Br\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print bR\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')
    check('print BR\'hel\\\n\\\\\\\'\\"\\a\\b\\f\\n\\r\\t\\v\\123\\o123\\xf3lo\'')

    check('''
# This contains nearly every legal token and grammar in Python.
# Used for the python_yacc.py --self-test input.

# Might as well try the other imports here
import a
import a.b
import a.b.c
import a.b.c.d.e.f.g.h

import a as A
import a as A, b as B
import a.b as AB, a.b as B, a.b.c
''')
    check('''
from mod1 import mod2
from mod1.mod2 import mod3

from qmod import *
from qmod.qmod2 import *
from a1 import a1

from a import (xrt,
   yrt,
   zrt,
   zrt2)
from a import (xty,
   yty,
   zty,
   zrt2,)
from qwe import *
from qwe.wer import *
from qwe.wer.ert.rty import *
''')
    check('''
from .a import y
from ..a import y
from ..a import y as z
#from ...qwe import * This is not allowed
from ...a import (y as z, a as a2, t,)
from .................... import a,b,c  # 20 levels
from ...................... import (a,b,c)  # 22 levels
''')
    check('''
a = 2
a,b = 1, 222
a, = 1,
a, b, = 1, 234
a, b, c
1;
1;27
1;28;
1;29;3
1;21;33;
a.b
a.b.c
a.b.c.d.e.f
''')
    check('''
# Different number
0xDEADBEEF
0xDEADBEEFCAFE
0xDeadBeefCafe
0xDeadBeefCafeL
0xDeadBeefCafel

0o123
0o177
1.234
10E-3
1.2e+03
-1.9e+03
9j
9.8j
23.45E-9j
''')
    check('''
# 'factor' operations
a = -1
a = +1
a = ~1
b ** c
a = + + + + 1

# 'comparison' 'comp_op' expressions
a < b
a > b
a == b
a >= b
a <= b
a != b
a in b
a is b
a not in b
a is not b
''')
    check('''
# arith_expr
1 + 2
1 + 2 + 3
1 + 2 + 3 + 4
1 - 2
1 - 2 - 3
1 - 2 + 3 - 4 + 5
# 
1 - 2 + - 3 - + 4
1 + + + + + 1

# factors
a * 1
a * 1 * 2
b / 2
b / 2 / 3
c % 9
c % 9 % 7
d // 8
d // 8 // 5

a * 1 / 2 / 3 * 9 % 7 // 2 // 1
''')
    check('''
truth or dare
war and peace
this and that or that and this
a and b and c and d
x or y or z or w
not a
not not a
not not not a
not a or not b or not c
not a or not b and not c and not d

# All of the print statements
print
print "x"
print "a",
print 1, 2
print 1, 2,
print 1, 2, 93
print >>x
print >>x, 1
print >>x, 1,
print >>x, 9, 8
print >>x, 9, 8, 7
''')
    check('''
def yield_function():
    yield
    yield 1
    x = yield

@spam.qwe
def eggs():
    pass

@spam
def eggs():
    pass

@spam.qwe()
def eggs():
    pass

@spam1.qwe()
@spam2.qwe()
@spam3.qwe()
@spam3.qwe()
def eggs():
    pass

@spam(1)
def eggs():
    pass

@spam2\
(\
)
def eggs2():
    pass
''')
    check('''

@spam3\
(\
this,\
blahblabh\
)
def eggs9():
    pass

@spam\
(
**this
)
def qweqwe():
    pass

@spam.\
and_.\
eggs\
(
**this
)
def qweqwe():
    pass


spam()
spam(1)
spam(1,2)
spam(1,2,3)
spam(1,)
spam(1,2,)
spam(1,2,3,)
spam(*a)
spam(**a)
spam(*a,**b)
spam(1, *a)
spam(1, *a, **b)
spam(1, **b)
def spam(x): pass
def spam(x,): pass
def spam(a, b): pass
def spam(a, b,): pass
def spam(a, b, c): pass
def spam(a, *args): pass
def spam(a, *args, **kwargs): pass
def spam(a, **kwargs): pass
def spam(*args, **kwargs): pass
def spam(**kwargs): pass
def spam(*args): pass

def spam(x=1): pass
def spam(x=1,): pass
def spam(a=1, b=2): pass
def spam(a=1, b=2,): pass
def spam(a=1, *args): pass
def spam(a=9.1, *args, **kwargs): pass
def spam(a="", **kwargs): pass
def spam(a,b=1, *args): pass
def spam(a,b=9.1, *args, **kwargs): pass
def spam(a,b="", **kwargs): pass

def spam(a=1, b=2, *args): pass
def spam(a=1, b=2, *args, **kwargs): pass
def spam(a=1, b=2, **kwargs): pass

def spam(a=1, b=2, c=33, d=4): pass

#def spam((a) = c): pass # legal in Python 2.5, not 2.6
#def spam((((a))) = cc): pass # legal in Python 2.5, not 2.6
def spam((a,) = c): pass
def spam((a,b) = c): pass
def spam((a, (b, c)) = x96): pass
def spam((a,b,c)=x332): pass
def spam((a,b,c,d,e,f,g,h)=x12323): pass
''')
    check('''
# This show that the compiler module uses the function name location
# for the ast.Function lineno, and not the "def" reserved word.
def \
 spam \
  ( \
  ) \
  : \
  pass

a += 1
a -= 1
a *= 2
a /= 2
a %= 3
a &= 4
a |= 4
a ^= 5
a <<= 6
a >>= 7
a **= 9
a //= 10

b \
 += \
   3

a = b = c
a = b = c = d
# Shows that the ast.Assign gets the lineno from the first '='
a \
 = \
  b \
   = \
     c

a < b < c < d < e
a == b == c != d != e

a | b | c | d
a & b & c & d
a | b | c & d & e
a ^ b
a ^ b ^ c ^ d
''')
    check('''
a << 1
a << 1 << 2
a << c() << d[1]
a >> 3
a >> 6 >> 5
a >> 6 >> 5 >> 4 >> 3
a << 1 >> 2 << 3 >> 4
''')
    check('''
del a
del a,
del a, b
del a, b,
del a, b, c
del a, b, c,
del a.b
del a.b,
del a.b.c.d.e.f
del a[0]
del a[0].b
del (a, b)
del a[:5]
del a[:5,1,2,...]
del [a,b,[c,d]]
''')
    check('''
x = ()
x = (0)
x = (a,)
# x\
#  \
# =\
# (\   <-- I put the Assign line number here
# a\
# ,\   <-- Python puts the Assign line number here
# )
''')
    check('''
def spam():
    a = (yield x)

s = "this " "is "   "string " "concatenation"
s = "so " \
   "is "  \
   "this."

#for x, in ((1,),):
#    print x

for i in range(10):
    continue
for a,b in x:
    continue
for (a,b) in x:
    break
''')
    check('''
# p_trailer_3 : LSQB subscriptlist RSQB
x[0]
x[0,]
x[0:1]
x[0:1:2]
x[:3:4]
x[::6]
x[8::9]

a[...]
a[:]
b[:9]
c[:9,]
d[-4:]
a[0]**3
c[:9,:1]
q[7:,]
q[::4,]
q[:,]
t[::2]
r[1,2,3]
r[1,2,3,]
r[1,2,3,4]
r[1,2,3,4,]
t[::]
t[::,::]
t[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
p[1,2:,3:4]
p[1,2:,3:4,5:6:7,::,:9,::2, 1:2:3,
  1,2:,3:4,5:6:7,::,:9,::2, 1:2:3]


x[1] = 1
x[1:2] = 2
x[1:] = 3
x[:1] = 4
x[:] = 5
a[1,2] = 11
a[1:2,3:4] = 12

[a] = [1]
[a] = 1,
[a,b,[c,d]] = 1,2,(3,4)
''')
    check('''
# this is an 'atom'
{}

# an atom with a dictmaker
{1:2}
{1:2,}
{1:2,3:4}
{1:2,3:4,}
{1:2,3:4,5:6}
{1:2,3:4,5:6,}
{"name": "Andrew", "language": "Python", "dance": "tango"}

# Some lists
[]
[1]
[1,]
[1,2]
[1,2,]
[1,2,3,4,5,6]
[1,2,3,4,5,6,]

# List comprehensions
[1 for c in s]
[1 for c1 in s1 for c2 in s2]
[1 for c1 in s1 for c2 in s2 for c3 in s3]
[1 for c in s if c]
[(c,c2) for c in s1 if c != "n" for c2 in s2 if c2 != "a" if c2 != "e"]

[x.y for x.y in "this is legal"]

# Generator comprehensions
(1 for c in s)
(1 for c in s for d in t)

(x.y for x.y in "this is legal")
(1 for c in s if c if c+1 for d in t for e in u if d*e == c if 1 if 0)

# class definitions
class Spam:
    pass
# This shows that Python gets the line number from the 'NAME'
class \
 Spam:
    pass

class Spam: pass

class Spam(object):
    pass

class \
 Spam \
  (
   object
 ) \
 :
 pass

class Spam(): pass

class \
  Spam\
  ():
  pass
''')
    check('''
# backquotes
# Terminal "," are not supported
`1`
`1,2`
`1,2,3`
`a,b,c,d,e,f,g,h`

def a1():
    return

def a2():
    return 1,2

def a3():
    return 1,2,3

try:
    f()
except:
    pass

try:
    f()
finally:
    pass

try:
    f()
except Spam:
    a=2

try:
    f()
except (Spam, Eggs):
    a=2
''')
# This is a Python 2.6-ism
    check('''
try:
    f()
except Spam as err:
    p()

try:
    f()
except Spam, err:
    p()


try:
    f()
except Spam:
    g()
except Eggs:
    h()
except (Vikings+Marmalade), err:
    i()

try:
    a()
except Spam: b()
except Spam2: c()
finally: g()

try:
    a()
except:
    b()
else:
    c()

try: a()
except: b()
else: c()
finally: d()

try:
    raise Fail1
except:
    raise

try:
    raise Fail2, "qwer"
except:
    pass

try:
    raise Fail3, "qwer", "trw23r"
except:
    pass

try:
    raise AssertionError("raise an instance")
except:
    pass
''')
    check('''
# with statements

with x1:
  1+2
with x2 as a:
  2+3
with x3 as a.b:
  9
with x4 as a[1]:
  10
with (x5,y6) as a.b[1]:
  3+4
with x7 as (a,b):
  4+5
#with x as (a+b):  # should not work
#  5+6
with x8 as [a,b,[c.x.y,d[0]]]:
  (9).__class__
''')
    check('''
# make this 'x' and get the error "name 'x' is local and global"
# The compiler module doesn't verify this correctly.  Python does
def spam(xyz):
    global z
    z = z + 1

    global x, y
    x,y=y,z

    global a, b, c
    a,b,c = b,c,a

exec "x=1"
exec "x=1" in x
exec "z=1" in z, y
exec "raise" in {}, globals()

assert 0
assert f(x)
assert f(x), "this is not right"
assert f(x), "this is not %s" % ["left", "right"][1]
''')
    check('''
if 1:
    g()

if 1: f()

if (a+1):
    f()
    g()
    h()
    pass
else:
    pass
    a()
    b()

if a:
    z()
elif b():
    y()
elif c():
    x

if a:
    spam()
elif f()//g():
    eggs()
else:
    vikings()


while 1:
    break

while a > 1:
    a -= 1
else:
    raise AssertionError("this is a problem")

for x in s:
    1/0
for (a,b) in s:
    2/0
for (a, b.c, d[1], e[1].d.f) in (p[1], t.r.e):
    f(a)
for a in b:
    break
else:
    print "b was empty"
    print "did you hear me?"
''')
    check('''
# testlist_safe
[x for x in 1]
#[x for x in 1,]  # This isn't legal
[x for x in 1,2]
[x for x in 1,2,]
[x for x in 1,2,3]
[x for x in 1,2,3,]
[x for x in 1,2,3,4]
[x for x in 1,2,3,4,]

#[x for x in lambda :2]
#[x for x in lambda x:2*x]  # bug in compiler.transfomer prevents
#[x for x in lambda x,y:x*y]  # testing "safe" lambdas with arguments
#[x for x in lambda x,y=2:x*y]

lambda x: 5 if x else 2
[ x for x in lambda: True, lambda: False if x() ]
#[ x for x in lambda: True, lambda: False if x else 2 ]


x = 1 if a else 2
y = 1 if a else 2 if b else 3

func = lambda : 1
func2 = lambda x, y: x+y
func3 = lambda x=2, y=3: x*y

f(1)
f(1,)
f(1,2)
f(1,2,)
f(1,2,3)
f(1,2,3,)
f(1,2,3,4)
f(1,2,3,4,)
f(a=1)
f(a=1,)
f(a=1, b=2)
f(a=1, b=2,)
f(a=1, b=2, c=3)
f(a=1, b=2, c=3,)
f(9, a=1)
f(9, a=1,)
f(9, a=1, b=2)
f(9, a=1, b=2,)
f(9, 8, a=1)
f(9, 7, a=1, b=2)

f(c for c in s)
f(x=2)
f(x, y=2)
f(x, *args, **kwargs)

#f(x+y=3)

## check some line number assignments.  Python's compiler module uses
## the first term inside of the bracket/parens/braces.  I prefer the
## line number of the first character (the '[', '(', or '{')

x = [


  "a", "b",
  # comment
  "c", "d", "e", "f"


]

y = (

  c for c in s)

def f():
  welk = (



      yield
      )

d = {


  "a":
 1,
  101: 102,
  103: 104}
''')
    check('''
# Check all the different ways of escaping and counting line numbers

"""
This text
goes over
various
lines.
"""

# this should have the right line number
x_triple_quoted = 3

\'\'\'
blah blah
and
blah
\'\'\'
''')
    check('''
# this should have the right line number
y_triple_quoted = 34

r"""
when shall we three meet again
"""

# this should have the right line number
z_triple_quoted = 3235

r\'\'\'
This text
goes over
various
lines.
\'\'\'

# this should have the right line number
x_triple_quoted = 373

u"""
When in the
course of human
events
"""
''')
    check('''
# this should have the right line number
x_triple_quoted = 65

ur\'\'\'
We hold these truths to be self-evident
\'\'\'

# this should have the right line number
y_triple_quoted = 3963

''')
    check('''
# Check the escaping for the newline
s1 = r\'\'\'
  This
has a newline\
and\
a\
few
more

\'\'\'

1

s1 = ur"""
Some more \
with\
newlines

"""
''')
    check('''
str123 = 'single quoted line\
line with embedded\
newlines.'

str367 = "another \
with \
embedded\
newlines."


u"\N{LATIN SMALL LETTER ETH}"
ur"\N{LATIN SMALL LETTER ETH}"

f(1
 +
 2)


print "The end"
''')

    for fileName in sorted(glob.glob("/usr/lib/python2.7/*.py")):
        if "/_" not in fileName:
            check(open(fileName).read(), fileName)
        else:
            print >> sys.stderr, "parse only:", fileName
            grammar.parse(open(fileName).read())
