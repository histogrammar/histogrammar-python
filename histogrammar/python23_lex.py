# flake8: noqa

#!/usr/bin/env python

# Written by Andrew Dalke
# Copyright (c) 2008 by Dalke Scientific, AB
# Modified by Jim Pivarski, 2016
#
# (This is the MIT License with the serial numbers scratched off and my
# name written in in crayon.  I would prefer "share and enjoy" but
# apparently that isn't a legally acceptable.)
#
# Copyright (c) 2008 Andrew Dalke <dalke@dalkescientific.com>
# Dalke Scientific Software, AB
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""PLY tokenizer for parsing Python"""

import re
import tokenize

from histogrammar.pycparser.ply import lex

# String literal from Python's Grammar/Grammar file to tokenization name
literal_to_name = {}

# List of tokens for PLY
tokens = []

# this list comes from keyword.list
# "as" is currently special, but with 2.6 it becomes a reserved keyword
#  (this is legal in pre 2.6: from UserDict import UserDict as as )
kwlist = ['and', 'as', 'assert', 'break', 'class', 'continue', 'def',
          'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
          'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
          'not', 'or', 'pass', 'print', 'raise', 'return', 'try',
          'while', 'with', 'yield']

RESERVED = {}
for literal in kwlist:
    name = literal.upper()
    RESERVED[literal] = name
    literal_to_name[literal] = name
    tokens.append(name)

# These are sorted with 3-character tokens first, then 2-character then 1.
for line in """
LEFTSHIFTEQUAL  <<=
RIGHTSHIFTEQUAL  >>=
DOUBLESTAREQUAL  **=
DOUBLESLASHEQUAL  //=

EQEQUAL ==
NOTEQUAL !=
NOTEQUAL <>
LESSEQUAL <=
LEFTSHIFT <<
GREATEREQUAL >=
RIGHTSHIFT >>
PLUSEQUAL +=
MINEQUAL -=
DOUBLESTAR **
STAREQUAL *=
DOUBLESLASH //
SLASHEQUAL /=
VBAREQUAL |=
PERCENTEQUAL %=
AMPEREQUAL &=
CIRCUMFLEXEQUAL ^=
RIGHTARROW ->

COLON :
COMMA ,
SEMI ;
PLUS +
MINUS -
STAR *
SLASH /
VBAR |
AMPER &
LESS <
GREATER >
EQUAL =
DOT .
PERCENT %
BACKQUOTE `
CIRCUMFLEX ^
TILDE ~
AT @

# The PLY parser replaces these with special functions
LPAR (
RPAR )
LBRACE {
RBRACE }
LSQB [
RSQB ]
""".splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, literal = line.split()
    literal_to_name[literal] = name
    if name not in tokens:
        tokens.append(name)  # N**2 operation, but N is small

    # Used to verify that I didn't make a typo
    # if not hasattr(tokenize, name):
    #    raise AssertionError("Unknown token name %r" % (name,))

    # Define the corresponding t_ token for PLY
    # Some of these will be overridden
    t_name = "t_" + name
    if t_name in globals():
        globals()[t_name] += "|" + re.escape(literal)
    else:
        globals()[t_name] = re.escape(literal)

# Delete temporary names
del t_name, line, name, literal

# make some changes to agree more closely with the compiler module.
# I think the compiler module is wrong for these cases
BACKWARDS_COMPATIBLE = False


def _raise_error(message, t, klass):
    lineno, lexpos, lexer = t.lineno, t.lexpos, t.lexer
    fileName = lexer.fileName

    # Switch from 1-based lineno to 0-based lineno
    geek_lineno = lineno - 1
    start_of_line = lexer.line_offsets[geek_lineno]
    end_of_line = lexer.line_offsets[geek_lineno+1]-1
    text = lexer.lexdata[start_of_line:end_of_line]
    offset = lexpos - start_of_line
    # use offset+1 because the exception is 1-based
    raise klass(message, (fileName, lineno, offset+1, text))


def raise_syntax_error(message, t):
    _raise_error(message, t, SyntaxError)


def raise_indentation_error(message, t):
    _raise_error(message, t, IndentationError)


TOKEN = lex.TOKEN

tokens = tuple(tokens) + (
    "NEWLINE",

    "NUMBER",
    "NAME",
    "WS",

    "STRING_START_TRIPLE",
    "STRING_START_SINGLE",
    "STRING_CONTINUE",
    "STRING_END",
    "STRING",

    "INDENT",
    "DEDENT",
    "ENDMARKER",
)

states = (
    ("SINGLEQ1", "exclusive"),
    ("SINGLEQ2", "exclusive"),
    ("TRIPLEQ1", "exclusive"),
    ("TRIPLEQ2", "exclusive"),
)


# I put this before t_WS so it can consume lines with only comments in them.
# This definition does not consume the newline; needed for things like
#    if 1: #comment
def t_comment(t):
    r"[ ]*\043[^\n]*"  # \043 is '#' ; otherwise PLY thinks it's an re comment
    pass

# Whitespace


def t_WS(t):
    r" [ \t\f]+ "
    value = t.value

    # A formfeed character may be present at the start of the
    # line; it will be ignored for the indentation calculations
    # above. Formfeed characters occurring elsewhere in the
    # leading whitespace have an undefined effect (for instance,
    # they may reset the space count to zero).
    value = value.rsplit("\f", 1)[-1]

    # First, tabs are replaced (from left to right) by one to eight
    # spaces such that the total number of characters up to and
    # including the replacement is a multiple of eight (this is
    # intended to be the same rule as used by Unix). The total number
    # of spaces preceding the first non-blank character then
    # determines the line's indentation. Indentation cannot be split
    # over multiple physical lines using backslashes; the whitespace
    # up to the first backslash determines the indentation.
    pos = 0
    while True:
        pos = value.find("\t")
        if pos == -1:
            break
        n = 8 - (pos % 8)
        value = value[:pos] + " "*n + value[pos+1:]

    if t.lexer.at_line_start and t.lexer.paren_count == 0:
        return t

# string continuation - ignored beyond the tokenizer level


def t_escaped_newline(t):
    r"\\\n"
    t.type = "STRING_CONTINUE"
    # Raw strings don't escape the newline
    assert not t.lexer.is_raw, "only occurs outside of quoted strings"
    t.lexer.lineno += 1


# Don't return newlines while I'm inside of ()s
def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)
    t.type = "NEWLINE"
    if t.lexer.paren_count == 0:
        return t

# The NUMBER tokens return a 2-ple of (value, original string)

# The original string can be used to get the span of the original
# token and to provide better round-tripping.

# imaginary numbers in Python are represented with floats,
#   (1j).imag is represented the same as (1.0j).imag -- with a float


@TOKEN(tokenize.Imagnumber)
def t_IMAG_NUMBER(t):
    t.type = "NUMBER"
    t.value = (float(t.value[:-1]) * 1j, t.value, t.lexer.kwds(t.lexpos))
    return t

# Then check for floats (must have a ".")


@TOKEN(tokenize.Floatnumber)
def t_FLOAT_NUMBER(t):
    t.type = "NUMBER"
    t.value = (float(t.value), t.value, t.lexer.kwds(t.lexpos))
    return t

# These are upgraded from patterns to functions so I can track the
# indentation level


@TOKEN(t_LPAR)
def t_LPAR(t):
    t.lexer.paren_count += 1
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_RPAR)
def t_RPAR(t):
    t.lexer.paren_count -= 1
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_LBRACE)
def t_LBRACE(t):
    r"\{"
    t.lexer.paren_count += 1
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_RBRACE)
def t_RBRACE(t):
    r"\}"
    t.lexer.paren_count -= 1
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_LSQB)
def t_LSQB(t):
    t.lexer.paren_count += 1
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_RSQB)
def t_RSQB(t):
    t.lexer.paren_count -= 1
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_DOT)
def t_DOT(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_COLON)
def t_COLON(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_BACKQUOTE)
def t_BACKQUOTE(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_AT)
def t_AT(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_PLUSEQUAL)
def t_PLUSEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_MINEQUAL)
def t_MINEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_STAREQUAL)
def t_STAREQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_SLASHEQUAL)
def t_SLASHEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_PERCENTEQUAL)
def t_PERCENTEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_AMPEREQUAL)
def t_AMPEREQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_VBAREQUAL)
def t_VBAREQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_CIRCUMFLEXEQUAL)
def t_CIRCUMFLEXEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_LEFTSHIFTEQUAL)
def t_LEFTSHIFTEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_RIGHTSHIFTEQUAL)
def t_RIGHTSHIFTEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_DOUBLESTAREQUAL)
def t_DOUBLESTAREQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_DOUBLESLASHEQUAL)
def t_DOUBLESLASHEQUAL(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_VBAR)
def t_VBAR(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_CIRCUMFLEX)
def t_CIRCUMFLEX(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_AMPER)
def t_AMPER(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_LEFTSHIFT)
def t_LEFTSHIFT(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_RIGHTSHIFT)
def t_RIGHTSHIFT(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_PLUS)
def t_PLUS(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_RIGHTARROW)
def t_RIGHTARROW(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_MINUS)
def t_MINUS(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


def t_DOUBLESTAR(t):
    r"\*\*"
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


def t_STAR(t):
    r"\*"
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


def t_DOUBLESLASH(t):
    r"//"
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


def t_SLASH(t):
    r"/"
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_PERCENT)
def t_PERCENT(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


@TOKEN(t_TILDE)
def t_TILDE(t):
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


def t_DOLLARNUMBER(t):
    r"\$[1-9][0-9]*"
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t


tokens = tokens + ("DOLLARNUMBER",)

# In the following I use 'long' to make the actual type match the
# results from the compiler module.  Otherwise there's no need for it.

# Python allows "0x", but in reading python-dev it looks like this was
# removed in 2.6/3.0.  I don't allow it.


def t_HEX_NUMBER(t):
    r"0[xX][0-9a-fA-F]+[lL]?"
    t.type = "NUMBER"
    value = t.value
    if value[-1] in "lL":
        value = value[:-1]
        f = long
    else:
        f = int
    t.value = (f(value, 16), t.value, t.lexer.kwds(t.lexpos))
    return t

# Python 2 allows "0o", but Python 3 doesn't.  This allows it: how to switch?


def t_OCT_NUMBER(t):
    r"0[oO]?[0-7]*[lL]?"
    t.type = "NUMBER"
    value = t.value
    if value[-1] in "lL":
        value = value[:-1]
        f = long
    else:
        f = int
    t.value = (f(value, 8), t.value, t.lexer.kwds(t.lexpos))
    return t


def t_DEC_NUMBER(t):
    r"[1-9][0-9]*[lL]?"
    t.type = "NUMBER"
    value = t.value
    if value[-1] in "lL":
        value = value[:-1]
        f = long
    else:
        f = int
    t.value = (f(value, 10), t.value, t.lexer.kwds(t.lexpos))
    return t


###################

# This is a q1: '
# This is a q2: "
# These are single quoted strings:  'this' "and" r"that"
# These are triple quoted strings:  """one""" '''two''' U'''three'''


error_message = {
    "STRING_START_TRIPLE": "EOF while scanning triple-quoted string",
    "STRING_START_SINGLE": "EOL while scanning single-quoted string",
}

# Handle "\" escapes


def t_SINGLEQ1_SINGLEQ2_TRIPLEQ1_TRIPLEQ2_escaped(t):
    r"\\(.|\n)"
    t.type = "STRING_CONTINUE"
    t.lexer.lineno += t.value.count("\n")
    return t

# Triple Q1


def t_start_triple_quoted_q1_string(t):
    r"([bB]|[uU])?[rR]?'''"
    t.lexer.push_state("TRIPLEQ1")
    t.type = "STRING_START_TRIPLE"
    if "r" in t.value or "R" in t.value:
        t.lexer.is_raw = True
    t.value = t.value.split("'", 1)[0]
    return t


def t_TRIPLEQ1_simple(t):
    r"[^'\\]+"
    t.type = "STRING_CONTINUE"
    t.lexer.lineno += t.value.count("\n")
    return t


def t_TRIPLEQ1_q1_but_not_triple(t):
    r"'(?!'')"
    t.type = "STRING_CONTINUE"
    return t


def t_TRIPLEQ1_end(t):
    r"'''"
    t.type = "STRING_END"
    t.lexer.pop_state()
    t.lexer.is_raw = False
    return t


def t_start_triple_quoted_q2_string(t):
    r'([bB]|[uU])?[rR]?"""'
    t.lexer.push_state("TRIPLEQ2")
    t.type = "STRING_START_TRIPLE"
    if "r" in t.value or "R" in t.value:
        t.lexer.is_raw = True
    t.value = t.value.split('"', 1)[0]
    return t


def t_TRIPLEQ2_simple(t):
    r'[^"\\]+'
    t.type = "STRING_CONTINUE"
    t.lexer.lineno += t.value.count("\n")
    return t


def t_TRIPLEQ2_q2_but_not_triple(t):
    r'"(?!"")'
    t.type = "STRING_CONTINUE"
    return t


def t_TRIPLEQ2_end(t):
    r'"""'
    t.type = "STRING_END"
    t.lexer.pop_state()
    t.lexer.is_raw = False
    return t


t_TRIPLEQ1_ignore = ""  # supress PLY warning
t_TRIPLEQ2_ignore = ""  # supress PLY warning


def t_TRIPLEQ1_error(t):
    raise_syntax_error()


def t_TRIPLEQ2_error(t):
    raise_syntax_error()


# Single quoted strings

def t_start_single_quoted_q1_string(t):
    r"([bB]|[uU])?[rR]?'"
    t.lexer.push_state("SINGLEQ1")
    t.type = "STRING_START_SINGLE"
    if "r" in t.value or "R" in t.value:
        t.lexer.is_raw = True
    t.value = t.value.split("'", 1)[0]
    return t


def t_SINGLEQ1_simple(t):
    r"[^'\\\n]+"
    t.type = "STRING_CONTINUE"
    return t


def t_SINGLEQ1_end(t):
    r"'"
    t.type = "STRING_END"
    t.lexer.pop_state()
    t.lexer.is_raw = False
    return t


def t_start_single_quoted_q2_string(t):
    r'([bB]|[uU])?[rR]?"'
    t.lexer.push_state("SINGLEQ2")
    t.type = "STRING_START_SINGLE"
    if "r" in t.value or "R" in t.value:
        t.lexer.is_raw = True
    t.value = t.value.split('"', 1)[0]
    return t


def t_SINGLEQ2_simple(t):
    r'[^"\\\n]+'
    t.type = "STRING_CONTINUE"
    return t


def t_SINGLEQ2_end(t):
    r'"'
    t.type = "STRING_END"
    t.lexer.pop_state()
    t.lexer.is_raw = False
    return t


t_SINGLEQ1_ignore = ""  # supress PLY warning
t_SINGLEQ2_ignore = ""  # supress PLY warning


def t_SINGLEQ1_error(t):
    raise_syntax_error("EOL while scanning single quoted string", t)


def t_SINGLEQ2_error(t):
    raise_syntax_error("EOL while scanning single quoted string", t)

###


# This goes after the strings otherwise r"" is seen as the NAME("r")
def t_NAME(t):
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    t.type = RESERVED.get(t.value, "NAME")
    t.value = (t.value, t.lexer.kwds(t.lexpos))
    return t

########


def _new_token(type, lineno):
    tok = lex.LexToken()
    tok.type = type
    tok.value = None
    tok.lineno = lineno
    tok.lexpos = -100
    return tok

# Synthesize a DEDENT tag


def DEDENT(lineno):
    return _new_token("DEDENT", lineno)

# Synthesize an INDENT tag


def INDENT(lineno):
    return _new_token("INDENT", lineno)

###


def t_error(t):
    raise_syntax_error("invalid syntax", t)


_lexer = lex.lex()


def _parse_quoted_string(start_tok, string_toks):
    # The four combinations are:
    #  "ur"  - raw_uncode_escape
    #  "u"   - uncode_escape
    #  "r"   - no need to do anything
    #  ""    - string_escape
    #  "br"  - no need to do anything
    #  "b"   - string_escape
    s = "".join(tok.value for tok in string_toks)
    quote_type = start_tok.value.lower()
    if quote_type == "":
        return s.decode("string_escape")
    elif quote_type == "r":
        return s
    elif quote_type == "u":
        return s.decode("unicode_escape")
    elif quote_type == "ur":
        return s.decode("raw_unicode_escape")
    elif quote_type == "b":
        return s.decode("string_escape")
    elif quote_type == "br":
        return s
    else:
        raise AssertionError("Unknown string quote type: %r" % (quote_type,))


def create_strings(lexer, token_stream):
    for tok in token_stream:
        if not tok.type.startswith("STRING_START_"):
            yield tok
            continue

        # This is a string start; process until string end
        start_tok = tok
        string_toks = []
        for tok in token_stream:
            if tok.type == "STRING_END":
                break
            else:
                assert tok.type == "STRING_CONTINUE", tok.type
                string_toks.append(tok)
        else:
            # Reached end of input without string termination
            # This reports the start of the line causing the problem.
            # Python reports the end.  I like mine better.
            raise_syntax_error(error_message[start_tok.type], start_tok)

        # Reached the end of the string
        if BACKWARDS_COMPATIBLE and "SINGLE" in start_tok.type:
            # The compiler module uses the end of the single quoted
            # string to determine the strings line number.  I prefer
            # the start of the string.
            start_tok.lineno = tok.lineno
        start_tok.type = "STRING"

        pos = start_tok.lexer.kwds(start_tok.lexpos)
        start_tok.value = (_parse_quoted_string(start_tok, string_toks), pos)
        yield start_tok


# Keep track of indentation state

# I implemented INDENT / DEDENT generation as a post-processing filter

# The original lex token stream contains WS and NEWLINE characters.
# WS will only occur before any other tokens on a line.

# I have three filters.  One tags tokens by adding two attributes.
# "must_indent" is True if the token must be indented from the
# previous code.  The other is "at_line_start" which is True for WS
# and the first non-WS/non-NEWLINE on a line.  It flags the check so
# see if the new line has changed indication level.

# Python's syntax has three INDENT states
#  0) no colon hence no need to indent
#  1) "if 1: go()" - simple statements have a COLON but no need for an indent
#  2) "if 1:\n  go()" - complex statements have a COLON NEWLINE and must indent
NO_INDENT = 0
MAY_INDENT = 1
MUST_INDENT = 2

# only care about whitespace at the start of a line


def annotate_indentation_state(lexer, token_stream):
    lexer.at_line_start = at_line_start = True
    indent = NO_INDENT
    saw_colon = False
    for token in token_stream:
        token.at_line_start = at_line_start

        if token.type == "COLON":
            at_line_start = False
            indent = MAY_INDENT
            token.must_indent = False

        elif token.type == "NEWLINE":
            at_line_start = True
            if indent == MAY_INDENT:
                indent = MUST_INDENT
            token.must_indent = False

        elif token.type == "WS":
            assert token.at_line_start == True
            at_line_start = True
            token.must_indent = False

        else:
            # A real token; only indent after COLON NEWLINE
            if indent == MUST_INDENT:
                token.must_indent = True
            else:
                token.must_indent = False
            at_line_start = False
            indent = NO_INDENT

        yield token
        lexer.at_line_start = at_line_start


# Track the indentation level and emit the right INDENT / DEDENT events.
def synthesize_indentation_tokens(token_stream):
    # A stack of indentation levels; will never pop item 0
    levels = [0]
    token = None
    depth = 0
    prev_was_ws = False
    for token in token_stream:
        # WS only occurs at the start of the line
        # There may be WS followed by NEWLINE so
        # only track the depth here.  Don't indent/dedent
        # until there's something real.
        if token.type == "WS":
            assert depth == 0
            depth = len(token.value)
            prev_was_ws = True
            # WS tokens are never passed to the parser
            continue

        if token.type == "NEWLINE":
            depth = 0
            if prev_was_ws or token.at_line_start:
                # ignore blank lines
                continue
            # pass the other cases on through
            yield token
            continue

        # then it must be a real token (not WS, not NEWLINE)
        # which can affect the indentation level

        prev_was_ws = False
        if token.must_indent:
            # The current depth must be larger than the previous level
            if not (depth > levels[-1]):
                raise_indentation_error("expected an indented block", token)

            levels.append(depth)
            yield INDENT(token.lineno)

        elif token.at_line_start:
            # Must be on the same level or one of the previous levels
            if depth == levels[-1]:
                # At the same level
                pass
            elif depth > levels[-1]:
                # indentation increase but not in new block
                raise_indentation_error("unexpected indent", token)
            else:
                # Back up; but only if it matches a previous level
                try:
                    i = levels.index(depth)
                except ValueError:
                    # I report the error position at the start of the
                    # token.  Python reports it at the end.  I prefer mine.
                    raise_indentation_error(
                        "unindent does not match any outer indentation level", token)
                for _ in range(i+1, len(levels)):
                    yield DEDENT(token.lineno)
                    levels.pop()

        yield token

    ### Finished processing ###

    # Must dedent any remaining levels
    if len(levels) > 1:
        assert token is not None
        for _ in range(1, len(levels)):
            yield DEDENT(token.lineno)


def add_endmarker(token_stream):
    tok = None
    for tok in token_stream:
        yield tok
    if tok is not None:
        lineno = tok.lineno
    else:
        lineno = 1
    yield _new_token("ENDMARKER", lineno)


_add_endmarker = add_endmarker


def make_token_stream(lexer, add_endmarker=True):
    token_stream = iter(lexer.token, None)
    token_stream = create_strings(lexer, token_stream)
    token_stream = annotate_indentation_state(lexer, token_stream)
    token_stream = synthesize_indentation_tokens(token_stream)
    if add_endmarker:
        token_stream = _add_endmarker(token_stream)
    return token_stream


_newline_pattern = re.compile(r"\r?\n")


def get_line_offsets(text):
    offsets = [0]
    for m in _newline_pattern.finditer(text):
        offsets.append(m.end())
    # This is only really needed if the input does not end with a newline
    offsets.append(len(text))
    return offsets


class PythonLexer(object):
    def __init__(self, lexer=None, fileName="<string>"):
        if lexer is None:
            lexer = _lexer.clone()
        self.lexer = lexer
        self.lexer.paren_count = 0
        self.lexer.is_raw = False
        self.lexer.fileName = fileName
        self.token_stream = None

    def input(self, data, add_endmarker=True):
        self.lexer.source = data
        self.lexer.input(data)
        self.lexer.paren_count = 0
        self.lexer.is_raw = False
        self.lexer.line_offsets = get_line_offsets(data)

        def kwds(pos):
            i = 0
            for i in xrange(len(self.lexer.line_offsets) - 1):
                if self.lexer.line_offsets[i + 1] > pos:
                    break
            return {"lineno": i + 1, "col_offset": pos - self.lexer.line_offsets[i]}
        self.lexer.kwds = kwds
        self.token_stream = make_token_stream(self.lexer, add_endmarker=True)

    def token(self):
        try:
            return self.token_stream.next()
        except StopIteration:
            return None

    def __iter__(self):
        return self.token_stream
