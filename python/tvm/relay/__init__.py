# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The Relay IR namespace containing the IR definition and compiler."""
from . import base
from . import ty
from . import expr
from . import env
from . import ir_pass

# Root operators
from .op import Op
from .op.reduce import *
from .op.tensor import *
from .op.transform import *
from . import nn
from . import vision
from . import image

from .scope_builder import ScopeBuilder

# Span
Span = base.Span

# Env
Environment = env.Environment

# Type
Type = ty.Type
TupleType = ty.TupleType
TensorType = ty.TensorType
Kind = ty.Kind
TypeVar = ty.TypeVar
TypeConstraint = ty.TypeConstraint
FuncType = ty.FuncType
TypeRelation = ty.TypeRelation
IncompleteType = ty.IncompleteType
scalar_type = ty.scalar_type

# Expr
Constant = expr.Constant
Tuple = expr.Tuple
Var = expr.Var
GlobalVar = expr.GlobalVar
Function = expr.Function
Call = expr.Call
Let = expr.Let
If = expr.If
TupleGetItem = expr.TupleGetItem

# helper functions
var = expr.var
const = expr.const
