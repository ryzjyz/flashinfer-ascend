"""Explicit additions to the Tensor-only ACLNN contract; no runtime imports."""

from dataclasses import asdict, dataclass
import math

from flashinfer.runtime import SchemaError


@dataclass(frozen=True)
class TensorDtypeConversion:
    """A declared storage representation, never an implicit numerical cast."""

    storage_dtype: str
    native_dtype: str
    element_multiplier: int = 1

    def __post_init__(self):
        if (self.storage_dtype, self.native_dtype, self.element_multiplier) not in {
            ("int32", "int4", 8),
            ("int64", "uint64", 1),
            ("float4_e2m1fn_x2", "float4_e2m1", 2),
            ("float8_e8m0fnu", "float8_e8m0", 1),
        }:
            raise SchemaError("unsupported explicit ACLNN storage representation")


_INTEGER_WIDTHS = {name: bits for bits in (8, 16, 32, 64)
                   for name in (f"int{bits}", f"uint{bits}")}
_ATTRIBUTE_KINDS = {**dict.fromkeys(_INTEGER_WIDTHS, "int"), "bool": "bool",
                    "float32": "float", "float64": "float", "string": "str",
                    "int_array": "list_int", "float_array": "list_float", "bool_array": "list_bool",
                    "scalar": "scalar", "dtype": "type"}


@dataclass(frozen=True)
class AttributeParameter:
    """One generated ACLNN attribute, in OpDef declaration order."""

    name: str
    dtype: str
    default: object
    minimum: object = None
    maximum: object = None
    optional: bool = False

    @property
    def opdef_dtype(self):
        return _ATTRIBUTE_KINDS[self.dtype]

    def to_dict(self):
        result = asdict(self)
        if self.dtype == "scalar" and isinstance(self.default, complex):
            result["default"] = {"real": self.default.real, "imag": self.default.imag}
        if not self.optional:
            result.pop("optional")
        return result

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or self.dtype not in _ATTRIBUTE_KINDS
            or type(self.optional) is not bool
            or self.optional and self.dtype not in {"int_array", "scalar"}
        ):
            raise SchemaError("invalid ACLNN attribute declaration")
        if self.dtype not in _INTEGER_WIDTHS and self.dtype not in {"float32", "float64"} and (
            self.minimum is not None or self.maximum is not None
        ):
            raise SchemaError(
                "boolean/string ACLNN attributes cannot declare numeric bounds"
            )
        for bound in (self.minimum, self.maximum):
            if bound is not None and (
                type(bound) not in (int, float) or not math.isfinite(bound)
            ):
                raise SchemaError("ACLNN attribute bounds must be finite numbers")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise SchemaError("ACLNN attribute bounds are empty")
        if self.dtype == "scalar" and isinstance(self.default, dict) and set(self.default) == {"real", "imag"}:
            object.__setattr__(self, "default", complex(self.default["real"], self.default["imag"]))
        default = self.validate(self.default)
        if (self.dtype.endswith("_array") or self.dtype == "dtype") and default is not None:
            object.__setattr__(self, "default", default)

    def validate(self, value):
        if value is None and self.optional:
            return None
        if self.dtype.endswith("_array"):
            element = {"int_array": lambda x: type(x) is int and -(2**63) <= x < 2**63,
                       "float_array": lambda x: type(x) in (int, float) and math.isfinite(x) and abs(x) <= 3.40282347e38,
                       "bool_array": lambda x: type(x) is bool}[self.dtype]
            if not isinstance(value, (list, tuple)) or not all(map(element, value)):
                raise SchemaError("invalid ACLNN array: " + self.name)
            return tuple(map(float, value)) if self.dtype == "float_array" else tuple(value)
        if self.dtype == "scalar":
            if type(value) not in (bool, int, float, complex) or type(value) is int and not -(2**63) <= value < 2**63:
                raise SchemaError("invalid ACLNN Scalar: " + self.name)
            return value
        if self.dtype == "dtype":
            from .recipe import _TENSOR_DTYPES
            name = str(value).removeprefix("torch.")
            if name not in _TENSOR_DTYPES:
                raise SchemaError("invalid ACLNN dtype: " + name)
            return name
        if self.dtype == "string":
            if not isinstance(value, str) or "\0" in value:
                raise SchemaError(
                    "ACLNN string attribute must be a string without NUL: " + self.name
                )
            return value
        if self.dtype == "bool":
            valid = type(value) is bool
        elif self.dtype in _INTEGER_WIDTHS:
            bits = _INTEGER_WIDTHS[self.dtype]
            low, high = (0, 1 << bits) if self.dtype.startswith("u") else (-(1 << (bits-1)), 1 << (bits-1))
            valid = type(value) is int and low <= value < high
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
            if valid and self.dtype == "float32":
                valid = abs(value) <= 3.40282347e38
        if (
            not valid
            or self.minimum is not None
            and value < self.minimum
            or self.maximum is not None
            and value > self.maximum
        ):
            raise SchemaError("invalid ACLNN attribute %s=%r" % (self.name, value))
        return float(value) if self.dtype in {"float32", "float64"} else value
