import tensorflow as tf
from inspect import Parameter, Signature

def make_signature(names):
    return Signature(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in names)

class StructMeta(type):
    def __new__(cls, name, bases, clsdict):
        clsobj = super().__new__(cls, name, bases, clsdict)
        sig = make_signature(clsobj._fields)
        setattr(clsobj, '__signature__', sig)
        return clsobj

class Structures(metaclass=StructMeta):
    _fields = []
    def __init__(self, *args, **kwargs) -> None:
        bound = self.__signature__.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self, name, val)

class Descriptor:
    def __init__(self, name=None):
        self.name = name
    
    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
    
    def __delete__(self, instance):
        raise AttributeError('Cant delete')

class Typed(Descriptor):
    ty = object
    def __set__(self, obj, value):
        if not isinstance(value, self.ty):
            raise TypeError(f'Excepted {self.ty} got -> {type(value)})')
        super().__set__(obj, value)

class TensorFlowDataSet(Typed):
    ty = tf.data.Dataset

class TensorFlowTensor(Typed):
    ty = tf.Tensor
  
class Bool(Typed):
    ty = bool

class String(Typed):
    ty = str

class Integer(Typed):
    ty = int

class Float(Typed):
    ty = float