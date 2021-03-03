from torch.nn import Module
import metann
import torch
from collections import OrderedDict
from copy import deepcopy


class DependentModule(Module):
    r"""
    This module provides an extension to nn.Module by add a subset to buffers, dependents. They are similar to parameter,
    but they are registered in buffers, so that they can have grad_fn.
    This module calls DependentModule.to_dependentmodule when it is created. It turns the module and all of its
    submodules into sub class of DependentModule

    Examples::

        >>>net = Sequential(Linear(10, 5), Linear(5, 2))
        >>>DependentModule(net)
        DependentSequential(
          (0): DependentLinear(in_features=10, out_features=5, bias=True)
          (1): DependentLinear(in_features=5, out_features=2, bias=True)
        )

    .. note::

        This class change the origin module when initializing, you might use

        >>>DependentModule(deepcopy(net))

        if you want the origin model stay unchanged.

    """
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Module):
            module = cls.to_dependentmodule(args[0])
        else:
            module = super(DependentModule, cls).__new__(cls, *args, **kwargs)

        return module

    def __init__(self, *args, **kwargs):
        self._dependents = metann.utils.SubDict(self._buffers)
        self._active_dependents = metann.utils.SubDict(self._dependents)
        self._dependents_shapes = {}

    def _reinit(self):

        self._dependents = metann.utils.SubDict(self._buffers)
        self._active_dependents = metann.utils.SubDict(self._dependents)
        self._dependents_shapes = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            value = self.to_dependentmodule(value)
        super(DependentModule, self).__setattr__(name, value)

    def register_dependent(self, name, tensor):
        r"""
        register a named tensor to dependents.
        :param name: name of dependent
        :param tensor:

        Examples::

            >>>dnet = DependentModule(net)
            >>>dnet.register_dependent('some_tensor', torch.randn(3, 3))
            >>>dnet.some_tensor
            tensor([[ 0.4434,  0.9949, -0.4385],
                    [-0.5292,  0.2555,  0.7772],
                    [-0.5386,  0.6152, -0.3239]])

        """
        if '_dependents' not in self.__dict__:
            raise AttributeError(
                "cannot assign dependent parameter before MetaModule.__init__() or MetaModule._reinit() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("dependent parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("dependent parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("dependent parameter name can't be empty string \"\"")
        elif hasattr(self, name) and not name in self._dependents:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError("cannot assign '{}' object to dependent parameter '{}' "
                            "(torch Tensor or None required)"
                            .format(torch.typename(tensor), name))
        else:
            if tensor is not None:
                self._active_dependents[name] = tensor
                self._dependents_shapes[name] = tensor.shape
            else:
                self._dependents[name] = tensor

    def named_dependents(self, prefix='', recurse=True):
        r"""
        :param prefix: the prefix of the names
        :param recurse: traverse only the direct submodules of self if set to False
        :return: iterator of name, dependent pairs of self and sub modules.
        """
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = (lambda module: module._dependents.items())(module)
            for k, v in members:
                if v in memo and v is not None:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def dependents(self, recurse=True):
        r"""

        :param recurse: traverse only the direct submodules of self if set to False
        :return: iterator of dependents of self and sub modules.
        """
        for name, param in self.named_dependents(recurse=recurse):
            yield param

    def update_shapes(self):
        r"""
        update the register shape of dependents. Call this method when a dependent is initialize with None and assign
        to a tensor. **Do not** call this method when you are using built-in methods only.
        :return:
        """
        def gen():
            for name, value in self._active_dependents.items():
                if value is None:
                    if name in self._dependents_shapes:
                        yield name, self._dependents_shapes[name]
                    else:
                        continue
                else:
                    yield name, value.shape

        self._dependents_shapes = dict(gen())

    def _substitute(self, name, value):
        if name not in self._dependents:
            raise KeyError("{} is not in dependent parameters".format(name))
        elif name in self._dependents_shapes.keys() and self._dependents_shapes[name] != value.shape:
            raise ValueError("size mismatch for {}, expect {}, got {}".format(
                name, self._dependents_shapes[name], value.shape))

        self._dependents[name] = value

    def _substitute_from_params_dict(self, params_dict, prefix, strict=True):
        for name in self._dependents:
            key = prefix + name
            if strict == True:
                if key in params_dict:
                    self._substitute(name, params_dict[key])
                else:
                    raise ValueError("params_dict and interim parameters mismatch, got {}".format(key))
            elif strict == 'one way':
                if key in params_dict:
                    self._substitute(name, params_dict[key])
            else:
                if key in params_dict:
                    try:
                        self._substitute(name, params_dict[key])
                    except (KeyError, ValueError):
                        pass

    def substitute(self, named_params, strict=True):
        r"""
        Substitute self's dependents with the tensors of same name
        :param named_params: iterator of name, tensor pairs
        :param strict: forbid named_params and self._dependents mismatch if set to True. default: True
        """
        params_dict = dict(named_params)

        def load(module: DependentModule, prefix='', _strict=True):
            module._substitute_from_params_dict(params_dict, prefix, strict=_strict)
            for name, child in module._modules.items():
                load(child, prefix+name+'.', _strict=_strict)

        load(self, _strict=strict)

    def substitute_from_list(self, params):
        r"""
        Substitute from tensor list.
        :param params: iterator of tensors
        """
        named_params = ((k, v) for (k, _), v in zip(self.named_dependents(), params))
        self.substitute(named_params, strict='one way')

    def update_actives(self):
        keys = set()
        for key in self._dependents.keys():
            if isinstance(self._dependents[key], torch.Tensor):
                keys.add(key)
        self._active_dependents = metann.utils.SubDict(self._dependents, keys)

    def clear_params(self, init=False, clear_filter=lambda x: True):
        r"""
        Clear all parameters of self and register them as dependents.
        :param init: Set the values of dependents to None if set to False, otherwise keep the value of origin parameters.
        :param clear_filter: Function that return False when those modules you don't want to clear parameters are input
        """

        def clear_fn(module: DependentModule):
            if clear_filter(module):
                for name, value in module._parameters.items():
                    module._dependents[name] = value.clone().detach().requires_grad_() if value is not None else None
                module._parameters = OrderedDict()
                module.update_actives()
                module.update_shapes()
            if not init:
                for key in module._dependents:
                    module._dependents[key] = None

        self.apply(clear_fn)
        return self

    @classmethod
    def _sub_class(cls, module: Module):
        if not isinstance(module, DependentModule):
            return type("Dependent"+type(module).__name__, (DependentModule, type(module)), {})
        else:
            return type(module)

    @classmethod
    def _make_subclass(cls, module: Module):
        if not isinstance(module, cls):
            module.__class__ = type("Dependent"+type(module).__name__, (cls, type(module)), {})
            module._reinit()
        return module

    @classmethod
    def to_dependentmodule(cls, module: Module, recurse=True):
        if not recurse:
            module = cls._make_subclass(module)
        else:
            module.apply(lambda x: cls.to_dependentmodule(x, recurse=False))
        return module

    @classmethod
    def stateless(cls, module: Module):
        r"""
        transform input module into a DependentModule whose parameters are cleared.
        :param module:
        :param clear_filter:
        """
        return cls.to_dependentmodule(deepcopy(module)).clear_params()

    @classmethod
    def inplace_stateless(cls, module: Module):
        r"""
        transform input module into a DependentModule whose parameters are cleared.
        :param module:
        :param clear_filter:
        """
        return cls.to_dependentmodule(module).clear_params()

