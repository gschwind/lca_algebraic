from collections import OrderedDict
from typing import Tuple, Dict
from copy import copy

import itertools
import brightway2 as bw
from pandas import DataFrame

import sympy
from sympy import lambdify, simplify, Function
from sympy.printing.numpy import NumPyPrinter

from .base_utils import _actName, _getDb, _method_unit
from .base_utils import _getAmountOrFormula
from .base_utils import _user_functions
from .base_utils import _concurent_map
from .helpers import *
from .helpers import _isForeground
from .params import _param_registry, _completeParamValues, _fixed_params, _expanded_names_to_names, _expand_param_names
from ipywidgets import Output
from itertools import product

def register_user_function(sym, func):
    """Register a custom function with is python implementation
    Parameters
    ----------
    sym : the sympy function expression
    func : the implementation of the function

    Usage
    -----
    >>> def func_add(*args):
            returm sum(*args)
    >>>
    >>> func_add_sym = register_user_function(sympy.Function('func_add', real=True, imaginary=False), func_add)
    >>> e = sympy.Symbol('a') * func_add_sym(sympy.Symbol('b'), sympy.Symbol('c'))
    >>> sympy.srepr(e)
    "Mul(Symbol('a'), Function('func_add')(Symbol('b'), Symbol('c')))"
    """
    global _user_functions
    _user_functions[sym.name] = (sym, func)
    return sym

def user_function(sym):
    """Function decorator to register user function

    Usage
    -----
    >>> @user_function(sympy.Function('func_add', real=True, imaginary=False))
    >>> def func_add(*args):
            returm sum(*args)
    >>>
    >>> e = sympy.Symbol('a') * func_add(sympy.Symbol('b'), sympy.Symbol('c'))
    >>> sympy.srepr(e)
    "Mul(Symbol('a'), Function('func_add')(Symbol('b'), Symbol('c')))"
    """
    return lambda func: register_user_function(sym, func)

def _impact_labels():
    """Dictionnary of custom impact names
        Dict of "method tuple" => string
    """
    # Prevent reset upon auto reload in jupyter notebook
    if not '_impact_labels' in builtins.__dict__:
        builtins._impact_labels = dict()

    return builtins._impact_labels

def set_custom_impact_labels(impact_labels:Dict) :
    """ Global function to override name of impact method in graphs """
    _impact_labels().update(impact_labels)

def _multiLCA(activities, methods):
    """Simple wrapper around brightway API"""
    bw.calculation_setups['process'] = {'inv': activities, 'ia': methods}
    debug(f"Compute LCA for {len(activities)} activities and {len(methods)} methods")
    lca = bw.MultiLCA('process')
    cols = [_actName(act) for act_amount in activities for act, amount in act_amount.items()]
    return pd.DataFrame(lca.results.T, index=[method_name(method) for method in methods], columns=cols)


def multiLCA(models, methods, **params):
    """Compute LCA for a single activity and a set of methods, after settings the parameters and updating exchange amounts.
    This function does not use algebraic factorization and just calls the Brightway2 vanilla code.

    Parameters
    ----------
    model : Single activity (root model) or list of activities
    methods : Impact methods to consider
    params : Other parameters of the model
    """

    if not isinstance(models, list):
        models = [models]

    # Freeze params
    dbs = set(model[0] for model in models)
    for db in dbs :
        if _isForeground(db) :
            freezeParams(db, **params)

    activities = [{act: 1} for act in models]
    return _multiLCA(activities, methods).transpose()


def _ensure_tech_activity_proxy(act_key, target_db):
    """
        We cannot reference directly biosphere in the model, since LCA can only be applied to products
        We create a dummy activity in our DB, with same code, and single exchange of amount '1'
    """
    dbname, code = act_key
    act = _getDb(dbname).get(code)

    # Biosphere ?
    if (dbname == BIOSPHERE3_DB_NAME) or ("type" in act and act["type"] in {"emission", "natural resource"}) :

        code_to_find = code + "#asTech"

        try:
            # Already created ?
            return _getDb(target_db).get(code_to_find)
        except:
            name = act['name'] + ' # asTech'

            # Create biosphere proxy in User Db
            res = newActivity(target_db, name, act['unit'], {act: 1},
                              code=code_to_find,
                              isProxy=True) # add a this flag to distinguish this dummy activity from others
            return res
    else :
        return act

_createTechProxyForBio = _ensure_tech_activity_proxy


def _filter_param_values(params, expanded_param_names) :
    return {key : val for key, val in params.items() if key in expanded_param_names}


def _lambdify_expr_to_variable_sequence(expr):
    xid = 0
    def _walk_expr(expr, stack = None):
        nonlocal xid
        stack = []
        argn = []
        for a in expr.args:
            s = _walk_expr(a)
            stack += s
            argn.append(stack[-1][0])
        name = f"_x_{xid}_x_"
        xid += 1
        if isinstance(expr, sympy.Symbol):
            value = f'(kwargs["{expr.name}"])'
        elif isinstance(expr, sympy.Number):
            value = str(float(expr))
        elif isinstance(expr, sympy.logic.boolalg.BooleanTrue):
            value = "True"
        elif isinstance(expr, sympy.logic.boolalg.BooleanFalse):
            value = "False"
        elif isinstance(expr, sympy.Tuple):
            value = "("+",".join(argn)+")"
        elif isinstance(expr, sympy.Pow):
            value = "({}**{})".format(argn[0],argn[1])
        elif isinstance(expr, sympy.Add):
            value = "("+"+".join(argn)+")"
        elif isinstance(expr, sympy.Mul):
            value = "("+"*".join(argn)+")"
        elif isinstance(expr, sympy.StrictGreaterThan):
            value = "({}>{})".format(argn[0],argn[1])
        elif isinstance(expr, sympy.core.numbers.Pi):
            value = "np.pi"
        elif isinstance(expr, sympy.Max):
            value = "mymax("+",".join(argn)+")"
        elif isinstance(expr, sympy.ceiling):
            value = "np.ceil("+",".join(argn)+")"
        elif isinstance(expr, sympy.Abs):
            value = "np.fabs("+",".join(argn)+")"
        else:
            value = f"{expr.__class__.__name__}("+",".join(argn)+")"
        stack.append((name, value, argn))
        return stack
    return _walk_expr(expr)

def _merge_same_variable(data):
    """Merge same expresion as unique variable"""
    debug("Applying pass0: ", len(data))
    # for k, v in data:
    #     debug(k, v)
    data = copy(data)

    # index of line that use a given variable
    # variable_name : list of line number
    index = defaultdict(list)
    for i, l in enumerate(data):
        for a in l[2]:
            index[a].append(i)

    for i in range(len(data)):
            if data[i] is None:
                continue
            for j in range(i+1, len(data)):
                if data[j] is not None and data[j][1]==data[i][1]:
                    t = data[j]
                    data[j] = None
                    # replace all occurence of variable data[j][0] by data[i][0]
                    for k in index[t[0]]:
                        if data[k] is None:
                                continue
                        data[k] = (data[k][0], re.sub(t[0], data[i][0], data[k][1]))
                        # Update the index (Maybe not needed)
                        index[data[i][0]].append(k)

                    #for k in range(j+1, len(data)):
                    #    if data[k] is None:
                    #            continue
                    #    data[k] = (data[k][0], re.sub(t[0], data[i][0], data[k][1]))
    data = [x for x in data if x is not None]
    debug("Finish pass0: ", len(data))
    return data

def _inline_only_once_used_variable(data):
    """Inline variable that appear only once"""
    debug("Applying pass1: ", len(data))
    data = copy(data)
    s = " ".join([x[1] for x in data if x is not None])
    x = dict()
    for i in range(len(data)):
        if data[i] is None:
            continue
        n = data[i][0]
        f = re.findall(n, s)
        if len(f) == 1:
            t = data[i]
            # We cannot remove item from a list that we are using
            # thus just replace removed item by None
            data[i] = None
            for k in range(i+1,len(data)):
                if data[k] is None:
                    continue
                data[k] = (data[k][0],re.sub(t[0], t[1], data[k][1]))
    data = [x for x in data if x is not None]
    debug("Finish pass1: ", len(data))
    return data

def _generate_lambda_python_code(data):
    s = []
    s.append("def foo(**kwargs):\n")
    for x in data:
        if x is None:
            continue
        s.append("    "+x[0]+"="+x[1]+"\n")
    s.append("    return "+data[-1][0]+"\n")
    # for x in s:
    #     print(x)
    return s

def _custom_lambdify(expr, user_functions):
    pass0 = _lambdify_expr_to_variable_sequence(expr)
    pass1 = _merge_same_variable(pass0)
    pass2 = _inline_only_once_used_variable(pass1)
    pass3 = _generate_lambda_python_code(pass2)

    # Create locales variable for the lambda
    # copy dictionnary to avoid changes in provided inputs
    l = copy(user_functions)

    # Create globals variables for the lambda
    g = copy(globals())
    g.update(user_functions)

    exec(compile("".join(pass3), 'lamdba.tmp.py', 'exec', optimize=2), g, l)

    # Return the generated functions from updated globals
    return l["foo"], "".join(pass3)

class LambdaWithParamNames :
    """
    This class represents a compiled (lambdified) expression together with the list of requirement parameters and the source expression
    """
    _use_internal_optimization = False

    def __init__(self, exprOrDict, expanded_params=None, params=None, sobols=None):
        """ Computes a lamdda function from expression and list of expected parameters.
        you can provide either the list pf expanded parameters (full vars for enums) for the 'user' param names
        """
        printer = NumPyPrinter({
            'fully_qualified_modules': False,
            'inline': True,
            'allow_unknown_functions': True,
             'user_functions': { }
        })

        modules = [{x[0].name : x[1] for x in _user_functions.values()}, 'numpy']

        if isinstance(exprOrDict, dict) :
            # Come from JSON serialization
            obj = exprOrDict
            # Parse
            self.params = obj["params"]

            # Full names
            self.expanded_params = _expand_param_names(self.params)
            local_dict = {x[0].name: x[0] for x in _user_functions.values()}
            self.expr = parse_expr(obj["expr"], local_dict=local_dict)

            if LambdaWithParamNames._use_internal_optimization:
                user_functions = {x[0].name : x[1] for x in _user_functions.values()}
                self.lambd, self.code = _custom_lambdify(self.expr, user_functions)
            else:
                # Sympy regular lambdify
                self.lambd = lambdify(self.expanded_params, self.expr, modules, printer=printer)

            self.sobols = obj["sobols"]

        else :
            self.expr = exprOrDict
            self.params = params

            if expanded_params is None :
                expanded_params = _expand_param_names(params)
            if self.params is None :
                self.params = _expanded_names_to_names(expanded_params)

            if LambdaWithParamNames._use_internal_optimization:
                user_functions = {x[0].name : x[1] for x in _user_functions.values()}
                self.lambd, self.code = _custom_lambdify(self.expr, user_functions)
            else:
                # Sympy regular lambdify
                self.lambd = lambdify(expanded_params, self.expr, modules, printer=printer)

            self.expanded_params = expanded_params
            self.sobols = sobols

    def complete_params(self, params):

        param_length = _compute_param_length(params)

        # Check and expand enum params, add default values
        res = _completeParamValues(params, self.params)

        # Check and Expand single params and transform them to np.array
        for k, v in res.items():
            if np.isscalar(v):
                res[k] = np.tile(v, (param_length,))
            else:
                res[k] = np.asarray(v)
        return res

    def compute(self, **params):
        """Compute result value based of input parameters """
        # Filter on required parameters
        params = _filter_param_values(params, self.expanded_params)
        return self.lambd(**params)

    @staticmethod
    def use_internal_optimization(b = True):
        LambdaWithParamNames._use_internal_optimization = b

    def serialize(self) :
        return dict(
            params=self.params,
            expr=str(self.expr),
            sobols=self.sobols)

    def __repr__(self):
        return repr(self.expr)

    def _repr_latex_(self):
        return self.expr._repr_latex_()
    
    def __call__(self, *args, **kwargs):
        """Alias for self.compute"""
        return self.compute(*args, **kwargs)


MethodId = Tuple[str, str, str]


class _LCALambdaCache:
    
    # Store the database status to invalidate cache when needed
    _database_status = dict()

    # Store lambdas (model, method) -> LambdaWithParamNames
    _lambdas_cache = dict()

    # Store activity symbols Activity to sympy.Symbol
    _activity_symbols = dict()

    # Store activity expr
    _actitity_expr = dict()
    
    # Store computed LCA for "base" LCA required in expression
    _background_lca_cache = dict()

    def __new__(cls, *args, **kwargs):
        raise Exception(f"{cls.__name__} is a namespace and cannot be instanciated")

    @classmethod
    def clear_cache(cls):
        cls._lambdas_cache.clear()
        cls._activity_symbols.clear()
        cls._actitity_expr.clear()
        cls._background_lca_cache.clear()
        cls._store_db_status()
        
    @classmethod
    def _store_db_status(cls):
        """Store current database status, i.e. modified and processed timestamp"""
        for k, v in bw.databases.items():
            cls._database_status[k] = {
                'modified': v['modified'],
                'processed': v['processed']
            }
            
    @classmethod
    def _database_has_changed(cls):
        """Check if any change has occured since last update of the cache"""
        # The set of databases has changed ?
        if set(bw.databases) != set(cls._database_status):
            return True
        # Metadata has changed ?
        for k0, d0 in cls._database_status.items():
            d1 = bw.databases[k0]
            if any(d0[k] != d1[k] for k in d0):
                return True

        return False
    
    @classmethod
    def _ensure_sane_cache(cls):
        """Invalidate the cache if needed"""

        # Flush all pending data if any
        bw.databases.clean()

        if cls._database_has_changed():
            cls.clear_cache()
        
    @classmethod
    def _compute_lca(cls, acts, methods) :
        """ Compute LCA and return (act, method) => value """

        # List activities with at least one missing value
        remaining_acts_methods = set(product(acts, methods))-set(cls._background_lca_cache)
        
        # Keep unique activites
        remaining_acts = list(set(x[0] for x in remaining_acts_methods))

        if (len(remaining_acts) > 0) :
            
            # Some activity belong biosphere and cannot be computed, thus create
            # a proxy activities to compute them.
            
            bw.calculation_setups['process'] = {
                'inv': [{_ensure_tech_activity_proxy(act, act[0]): 1} for act in remaining_acts],
                'ia': methods
            }
            
            debug(f"Compute LCA for {len(remaining_acts)} activities and {len(methods)} methods")
            lca = bw.MultiLCA('process')
            
            # _ensure_tech_activity_proxy may change databases, but it does not 
            # invalidate the cache, thus store the new state of the database
            cls._store_db_status()
            
            # Set output from dataframe
            for im, met in enumerate(methods) :
                for ia, act in enumerate(remaining_acts) :
                    cls._background_lca_cache[(act, met)] = lca.results[ia, im]

        return copy(cls._background_lca_cache)
    
    @classmethod
    def get_lambda_for_models_with_methods(cls, models, methods) -> LambdaWithParamNames :
        """
        Compile list of models (=activities) and methods (=impacts) to SymPy expression,
        replacing the background activities by the values of the impacts

        :param models: List of activitiies
        :param methods: List of impact methods (tuples)
        :return: dict of (model, method) => LambdaWithParamNames
        """
        
        cls._ensure_sane_cache()  
      
        # pre-generate model expression to extract all possible _activity_symbols
        for model in models:
            cls.get_activity_expr(model)

        # Compute LCA for all background activities
        lcas = cls._compute_lca(cls._activity_symbols.keys(), methods)

        ret = dict()
        for model, method in product(models, methods) :

            if (model, method) not in cls._lambdas_cache:

                expr = cls.get_activity_expr(model)
    
                # Compute the required params
                lca_parameters = set(expr.free_symbols)-set(cls._activity_symbols.values())
                expected_names = set(map(str, lca_parameters))

                # If we expect an enum param name, we also expect the other ones : enumparam_val1 => enumparam_val1, enumparam_val2, ...
                expected_names = _expand_param_names(_expanded_names_to_names(expected_names))

                # Collect impact values for symbols of activities
                sub = {s: lcas[(a, method)] for a, s in cls._activity_symbols.items()}

                # Create lambda that replace symbols of activities by their values
                cls._lambdas_cache[(model, method)] = LambdaWithParamNames(expr.xreplace(sub), expected_names)
        
            ret[(model, method)] = cls._lambdas_cache[(model, method)]

        return ret

    @classmethod
    def get_activity_symbol(cls, activity : Activity):
        """ Transform an activity to a named symbol and keep cache of it """

        db_name, code = activity.key

        # Look in cache
        if (db_name, code) not in cls._activity_symbols:
            act = _getDb(db_name).get(code)
            slug = _slugify(act['name'])

            # Ensure unique symbols
            s = symbols(f"{slug}_{len(cls._activity_symbols)}")
            cls._activity_symbols[(db_name, code)] = s

        return cls._activity_symbols[(db_name, code)]

    @classmethod
    def _get_activity_expr(cls, act: Activity, visited : set = set()):
        res = 0

        with DbContext(act.key[0]):
            outputAmount = act.getOutputAmount()

            if not _isForeground(act["database"]) :
                # We reached a background DB ? => stop developping and create reference to activity
                return cls.get_activity_symbol(act)

            for exch in act.exchanges():

                formula = _getAmountOrFormula(exch)

                if isinstance(formula, types.FunctionType):
                    # Some amounts in EIDB are functions ... we ignore them
                    warn("WARNING: Some amounts in EIDB are functions!")
                    continue

                #  Production exchange
                if exch['input'] == exch['output']:
                    continue

                input_db, input_code = exch['input']
                sub_act = _getDb(input_db).get(input_code)

                # Background DB => reference it as a symbol
                if not _isForeground(input_db) :

                    # Add to dict of background symbols
                    act_expr = cls.get_activity_symbol(sub_act)

                # Our model : recursively it to a symbolic expression
                else:

                    visited = visited|{act}
                    if sub_act in visited :
                        raise Exception("Found recursive activities : " + ", ".join(_actName(act) for act in visited))

                    act_expr = cls._get_activity_expr(sub_act, visited)

                avoidedBurden = 1

                if exch.get('type') == 'production' and not exch.get('input') == exch.get('output') :
                    debug("Avoided burden", exch[name])
                    avoidedBurden = -1

                #debug("adding sub act : ", sub_act, formula, act_expr)

                res += formula * act_expr * avoidedBurden

            return res / outputAmount

    @classmethod
    def get_activity_expr(cls, act: Activity):
        """
        Computes a symbolic expression of the impacts for given model,
        referencing impacts of background activities and model parameters

        This expresion does not depend on impact methods, because for all
        methods, the final impact is a weighted sum of the impacts of each exchanges
        of the given method.

        (sympy_expr, dict of symbol => activity)
        :param act: Activity
        :param act_symbols: Cache of activity -> Symbol (filled during this process)
        :return: Sympy expression
        """

        if act.key not in cls._actitity_expr:
            with DbContext(act.key[0]):
                expr = cls._get_activity_expr(act)
                if isinstance(expr, float) :
                    expr = simplify(expr)
                else:
                    # Replace fixed params with their default value
                    expr = _replace_fixed_params(expr, _fixed_params().values())
            cls._actitity_expr[act.key] = expr

        return cls._actitity_expr[act.key]

    @classmethod
    def lambdify_expr(cls, model, method, expr):
        """Return the lambdified expr within the model context

        The expr must appear within the model.
        """
        
        # Ensure that all intermediate values are computed
        cls.get_lambda_for_models_with_methods([model], [method])
        
        # Compute LCA for all background activities
        lcas = cls._compute_lca(cls._activity_symbols.keys(), [method])
    
        # Compute the required params
        lca_parameters = set(expr.free_symbols)-set(cls._activity_symbols.values())
        expected_names = set(map(str, lca_parameters))

        # If we expect an enum param name, we also expect the other ones : enumparam_val1 => enumparam_val1, enumparam_val2, ...
        expected_names = _expand_param_names(_expanded_names_to_names(expected_names))

        # Collect impact values for symbols of activities
        sub = {s: lcas[(a, method)] for a, s in cls._activity_symbols.items()}

        # Create lambda that replace symbols of activities by their values
        return LambdaWithParamNames(expr.xreplace(sub), expected_names)
        

def get_lambda_for_models_with_methods(*args, **kwargs):
    return _LCALambdaCache.get_lambda_for_models_with_methods(*args, **kwargs)


def lambdify_expr(expr):
    return LambdaWithParamNames(expr, params=[param.name for param in _param_registry().values()])


def _modelToLambdas(model: ActivityExtended, methods):
    """
    Wraps _modelsToLambdas for a single model, returning a list of lambda function in the same order as methods.
    """
    with DbContext(model) :
        res = get_lambda_for_models_with_methods([model], methods)
        return list(res[(model, method)] for method in methods)


def method_name(method):
    """Return name of method, taking into account custom label set via set_custom_impact_labels(...) """
    if method in _impact_labels() :
        return _impact_labels()[method]
    return method[1] + " - " + method[2]

def _slugify(str) :
    return re.sub('[^0-9a-zA-Z]+', '_', str)

def _compute_param_length(params) :
    # Check length of parameter values
    params_length_list = [np.asarray(x).size for x in params.values() if np.asarray(x).size != 1]
    if len(params_length_list) < 2:
        return 1
    if not all([a == b for a, b in zip(params_length_list[:-1],params_length_list[1:])]):
        raise Exception("Parameters should be a single value or a list of same number of values")
    return params_length_list[0]


def _applyParams(lambd, params) :
    """
        Apply given params to lambda function.
        return either a single value of numpy vector
    """
    completed_params = lambd.complete_params(params)
    res =  lambd.compute(**completed_params)
    if isinstance(res, (list, np.ndarray)) and len(res) == 1:
        return res[0]
    return res


def _postMultiLCAAlgebric(methods, lambdas, **params):
    '''
        Compute LCA for a given set of parameters and pre-compiled lambda functions.
        This function is used by **multiLCAAlgebric**.

        Returns a dataframe.

        Parameters
        ----------
        methods: List of impact methods
        lambdas: dict of lambda:alpha
        **params : Parameters of the model
    '''

    param_length = _compute_param_length(params)

    # Init output
    res = np.zeros((len(methods), param_length), float)

    # Compute result on whole vectors of parameter samples at a time : lambdas use numpy for vector computation
    def process(args):
        imethod, lambd = args
        return (imethod, _applyParams(lambd, params))

    # Use multithread for that
    for imethod, value in _concurent_map(process, enumerate(lambdas)):
        res[imethod, :] = value

    return pd.DataFrame(
        res,
        index=[method_name(method) + "[%s]" % _method_unit(method) for method in methods]).transpose()


# Add default values for issing parameters or warn about extra params
def _filter_params(params, expected_names, model) :
    res = params.copy()

    expected_params_names = _expanded_names_to_names(expected_names)
    for expected_name in expected_params_names:
        if expected_name not in params:
            default = _param_registry()[expected_name].default
            res[expected_name] = default
            error("Missing parameter %s, replaced by default value %s" % (expected_name, default))

    for key, value in params.items():
        if not key in expected_params_names:
            del res[key]
            if model :
                error("Param %s not required for model %s" % (key, model))
    return res


def multiLCAAlgebricRaw(
        models : Activity,
        methods,
        **params):
    """
    Main parametric LCIA method : Computes LCA by expressing the foreground model as symbolic expression of background activities and parameters.
    Then, compute 'static' inventory of the referenced background activities.
    This enables a very fast recomputation of LCA with different parameters, useful for stochastic evaluation of parametrized model

    Parameters
    ----------
    models : List of model
    methods : List of methods of impact
    params : You should provide named values of all the parameters declared in the model. \
             Values can be single value or list of samples, all of the same size
    """
 
    lambdas_per_model_method = get_lambda_for_models_with_methods(models, methods)
    param_length = _compute_param_length(params)
    res = dict()
    for model, method in product(models, methods):
        with DbContext(model) :
            model_n = _actName(model)
            lambd = lambdas_per_model_method[(model, method)]
            values = _applyParams(lambd, params)
            res[(model, method)] = values
    return res


def multiLCAAlgebric(
        models,
        methods,
        raw=False, # If raw if true, returns dict of [(model, method)] -> value(s). Otherwize, returns Dataframe
        **params):
    """
    Main parametric LCIA method : Computes LCA by expressing the foreground model as symbolic expression of background activities and parameters.
    Then, compute 'static' inventory of the referenced background activities.
    This enables a very fast recomputation of LCA with different parameters, useful for stochastic evaluation of parametrized model

    Parameters
    ----------
    models :
        Single model or
        List of model or
        List of (model, alpha)
        or Dict of model:amount
        In case of several models, you cannot use list of parameters
    methods : List of methods / impacts to consider
    extract_activities : Optionnal : list of foregound or background activities. If provided, the result only integrate their contribution
    params : You should provide named values of all the parameters declared in the model. \
             Values can be single value or list of samples, all of the same size
    """
    if not isinstance(methods, list) :
        methods = [methods]

    if isinstance(models, list):
        def to_tuple(item) :
            if isinstance(item, tuple) :
                return item
            else:
                return (item, 1)
        models = dict(to_tuple(item) for item in models)
    elif not isinstance(models, dict):
        models = {models:1}

    for model in models.keys():
        dbname = model.key[0]
        with DbContext(dbname):

            # Check no params are passed for FixedParams
            for key in params:
                if key in _fixed_params() :
                    error("Param '%s' is marked as FIXED, but passed in parameters : ignored" % key)

            debug("computing : ", model)

    lambdas_per_model_method = get_lambda_for_models_with_methods(models.keys(), methods)

    param_length = _compute_param_length(params)

    res = defaultdict(dict)

    # Choose shape of the dataframe depending of size of each axe
    if param_length > 1 :
        row_axe = "params"
        if len(models) > 1 :
            if len(methods) > 1 :
                raise Exception("Only two of three dimensions (models, methods, param list) can be larger than 1")
            else:
                col_axe = "models"
        else:
            col_axe = "methods"
    else :
        col_axe = "methods"
        row_axe = "models"


    for method in methods :

        method_n = method_name(method) + "[%s]" % _method_unit(method)

        for model, alpha in models.items() :
            with DbContext(model) :

                model_n = _actName(model)

                lambd = lambdas_per_model_method[(model, method)]
                values = _applyParams(lambd, params) * alpha

                if raw :
                    res[(model, method)] = values
                else:
                    col = method_n if col_axe == "methods" else model_n

                    if row_axe == "params" :
                        res[col] = values
                    else:
                        row = method_n if row_axe == "methods" else model_n
                        res[col][row] = values

    if raw :
        return dict(res)
    else:
        return DataFrame.from_dict(res)




def _replace_fixed_params(expr, fixed_params, fixed_mode=FixedParamMode.DEFAULT) :
    """Replace fixed params with their value."""
    sub = {symbols(key): val for param in fixed_params for key, val in param.expandParams(param.stat_value(fixed_mode)).items()}
    return expr.xreplace(sub)



def _reverse_dict(dic):
    return {v: k for k, v in dic.items()}



