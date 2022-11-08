from .constraint import Constraint

import numbers # Used for input validation
from numpy import array, identity

class Spring(Constraint):
    """
    A spring is defined by:
    1) its rest length, usually noted by the letter "L".
        * The rest length is defined as the length of a spring in state of equilibrium,
        * a.k.a. when no deformation forces are applied onto it.
        * A deformation force on a spring is any force whose one component (at least) is on the same axis
        * as the axis formed by the line connecting the two ends of the spring.
        * The distinction of "deformation force" is made so as to not take into consideration any forces
        * which only transpose the spring.
        * In the context of this software this is redundant,
        * as springs are considered to be massless & volumeless and as such cannot themselves be transposed by a force.
    2) its stiffness --which for simplification reasons will be considered constant along its length--,
    usually notated by the letter "k".
    * The stiffness of a spring expresses the spring's tendency to revert to its rest length:
    * the greater the value of k, the bigger the tendency.
    * In order for the value of k to interpret its physical meaning (i.e. the stiffness of the spring),
    * k > 0 must always be the case. k = 0 would equate the absence of the spring (as there is no tendency to revert back to L). 
    
    """
    __k = None # k := stiffness of spring
    __L = None # L := rest length of spring

    def __init__(self, k: float = 1.0, L: float = 1.0):
        if not isinstance(k, numbers.Number):
            raise TypeError("Provided k value is not a numerical value.")
        self.__k = k

        if not isinstance(L, numbers.Number):
            raise TypeError("Provided L value is not a numerical value")
        self.__L = L

    def wAA(self, w: float = 1, A: array = identity(Constraint.D)) -> array:
        # "The * operator can be used as a shorthand for np.multiply on ndarrays".
        # https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
        return w * A.T * A

    def wAB(self, q, w: float = 1, A: array = identity(Constraint.D), B: array = identity(Constraint.D)) -> array:
        return w * A.T * B