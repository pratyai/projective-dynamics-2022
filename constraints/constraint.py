import numbers # Used for input validation
from numpy import array, identity

class Constraint:
    """
    The 'Constraint' class is a mixin;
    * The purpose of a mixin --or 'trait' as it is called in other languages, like Scala--
    * is to enrich the API (fields & methods) of any class which inherit the mixin class.
    * However, a mixin is not a base class in itself. Alternatively expressed, the creation of an independent mixin object is non-sensical for the software.
    The framework provided by Projective Dynamics decouples 'distance measure' and 'manifold' by arriving at the equation:
    W(q, p) = d(q, p) + δE(p), where
    - q := is a point in geometric space
    - p := is a point on the constraint manifold
    - W : R^n x R^n -> R is a penalty function called 'potential energy function'
    - d : R^n x R^n -> R is a (quadratic) distance measure function.
        Quadratic distance measure: w/2||Aq - Bp||^2, where
                                    a) w is a non-negative real weight.
                                    b) A and B are constant matrices which express the peculiarity/physical properties of the constraint.
                                    * 1) Notice how when taking the derivative of the distance measure,
                                    * the 2 is cancelled out by the 1/2.
                                    * 2) Since matrices A and B are constant, they can be precomputed!

    - δE : R^n -> {0, +∞} is a function which decides whether the input position is strictly on or outside of the manifold.
        Notice that when the provided p is outside the manifold, then the value punishment function W becomes +∞, which
        --provided that we wish to solve a minimization problem-- automatically rejects the solution.
    * - the constant 'n' in the above definitions express the dimensionality of the problem.
    *   Typically, for real world applications, n = 3.
    

    """

    # dimensions of the geometric space
    D = 3

    # For now copying Pratyai's logic.
    def wAA(self, w: float = 1.0, A: array = identity(D)) -> array:
        '''
        Return w_i * A_i' * A_i for this constraint, excluding the selection term S_i.
        '''
        if not isinstance(w, numbers.Number):
            raise TypeError(f"Weight 'w' was expected to be of {numbers.Number} type. Type of the object provided is {type(w)}.")
        
        raise NotImplementedError("wAA() of the constraint has not been unimplemented.")

    # For now copying Pratyai's logic.
    def wABp(self, q: array, w: float = 1.0, A: array = identity(D), B: array = identity(D)) -> array:
        '''
        Return w_i * A_i' * B_i * p for this constraint, excluding the selection term S_i.
        q := a single point of the object (expressed in world coordinates) which will be projected unto this constraint.
        '''
        if not isinstance(w, numbers.Number):
            raise TypeError(f"Weight 'w' was expected to be of {numbers.Number} type. Type of the object provided is {type(w)}.")
        
        raise NotImplementedError("wAB() of the constraint has not been unimplemented.")

    def _auxiliary_variable_p_of_q(self, q:array) -> array:
        """
        Inputs:
        - q := point of object which is subject this constraint.
        Output:
        - auxiliary variable p of this constraint according to which q shall be projected.
        The auxiliary variable p is dependent on two things:
        1) The nature of the constraint. Thus, it is correct to define this function inside the 'Constraint' class
        ? Is it correct?
        2) The reference point q which will be projected unto the manifold.
            This is an external input and depends solely on the state of the system at a point in time.
        """
        raise NotImplementedError("'_auxiliary_variable_p_of_q()' of the constraint has not been implemented.")

    def project_on_constraint(self, q:array) -> array:
        """
        Inputs:
        - q := point of object which is subject to this constraint.
        Output:
        - New position of q after being projected on the manifold.
        """

        # TODO : Call "_auxiliary_variable_p_of_q(q)" internally to get the p according to the nature of the constraint.
        # TODO: After the previous step, invoke the projection according to the implementation required by the constraint.

        raise NotImplementedError("'project_on_constraint()' of the constraint has not been implemented.")


