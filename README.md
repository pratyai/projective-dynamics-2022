# projective-dynamics-2022

## References
* http://www.projectivedynamics.org

## API guideline

### General
All the python code here are to be used as a reference implementation, prioritizing ease-of-understanding over performance. So, occasionally a somewhat suboptimal way to implement things is acceptable as long it's correct -- although there is no need to deliberately unoptimize an otherwise easy piece of code.

If the python implementation turns out to be poor even with reasonable optimization without losing its readability, we *may* choose to have a second implementation focusing on performance while keeping the first one as a reference.

### Constraints
Check the class `Constraint` in `src/constraints/constraint.py` for the interface and derivations of it for the various implementations.

While constraints are to yield the projections of the input point on some geometric manifold, in practice we don't necessarily need the points themselves to use in the global solver. It is sufficient to get the values of

* $wA^TA$ (which depends on the constraint's type and parameters, but *not* the point to be projected)
* $wA^TBp$ (which depends on the constraint's type and parameters, but *also* the projected point, which might be a non-linear mapping of the point to be projected)

for each constraint. Which is why the `Constraint` interface class offers only those to methods.

The implementations of various constraints should live in `src/constraints/` (e.g. currently `src/constraints/spring.py` is there). In principle, constraints *should not* be aware of the global state of the system.

### System
Check the class `System` in `src/system.py`.

It keeps track of various constraints, asks them for the local solutions, and computes the global solution.
