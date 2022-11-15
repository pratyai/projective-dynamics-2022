# projective-dynamics-2022

## References
* http://www.projectivedynamics.org

## How to run

1. `cd` to `src/`.
1. If you want the `matplotlib` UI, then run `python3 -m demos.demo`.
1. If you want the `polyscope` UI, then run `python3 -m demos.demo2`.

## API guideline

### General
All the python code here are to be used as a reference implementation, prioritizing ease-of-understanding over performance. So, occasionally a somewhat suboptimal way to implement things is acceptable as long it's correct -- although there is no need to deliberately unoptimize an otherwise easy piece of code.

If the python implementation turns out to be poor even with reasonable optimization without losing its readability, we *may* choose to have a second implementation focusing on performance while keeping the first one as a reference.

### Constraints
Check the class `Constraint` in `src/constraint.py` for the interface and derivations of it for the various implementations.

Constraints are to yield the projections of the input point on some geometric manifold. So, the `Constraint` interface offers only a `project()` method to compute that projected point. Additionally it offers the per-constraint parameters `w`, `A`, and `B`.

The implementations of various constraints should live in `src/` (e.g. currently `src/spring.py` is there). In principle, constraints *should not* be aware of the global state of the system, although they can accept callables that can yield information that is only available in the global state (e.g. for spring constraint, the position of the other end of the spring is another vertex on the system).

### System
Check the class `System` in `src/system.py`.

It keeps track of various constraints, asks them for the local solutions, and computes the global solution.

---

# TODO

1. Figure out why our one dimensional test case is failing.
2. Come up with a set of unit tests to verify our code.
3. Create a class to streamline scene setup. Pass different scenes very quickly to try out different simulation scenarios.
4. Add some sort of method to the System class that ouputs "debug primitives" to pass to Polyscope.
