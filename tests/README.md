# Tests for OpenTurbine's CI process

OpenTurbine seeks to provide a comprehensive test suite to aid in development and establish credability with our users.
These tests are to be run frequently during code development and will be required to pass CI before merging on GitHub.
The tests are organized into the following groups:

- **unit_tests** Unit tests are small tests that attempt to test a single behavior in isolation.  Some functions may be hard to sufficiently unit test, but even those can usually be broken into smaller parts wich themselves can be tested.  These tests should always be checked against analytically derrived results.

- **regression_tests** Regression tests test the interaction of several, more complex parts of the code.  These tests can more easily target realistic and complex behaviors as well as functions that require significant amounts of state, but they also fail to provide total confidence in any single part of the code.  These tests are usually checked against "gold" values previously obtained from running the test and validated against other, known accurate, solutions.

- **documentation_tests** Documentation tests are stand-alone executables which function as tutorials for how to use OpenTurbine to set up problems.  The comments in these tests are much more extensive than in other tests and are meant to be used by OpenTurbine users to inform their application of our APIs.
