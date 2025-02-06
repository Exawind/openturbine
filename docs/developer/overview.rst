.. _overview:

Overview
========

Developers of OpenTurbine should understand the workflows described here.
Adherence to these processes ensures high quality code with minimal maintenance.
The development process involves orchestating a few different systems
and people so that work is not duplicated or conflicting. The primary
objective of these processes is communication and signaling to
other developers and ourselves now and in the future.

.. contents::

Version control
--------------

OpenTurbine uses git with a basic version of git-flow. The ``main`` branch is
stable, contains release-ready code, and is updated least frequently. The
``develop`` branch contains tested and well developed code. It is the staging
branch for the next release. New work happens on feature branches prefixed
with ``feature/`` or ``f/``, and these are merged into ``develop``. Bug fixes are
on branches prefixed with ``bug/`` or ``b/``, and these can be merged into
``develop`` or ``main`` depending on the severity of the bug.

The commit history should serve as a record of changes and include good
contextual information. When staging changes and writing a commit message,
ask yourself "What will this mean to me in two years?" Indeed, there is a
real possibility that well after you've lost your current context you or
your colleague will read through your commits in an attempt to understand
what you were thinking right now. Do your future self the favor of
authoring a long lasting commit message with a well-scoped set of changes.

Some tips:

- Each commit should encompass one well-scoped change. A smell test is
  whether you've used the word "and" in the commit message. For example,
  a commit message of "Add a new CLI flag for XYZ and improve speed via
  ABC" should probably be split into two commits, "Add new CLI flag for
  XYZ" and "Improve speed via ABC".
- The sequence of commits can be as meaningful as the individual changes.
  Construct the commit history so that someone can move through each
  commit in a series and understand why and how the changes were made
  in sequence.
- Practice editing a branch's commit history with interactive
  rebase: ``git rebase -i``.

Pull requests
------------

The driving question for each pull request should be "How can I
convince reviewers that this pull request should be merged?"

Pull requests should include a well-scoped set of changes. It is
often reasonable to split a single branch into multiple pull
requests so that each PR does a single thing. The pull request
should have a complete description of the change as well as its
impact on the rest of the code. It is helpful to include diagrams
of the new code and how it fits into the existing architecture.
New code should include tests to demonstrate verification of
the methodology and robustness of the implementation.

Reviewers should be very verbose in their comments and feedback. If there
is a question, it should be asked without hesitation. This builds a searchable
body of information that can ultimately be compiled into more formal
documentation or serve as an informal reference on GitHub itself.

Developer workflow
----------------

Prior to starting work on a new project, a Discussion should be created
to describe the scope of work and design the implementation. The original
post (OP) should serve as the primary document and comments should be added
by reviewers. Typically, only the original author will modify the OP, but
all contributors can participate in the design process with comments. Once
the concept is well defined, an Issue should be opened with the final
description from the Discussion.

Issues allow work to be tracked in Project boards. Issues are generally a
bug report or a feature request. In addition to the information from the
Design Discussion, issues should include anticipated time frame and developer
assignments. This is a great place to coordinate a team of developers as well
as link issues to other issues to understand dependencies.

Finally, when the work is complete, a pull request should be created and
reviewers assigned.

Adding unit tests
----------------

All pull requests should be maximally tested in the unit test suite. This
includes new code as well as existing code that is modified. To add a new
unit test, first decide if the tests will consist of a new group of tests
or whether it logically fits into an existing group. GTest groups tests
into "test suites" during test declaration:

.. code-block:: cpp

    TEST(TestSuiteName, TestName) {
        test body
    }

A test suite contains multiple tests and the unit test runner can
filter by test suite or test name. New tests should be organized so that
they organizationally reference the source code and logically fit
into an appropriate test suite.

Add a new test to an existing test suite by simply creating a new
test block with an appropriate test name. Be sure to document
the test and any relevant background.

A new test suite requires registering with CMake. First, create a
new C++ file to contain the test suite. Then, add it to the list
of sources for ``${oturb_unit_test_exe_name}`` in
``tests/unit_tests/CMakeLists.txt``.

The test suite can be listed from the runner with the following
command:

.. code-block:: bash

    ./openturbine_unit_tests --gtest_list_tests

Checklist for code contributions
------------------------------

Ensure that all of these steps are complete prior to submitting a pull request
as ready for review.

- Context for the code changes (validation)
- Description of the code changes including how they relate to the
  project architecture and design
- Proof of correctness (verification)
- Tests for all changes included in the pull request included in the
  automated tests
- Documentation for changes
- Documentation for high level references that are impacted; for example,
  installation instructions, API reference, or user guides
