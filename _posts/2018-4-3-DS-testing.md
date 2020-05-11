---
layout: post
title: Testing in data science
---

Testing code in data science isn't always as straightforward as with software development. In this post we'll outline some tools, techniques, and tips to avoid being woken up in the middle of the night from a pager.

## Unit Tests
Unit tests are low-level tests designed to test isolated units of software that perform some logic. They consist of testing individual methods and functions of the classes, components, or modules used by our software. The most critical factor of a unit test is that it does not require a configured environment, any other services, a network connection, nothing. A unit test suite should ideally take on the order of 10s of seconds, and single tests: fractions of a second. Whenever we're unit testing code that calls out to a service like ES, we can use something like mocking, patching, or dependency injection so that we only test the logic and not the interaction between the external service. 

Cases we'd want to unit test: 
* Massive amounts of supporting data transformations are used in order to get data in and out of packaged ML algorithms. Make sure these transformations are tested and modularized if re-used extensively
* Beware that if we join data from a table at training and serving time, the data in the table may change so we can think of edge cases might want to add to our tests
* Specifying a schema for a dataset can make sure that the numbers aren't off from what we expect. Are column types as expected? Are null values allowed? Is the distribution of values acceptable? For example, if human height was a feature, we'd want to assert that the data is between 1 and 10 feet.

## Functional Tests
Functional tests are used to test code interactions with an external service or other functions/modules/libraries. A functional test can execute a focused interaction with a service with an instance of that service running. Another important factor to consider in creating a functional test is that it should only depend on the single service being tested, all other services must be mocked. Some examples may be: querying AWS for some data and then applying some transformations to it or creating some functions to build a query to query Elastic Search. The logic around the data transformations and query builder should be unit tested, but in order to make sure they are interacting with the services correctly, we can create functional tests. 

Cases we'd want to write functional tests: 
* Code interactions with external services like databases and APIs. Ex) do we get back the response we expect?
* Code interactions with other modules, libraries or functions. Ex) do two functions work together as we'd expect?
* To verify model training works properly we can train for a couple steps/iterations on a small subset of data and verify the loss decreases.

## End-to-end Tests
An end to end test is built around a workflow, and very importantly involves no mocking (but should use fixtures!). To improve E2E test reliability, E2E tests should have all of their code built off pieces exercised by individual functional tests where possible, rather than building custom test code simply for the end to end test. Data scientists can use E2E tests to check that a data pipeline is functioning as expected from data gathering to model output/predictions. This can be done with a small dataset to keep the time required to run tests at a minimum. This can also help with reproducibility - running a pipeline once would ideally produce the same results as running it again with the same data. For reproducibility keep in mind that it's possible to set random number generator seeds in order to control for less deterministic models and sort data at steps where order matters.

## Pytest
In order to implement the three tests described above, the Python library [pytest](https://docs.pytest.org/en/latest/contents.html) provides a lot of great functionality. To summarize, here are some usage patterns and concepts to focus on:
* Adhere to [pytest's good-practices](https://docs.pytest.org/en/latest/goodpractices.html) and create a test folder that mirrors the source folder we're testing. Any file (and function within that file) can be named with `test_*` pytest will run if we invoke `pytest` inside that directory. If we just want to run one function we can do something like `pytest test_something.py::test_func`.
* To help debug a failing test we can set up breakpoints in the IDE or use the python built in `breakpoint()` function for Python 3 or `import pdb; pdb.set_trace()` for Python 2. Note: if we print/log to the console during a test, it will not actually print the output unless the test fails. There are ways to change this behavior such as using the [caplog fixture](https://docs.pytest.org/en/latest/logging.html#caplog-fixture).
* Whenever we're creating a test that requires running the same function multiple times with different inputs, we can use the [@pytest.mark.parametrize](https://docs.pytest.org/en/latest/parametrize.html) decorator.
* If there is an object or functionality that is required for multiple tests to run we can initialize it with a fixture and use the [@pytest.fixture decorator](https://docs.pytest.org/en/latest/fixture.html). A `conftest.py` file allows us to share fixtures across different files.
* Pytest provides [monkeypatch](https://docs.pytest.org/en/latest/monkeypatch.html) for mocking
* For testing operations that read/write data to files, we should create [temporary directories](https://docs.pytest.org/en/latest/tmpdir.html) where possible.
* For testing non-deterministic outputs, it may be worth looking at [flaky](https://docs.pytest.org/en/latest/flaky.html) tests as a last resort so that the test is automatically retried a specified number of times.

## Acceptance Testing
Is our model good enough to be deployed? Acceptance tests are formal tests executed to verify if a system satisfies its business requirements. They require the entire application to be up and running and focuses on replicating user behaviors. They can also go further and measure the performance of the system and reject changes if certain goals are not met. Here we'd want to make sure that offline model proxies or business metrics such as user engagement correlate with model performance metrics such as accuracy.

## Monitoring
In order to make sure that code is performing as expected while deployed, we can set up hooks to monitor parts of our code or model through a tool like Datadog or Sumo Logic. In general, try to quantify desirable and undesirable behaviors and track those metrics.

Some metrics we might want to monitor include:
* Automated alerting if we see the distribution of production/serving data starting to diverge from the distribution of the data a model was trained on. 
* Since the data being served is different from the data a model was trained on, we can log a sample of actual serving data, train/predict offline and show performance periodically. We can then start re-training on this newer data as well to avoid model staleness
* Memory usage and time to train/predict which should be constant over time

Python also has the [logging](https://docs.python.org/3/library/logging.html) module which is helpful to know how to use. A model pipeline should have logs for debugging where a user can observe the step by step computation of ETL, training, or inference. Coupled with monitoring, this will facilitate investigation if issues are reported. Basic usage would be to initialize it at the top of our python module using `import logging; logging.basicConfig(level=logging.INFO); _LOGGER = logging.getLogger(__name__)` and use it like `_LOGGER.info("some log")`.

## Writing testable code and other tips
* To facilitate testing we might have a python file which has a couple of public functions and within those, all we do is make calls to private functions implemented in that module. We can then write unit tests for each of the private functions and functional tests for the public functions to make sure the private functions interact properly with each other.
* Before fixing a bug, write a test that exposes that bug. Once it's fixed, the test should pass and that bug will be less likely to show up again
* To assert Numpy arrays are equal, use the Numpy testing library. Similarly for Pandas DataFrames, use the Pandas testing library
* Search github for terms like `pytest, mock, conftest, fixture`, etc to see examples of these concepts can be used in Python. It's also useful to look at external libraries' source code that are well maintained, to find other best practices.
* If we're deploying a model or application, we'll want to make sure that it behaves the same way when it runs locally compared to when it's deployed. A good way to do this is creating a docker image that can be used in both environments. We can use `docker-compose` to define setup arguments and environment variables for the container.
* Re­use code between training and serving pipelines whenever possible to avoid digressions and make code easier to test.

Go prevent those sleepless nights! Feedback is always welcome.

