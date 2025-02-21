load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "cluster_coordinator",
    srcs = ["cluster_coordinator.py"],
    srcs_version = "PY3",
    deps = [
        ":coordinator_context",
        ":metric_utils",
        ":utils",
        ":values",
        ":watchdog",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:executor",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:remote",
        "@six_archive//:six",
    ],
)

py_library(
    name = "coordinator_context",
    srcs = [
        "coordinator_context.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
    ],
)

distribute_py_test(
    name = "cluster_coordinator_test",
    srcs = ["cluster_coordinator_test.py"],
    python_version = "PY3",
    shard_count = 50,
    tags = [
        "multi_gpu",
        "no_oss",  # TODO(b/214432000): Very flaky under Docker
        "no_pip",
        "noasan",  # TODO(b/171040359): Flaky timeout, even if maximum shards
        "notpu",
        "notsan",  # TODO(b/171040359): Flaky timeout, even if maximum shards
    ],
    xla_tags = [
        "no_cuda_asan",  # Race condition on async test
    ],
    deps = [
        ":cluster_coordinator",
        ":values",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "fault_tolerance_test",
    srcs = ["fault_tolerance_test.py"],
    python_version = "PY3",
    shard_count = 36,
    tags = [
        "noasan",  # Multi-process runner does not work with test sanitizers
        "nomac",  # TODO(b/177065434)
        "notsan",  # Multi-process runner does not work with test sanitizers
    ],
    deps = [
        ":cluster_coordinator",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/training:training_lib",
    ],
)

py_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:monitoring",
    ],
)

tf_py_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":cluster_coordinator",
        ":metric_utils",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "remote_eager_lib",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "watchdog",
    srcs = ["watchdog.py"],
    srcs_version = "PY3",
    deps = [],
)

tf_py_test(
    name = "watchdog_test",
    srcs = ["watchdog_test.py"],
    python_version = "PY3",
    deps = [
        ":watchdog",
        "//tensorflow/python/eager:test",
    ],
)
