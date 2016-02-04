licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

load("/tensorflow/tensorflow", "tf_copts")
load("/tensorflow/core/platform/default/build_config", "tf_proto_library")

cc_library(
    name = "ps_headers",
    hdrs = glob([
        "include/*.h",
    ]),
    includes = [".", "include"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(["src/meta.proto"]),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,  # TODO(irving): Handle 3
    visibility = ["//visibility:public"],
)

cc_library(
    name = "protos_cc",
    copts = tf_copts(),
    deps = [
        ":protos_all_cc",
    ],
)

cc_library(
    name = "ps_lib",
    srcs = glob([
        "src/*.h",
        "src/*.cc",
    ]),
    copts = tf_copts(),
    linkopts = ["-lpthread"],
    visibility = ["//visibility:public"],
    deps = [
        ":ps_headers",
        ":protos_cc",
        "//third_party/zmq:zmq",
    ],
    alwayslink = 1,
)
