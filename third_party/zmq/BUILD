licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

cc_library(
    name = "zmq_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [".", "include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "zmq_lib",
    srcs = [
        "lib/libzmq.so.5.0.0",
    ],
    data = [
        "lib/libzmq.so.5.0.0",
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "zmq",
    visibility = ["//visibility:public"],
    deps = [
        ":zmq_headers",
        ":zmq_lib",
    ],
)


