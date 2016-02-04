# Description:
# psstore
package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("/tensorflow/tensorflow", "tf_copts")

cc_library(
    name = "psstore_lib",
    srcs = glob(
        [
            "**/*.h",
            "**/*.cc",
        ],
    ),
    copts = tf_copts(),
    linkopts = [
        "-lpthread",
        "-lm",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        #"//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
        "//tensorflow/ps-lite:ps_lib",
    ],
)

cc_binary(
    name = "psstore_bin",
    srcs = ["main.cc"],
    copts = tf_copts(),
    linkopts = [
        "-lpthread",
        "-lm",
    ],
    deps = [
        ":psstore_lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
